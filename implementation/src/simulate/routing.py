"""Decision rules: RecServe's escalation test and the energy-aware routing rule.

Tiers are referred to by their index in TIERS (0 = user, 1 = ONU, 2 = OLT); `rates` maps
a tier's name to its (J per prompt token, J per generated token).
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import bisect
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from energy.three_tier import TIERS

TOP = len(TIERS) - 1
"""Index of the highest tier (the OLT)."""


class Window:
    """RecServe's confidence history for one tier: the last `size` confidences, kept sorted.

    A tier escalates a query when its answer's confidence is below quantile(beta) of this
    window, so the threshold follows the tier's own recent answers and beta sets roughly
    the share of queries it passes on.
    """

    def __init__(self, size: int):
        """Create an empty window.

        Args:
            size: How many recent confidences to keep (RecServe recommends 300-1000).
        """
        self.size = size
        self.recent = collections.deque()
        """The confidences in arrival order, to know which one to drop."""

        self.ordered = []
        """The same confidences, sorted, to read quantiles."""

    def fill(self, confidences) -> None:
        """Start the window from confidences shipped with the model instead of empty.

        Args:
            confidences: At most `size` confidences, oldest first.
        """
        self.recent = collections.deque(confidences)
        self.ordered = sorted(self.recent)

    def __len__(self):
        """Return how many confidences the window holds now."""
        return len(self.recent)

    def add(self, confidence: float) -> None:
        """Record one more confidence, dropping the oldest once the window is full.

        Args:
            confidence: The confidence of the answer the tier just produced.
        """
        self.recent.append(confidence)
        bisect.insort(self.ordered, confidence)
        if len(self.recent) > self.size:
            oldest = self.recent.popleft()
            del self.ordered[bisect.bisect_left(self.ordered, oldest)]

    def quantile(self, p: float) -> float:
        """Return the p-quantile, interpolated between order statistics as numpy.percentile does.

        Args:
            p: The quantile, between 0 and 1 (RecServe's beta).

        Returns:
            The confidence threshold.
        """
        position = p * (len(self.ordered) - 1)
        below = int(position)
        above = min(below + 1, len(self.ordered) - 1)
        return self.ordered[below] + (self.ordered[above] - self.ordered[below]) * (
            position - below
        )


class Learned:
    """What one household's tier knows about itself and the tiers above it.

    Three ways to know the answer lengths (--stats):
      household  learned from the packets that ride back down on this household's own
                 answers, as exponentially weighted moving averages (EWMA);
      pooled     the same, but one pool shared by every household (a stand-in for values
                 set in advance from the whole population);
      published  shipped with each tier's model and then replaced by the averages each
                 tier publishes about its own answers (OwnStats), which adopt() takes.
    The escalation rates are RecServe's beta (--escalation beta) or learned like the
    answer lengths (--escalation learned). Energy rates are always the household's own.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(
        self,
        alpha: float,
        warmup: int,
        rate_alpha: float | None = None,
        pool: Learned | None = None,
        escalation: float | None = None,
    ):
        """Start with nothing learned.

        Args:
            alpha: EWMA weight for answer lengths and escalation rates.
            warmup: Packets per tier needed before the household decides.
            rate_alpha: EWMA weight for reported energy rates (default: alpha).
            pool: Shared question statistics to use instead of this household's own.
            escalation: Every tier's escalation rate, fixed (RecServe's beta); None to
                learn it from the packets.
        """
        self.alpha, self.warmup = alpha, warmup
        self.rate_alpha = alpha if rate_alpha is None else rate_alpha
        self.escalation = escalation

        self.rates: dict[str, tuple[float, float]] = {}
        """Each tier's reported (J/prompt token, J/generated token), as heard by this household."""

        if pool is None:
            self.tokens: dict[str, float] = {}
            """How many tokens each tier generates per answer."""

            self.p_further: dict[str, float] = (
                {} if escalation is None else dict.fromkeys(TIERS, escalation)
            )
            """How often a query that reached each tier went further up."""

            self.heard: collections.Counter = collections.Counter()
            """How many packets from each tier have been absorbed."""
        else:
            self.tokens, self.p_further, self.heard = pool.tokens, pool.p_further, pool.heard

    def ship(self, tokens: dict, beta: float) -> None:
        """Start from values shipped with the models, so no warm-up is needed for them.

        Args:
            tokens: {tier: tokens generated per answer}, as measured for each tier's model.
            beta: RecServe's escalation quantile, the escalation rate its test gives by
                construction, used until a tier publishes its own.
        """
        for tier, value in tokens.items():
            self.tokens[tier] = value
            self.p_further.setdefault(tier, beta)
            self.heard[tier] = max(self.heard[tier], self.warmup)

    # ==========================================
    # Learning
    # ==========================================
    def absorb(self, packet: dict, answered_at: str) -> None:
        """Learn from one answer's packet.

        Args:
            packet: {tier: (J/prompt token, J/generated token, tokens generated)}.
            answered_at: The tier whose answer was kept; every tier below it passed the
                query on, which updates p_further.
        """
        for tier, (j_prompt, j_generated, generated) in packet.items():
            self._ewma(self.tokens, tier, generated)
            if self.escalation is None:
                went_further = float(TIERS.index(answered_at) > TIERS.index(tier))
                self._ewma(self.p_further, tier, went_further)
            self._hear_rates(tier, j_prompt, j_generated)
            self.heard[tier] += 1

    def adopt(self, packet: dict) -> None:
        """Take the averages each tier published about its own answers (--stats published).

        Args:
            packet: {tier: (J/prompt token, J/generated token, the tier's OwnStats)}.
        """
        for tier, (j_prompt, j_generated, own) in packet.items():
            self.tokens[tier] = own.tokens
            if self.escalation is None and own.p_further is not None:
                self.p_further[tier] = own.p_further
            self._hear_rates(tier, j_prompt, j_generated)
            self.heard[tier] = max(self.heard[tier], own.answers)

    def _hear_rates(self, tier: str, j_prompt: float, j_generated: float) -> None:
        """Move a tier's believed energy rates towards the ones just reported."""
        old = self.rates.get(tier)
        a = self.rate_alpha
        self.rates[tier] = (
            (j_prompt, j_generated)
            if old is None
            else ((1 - a) * old[0] + a * j_prompt, (1 - a) * old[1] + a * j_generated)
        )

    def _ewma(self, averages: dict, key: str, x: float) -> None:
        """Set averages[key] to the EWMA of averages[key] and x (x itself the first time)."""
        averages[key] = (
            x if key not in averages else (1 - self.alpha) * averages[key] + self.alpha * x
        )

    def ready(self, tiers) -> bool:
        """Return whether every one of these tiers has reported at least `warmup` times.

        Args:
            tiers: The tier names the decision would use.
        """
        return all(self.heard[t] >= self.warmup for t in tiers)


class OwnStats:  # pylint: disable=too-few-public-methods
    """What one tier knows about its own answers, published on them with --stats published.

    The OLT has one for the whole PON, since it answers every household; the phone and the
    ONU have one per household. A tier records every answer it produces, so it learns its
    averages far faster than a household that only hears the answers it asked for.
    """

    def __init__(self, alpha: float, shipped_tokens: float):
        """Start from the answer length measured for the tier's model.

        Args:
            alpha: EWMA weight for the answer lengths and escalation rate.
            shipped_tokens: Tokens per answer measured for the model before deployment.
        """
        self.alpha = alpha
        self.tokens = shipped_tokens
        """How many tokens the tier generates per answer."""

        self.p_further: float | None = None
        """How often its answers escalate; None for the top tier, which never escalates."""

        self.answers = 0
        """How many answers it has produced."""

    def record(self, generated: float, escalated: bool | None) -> None:
        """Record one answer.

        Args:
            generated: Tokens the answer took.
            escalated: Whether RecServe's test passed the query on (None at the top tier).
        """
        self.tokens = _ewma_step(self.tokens, generated, self.alpha)
        if escalated is not None:
            self.p_further = _ewma_step(self.p_further, float(escalated), self.alpha)
        self.answers += 1


def _ewma_step(average: float | None, value: float, alpha: float) -> float:
    """Return the EWMA after one more value (the value itself the first time)."""
    return value if average is None else (1 - alpha) * average + alpha * value


# ==========================================
# Routing rule
# ==========================================
def answer_energy(tier: str, prompt_tokens: float, rates: dict, learned: Learned) -> float:
    """Return the expected joules for a tier to answer.

    Args:
        tier: The tier's name.
        prompt_tokens: The query's prompt length.
        rates: The J per token each tier is believed to cost now.
        learned: The household's knowledge (answer lengths).

    Returns:
        The tier's rates times the prompt and the learned answer length.
    """
    j_prompt, j_generated = rates[tier]
    return j_prompt * prompt_tokens + j_generated * learned.tokens[tier]


def _forward_choice(here: int, costs: dict, run_energy: float, p_here: float, delta: float) -> int:
    """Return the tier above that tier `here` forwards to on arrival, or `here` to run it.

    Running here costs this tier's energy plus, if it escalates, the cheapest way on. The
    query is forwarded only if a tier above is cheaper by more than the margin delta.
    """
    cheapest = min(costs, key=costs.get)
    run_here = run_energy + p_here * costs[cheapest]
    return cheapest if run_here - costs[cheapest] > delta * run_here else here


def _escalation_choice(here: int, costs: dict, delta: float) -> int:
    """Return the tier an escalated query goes to: the next one up, or a cheaper later one."""
    cheapest = min(costs, key=costs.get)
    following = here + 1
    return cheapest if costs[following] - costs[cheapest] > delta * costs[following] else following


def _plan(
    lowest: int, prompt_tokens: float, rates: dict, learned: Learned, delta: float
) -> tuple[dict, dict]:
    """Return C_j and where a query handed to tier j ends up running, for every tier from `lowest` up.

    Computed from the top down, so each C_j follows what tier j itself would do with the
    query: C_J = E_J, C_j = C_k if tier j forwards it on arrival to k, and otherwise
    C_j = E_j + p_j * C_n, with n where tier j sends it if its answer escalates. A tier that
    would forward a query is passed straight through, since the query reaches the tier it
    forwards to either way.

    Args:
        lowest: Index of the lowest tier to plan for.
        prompt_tokens: The query's prompt length.
        rates: The J per token each tier is believed to cost now.
        learned: The household's knowledge (answer lengths, escalation rates).
        delta: Relative margin a forward or a skip must save.

    Returns:
        ({tier index: expected joules until answered}, {tier index: index of the tier that runs it}).
    """
    costs = {TOP: answer_energy(TIERS[TOP], prompt_tokens, rates, learned)}
    runs_at = {TOP: TOP}
    for index in range(TOP - 1, lowest - 1, -1):
        tier = TIERS[index]
        above = {k: costs[k] for k in range(index + 1, TOP + 1)}
        run_energy = answer_energy(tier, prompt_tokens, rates, learned)
        forward = _forward_choice(index, above, run_energy, learned.p_further[tier], delta)
        if forward != index:
            costs[index], runs_at[index] = costs[forward], runs_at[forward]
        else:
            following = _escalation_choice(index, above, delta)
            costs[index] = run_energy + learned.p_further[tier] * costs[following]
            runs_at[index] = index
    return costs, runs_at


def cost_to_completion(
    start: int, prompt_tokens: float, rates: dict, learned: Learned, delta: float
) -> float:
    """Return the expected joules from handing the query to tier `start` until it is answered.

    Every tier from `start` up is assumed to decide as the policy does (see _plan). The
    policy decides where the OLT's rate in `rates` comes from.

    Args:
        start: Index of the tier that gets the query.
        prompt_tokens: The query's prompt length.
        rates: The J per token each tier is believed to cost now.
        learned: The household's knowledge (answer lengths, escalation rates).
        delta: Relative margin a forward or a skip must save.

    Returns:
        The expected energy, in joules.
    """
    return _plan(start, prompt_tokens, rates, learned, delta)[0][start]


def on_arrival(here: int, prompt_tokens: float, rates: dict, learned: Learned, delta: float) -> int:
    """Decide whether to run the query here or forward it straight to a tier above.

    Args:
        here: Index of the tier the query is at.
        prompt_tokens: The query's prompt length.
        rates: The J per token each tier is believed to cost now.
        learned: The household's knowledge.
        delta: Relative margin a forward must save.

    Returns:
        The index of the tier to run the query at (`here` to run it here).
    """
    if here == TOP:
        return here
    return _plan(here, prompt_tokens, rates, learned, delta)[1][here]


def on_escalation(
    here: int, prompt_tokens: float, rates: dict, learned: Learned, delta: float
) -> int:
    """Decide where an escalated query goes: the next tier up, or a later, cheaper one.

    Args:
        here: Index of the tier that escalated.
        prompt_tokens: The query's prompt length.
        rates: The J per token each tier is believed to cost now.
        learned: The household's knowledge.
        delta: Relative margin a skip must save.

    Returns:
        The index of the tier that will run it next.
    """
    costs, runs_at = _plan(here + 1, prompt_tokens, rates, learned, delta)
    return runs_at[_escalation_choice(here, costs, delta)]
