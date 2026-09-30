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

    Learned from the packets that ride back down on answers, as exponentially weighted
    moving averages (EWMA). Answer lengths, escalation rates and packet counts describe the
    questions, not the household, so with --shared-stats they are one pool shared by every
    household; energy rates are always the household's own.
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
    ):
        """Start with nothing learned.

        Args:
            alpha: EWMA weight for answer lengths and escalation rates.
            warmup: Packets per tier needed before the household decides.
            rate_alpha: EWMA weight for reported energy rates (default: alpha).
            pool: Shared question statistics to use instead of this household's own.
        """
        self.alpha, self.warmup = alpha, warmup
        self.rate_alpha = alpha if rate_alpha is None else rate_alpha

        self.rates: dict[str, tuple[float, float]] = {}
        """Each tier's reported (J/prompt token, J/generated token), as heard by this household."""

        if pool is None:
            self.tokens: dict[str, float] = {}
            """How many tokens each tier generates per answer."""

            self.p_further: dict[str, float] = {}
            """How often a query that reached each tier went further up."""

            self.heard: collections.Counter = collections.Counter()
            """How many packets from each tier have been absorbed."""
        else:
            self.tokens, self.p_further, self.heard = pool.tokens, pool.p_further, pool.heard

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
            went_further = float(TIERS.index(answered_at) > TIERS.index(tier))
            self._ewma(self.tokens, tier, generated)
            self._ewma(self.p_further, tier, went_further)
            old = self.rates.get(tier)
            a = self.rate_alpha
            self.rates[tier] = (
                (j_prompt, j_generated)
                if old is None
                else ((1 - a) * old[0] + a * j_prompt, (1 - a) * old[1] + a * j_generated)
            )
            self.heard[tier] += 1

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


def cost_to_completion(start: int, prompt_tokens: float, rates: dict, learned: Learned) -> float:
    """Return the expected joules from handing the query to tier `start` until it is answered.

    C(j) = E_j + p_j * C(j+1): tier j always runs; with probability p_j the query goes one
    tier further. The policy decides where the OLT's rate in `rates` comes from.

    Args:
        start: Index of the tier that gets the query.
        prompt_tokens: The query's prompt length.
        rates: The J per token each tier is believed to cost now.
        learned: The household's knowledge (answer lengths, escalation rates).

    Returns:
        The expected energy, in joules.
    """
    total, chance_of_reaching = 0.0, 1.0
    for index in range(start, TOP + 1):
        tier = TIERS[index]
        total += chance_of_reaching * answer_energy(tier, prompt_tokens, rates, learned)
        if index < TOP:
            chance_of_reaching *= learned.p_further[tier]
    return total


def on_arrival(here: int, prompt_tokens: float, rates: dict, learned: Learned, delta: float) -> int:
    """Decide whether to run the query here or forward it straight to a tier above.

    Running here costs this tier's energy plus, if it escalates, the cheapest way on. The
    query is forwarded only if a tier above is cheaper by more than the margin delta.

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
    above = {
        k: cost_to_completion(k, prompt_tokens, rates, learned) for k in range(here + 1, TOP + 1)
    }
    tier = TIERS[here]
    run_here = answer_energy(tier, prompt_tokens, rates, learned) + learned.p_further[tier] * min(
        above.values()
    )
    cheapest = min(above, key=above.get)
    return cheapest if run_here - above[cheapest] > delta * run_here else here


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
        The index of the next tier to try.
    """
    above = {
        k: cost_to_completion(k, prompt_tokens, rates, learned) for k in range(here + 1, TOP + 1)
    }
    cheapest = min(above, key=above.get)
    return cheapest if above[here + 1] - above[cheapest] > delta * above[here + 1] else here + 1
