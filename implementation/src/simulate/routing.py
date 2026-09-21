"""The decision rules: RecServe's escalation test and the energy-aware routing rule.

Part of the simulator (simulate.py).

    Window              RecServe's rule: a tier escalates a query when its confidence
                        falls below the beta-quantile of the tier's recent confidences.
    Learned             what a household knows: answer lengths, how often queries go
                        further up, and the OLT's rate as last heard.
    cost_to_completion  expected joules from handing a query to tier j until answered:
                        C(j) = E_j + p_j * C(j+1).
    on_arrival          run the query here, or forward it straight to a tier above?
    on_escalation       send it to the next tier up, or skip to a later one?
"""
from __future__ import annotations

import bisect
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # implementation/src
from energy.three_tier import TIERS  # noqa: E402

TOP = len(TIERS) - 1                               # index of the highest tier (the OLT)


class Window:
    """RecServe's confidence history for one tier: the last n confidences, kept sorted.

    RecServe's escalation test: a tier escalates a query when the confidence of
    its answer is below quantile(beta) of this window. The threshold therefore
    adapts to the tier's own recent answers; beta sets roughly what share of
    queries it passes on.
    """

    def __init__(self, n: int):
        """n: how many recent confidences to keep (RecServe recommends 300-1000)."""
        self.n, self.q, self.s = n, collections.deque(), []

    def __len__(self):
        """How many confidences the window holds now."""
        return len(self.q)

    def add(self, x: float) -> None:
        """Record one more confidence, dropping the oldest once the window is full."""
        self.q.append(x)
        bisect.insort(self.s, x)
        if len(self.q) > self.n:
            del self.s[bisect.bisect_left(self.s, self.q.popleft())]

    def quantile(self, p: float) -> float:
        """Linear interpolation between order statistics, as numpy.percentile."""
        pos = p * (len(self.s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(self.s) - 1)
        return self.s[lo] + (self.s[hi] - self.s[lo]) * (pos - lo)


class Learned:
    """What one household's tier knows about itself and the tiers above it.

    Everything is learned from the packets that ride back down on answers, as
    exponentially weighted moving averages (EWMA):
        tokens[t]   how many tokens tier t generates per answer
        p_on[t]     how often a query that reached tier t went further up
        rates[t]    tier t's reported energy rates (J per prompt / generated token)
    tokens and p_on describe the questions, not the household, so with
    --shared-stats they are one pool shared by every household (pool=...).
    rates are always the household's own: what it heard from the OLT.
    """

    def __init__(self, alpha: float, warmup: int, rate_alpha: float | None = None, pool: "Learned | None" = None):
        """alpha: EWMA weight for tokens and p_on; rate_alpha: for rates (default alpha);
        warmup: reports per tier needed before deciding; pool: shared question statistics."""
        self.alpha, self.warmup = alpha, warmup
        self.rate_alpha = alpha if rate_alpha is None else rate_alpha   # weight on reported energy rates
        self.rates: dict[str, tuple[float, float]] = {}   # EWMA of reported (pf, dec): always this household's
        if pool is None:
            self.tokens: dict[str, float] = {}             # EWMA tokens generated there
            self.p_on: dict[str, float] = {}               # EWMA P(query went beyond it)
            self.n: collections.Counter = collections.Counter()
        else:                                              # question statistics shared by every household
            self.tokens, self.p_on, self.n = pool.tokens, pool.p_on, pool.n

    def absorb(self, packet: dict, final: str) -> None:
        """Learn from one answer's packet: {tier: (J/prompt token, J/generated token, tokens)}.

        final is the tier whose answer was kept: every tier below it passed the
        query on, which updates p_on.
        """
        for tier, (jpf, jdec, gen) in packet.items():
            went_on = float(TIERS.index(final) > TIERS.index(tier))
            self._ewma(self.tokens, tier, gen)
            self._ewma(self.p_on, tier, went_on)
            old = self.rates.get(tier)
            a = self.rate_alpha
            self.rates[tier] = (jpf, jdec) if old is None else (
                (1 - a) * old[0] + a * jpf, (1 - a) * old[1] + a * jdec)
            self.n[tier] += 1

    def _ewma(self, d: dict, k: str, x: float) -> None:
        """d[k] <- EWMA of d[k] and x (x itself the first time)."""
        d[k] = x if k not in d else (1 - self.alpha) * d[k] + self.alpha * x

    def ready(self, tiers) -> bool:
        """True once every one of these tiers has reported at least `warmup` times."""
        return all(self.n[t] >= self.warmup for t in tiers)


def cost_to_completion(j: int, prompt: float, rates: dict, L: Learned) -> float:
    """Expected joules from handing the query to tier j until it is answered.

    C(j) = E_j + p_j * C(j+1), with E_j = J_pf_j * prompt + J_dec_j * tokens_j:
    tier j always runs; with probability p_j the query goes one tier further.
    rates: the J per token each tier is believed to cost now (the policy decides
    where the OLT's comes from).
    """
    total, reach = 0.0, 1.0
    for k in range(j, TOP + 1):
        t = TIERS[k]
        jpf, jdec = rates[t]
        total += reach * (jpf * prompt + jdec * L.tokens[t])
        if k < TOP:
            reach *= L.p_on[t]
    return total


def on_arrival(i: int, prompt: float, rates: dict, L: Learned, delta: float) -> int:
    """Run the query at tier i (returns i), or forward it straight to a tier above.

    Running here costs this tier's energy plus, if it escalates, the cheapest way
    on. Forward only if a tier above is cheaper by more than the margin delta.
    """
    if i == TOP:
        return i
    costs = {k: cost_to_completion(k, prompt, rates, L) for k in range(i + 1, TOP + 1)}
    jpf, jdec = rates[TIERS[i]]
    run_here = jpf * prompt + jdec * L.tokens[TIERS[i]] + L.p_on[TIERS[i]] * min(costs.values())
    best = min(costs, key=costs.get)
    return best if run_here - costs[best] > delta * run_here else i


def on_escalation(i: int, prompt: float, rates: dict, L: Learned, delta: float) -> int:
    """After tier i escalated: the next tier up (i + 1), or skip to a later, cheaper one."""
    costs = {k: cost_to_completion(k, prompt, rates, L) for k in range(i + 1, TOP + 1)}
    best = min(costs, key=costs.get)
    return best if costs[i + 1] - costs[best] > delta * costs[i + 1] else i + 1
