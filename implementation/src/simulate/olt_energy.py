"""What an OLT query really costs, and what the OLT can tell the tiers below.

Part of the simulator (simulate.py).

    Energy       the true energy rates of every tier; the OLT's depend on the batch
                 a query meets, under average or marginal accounting.
    OltReporter  the OLT's own mean rate over its last few minutes of traffic: the
                 report that rides on answers (piggyback) or goes out on the PON's
                 downstream broadcast (broadcast policy).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # implementation/src
from energy.three_tier import OltCurve  # noqa: E402


class Energy:
    """True energy rates per tier; the OLT's depend on its batch.

    accounting 'average' charges a query its share of the batch's energy, the
    measured J per token at that batch. 'marginal' charges the energy it adds to
    the network: the OLT is on, and serving other PONs, whether or not this
    query goes there (energy_tests.md §8.3). The user and ONU figures are the
    same under both.
    """

    def __init__(self, curve: OltCurve, pub: dict, accounting: str = "average", speeds: dict | None = None):
        """curve: the OLT's measured curve (at the chosen boundary); pub: phone and ONU rates
        (energy.three_tier.published_rates); accounting: 'average' or 'marginal';
        speeds: phone and ONU tokens per second (published_speeds), for seconds() only.
        """
        self.curve, self.pub, self.speeds = curve, pub, speeds
        self.olt = curve.marginal_rates if accounting == "marginal" else curve.rates
        self._cache: dict = {}

    def rates(self, tier: str, batch: int) -> tuple[float, float]:
        """True (J per prompt token, J per generated token) of a tier; the OLT's at this batch size."""
        if tier == "olt":
            return self.olt(batch)
        return self.pub[tier]["pf"], self.pub[tier]["dec"]

    def seconds(self, tier: str, batch: int, prompt_tokens: float, gen_tokens: float) -> float:
        """How long this tier takes to answer: its compute time only (network time is milliseconds).

        Phone: prompt at its prefill speed plus answer at its decode speed. ONU: the
        answer at its all-in speed (prompt included). OLT: the answer at the
        per-sequence speed of the batch it joined (OltCurve.service_s), prompt included.
        """
        if tier == "olt":
            return self.curve.service_s(batch, gen_tokens)
        v = self.speeds[tier]
        return (prompt_tokens / v["pf"] if v["pf"] else 0.0) + gen_tokens / v["dec"]

    def expected_olt(self, load: float) -> tuple[float, float]:
        """E[rates] over the batch an arrival meets, 1 + Poisson(load)."""
        key = round(load, 2)
        if key not in self._cache:
            L = max(key, 0.0)
            kmax = int(L + 8 * math.sqrt(L) + 10)
            pf = dec = 0.0
            for k in range(kmax + 1):
                # Poisson(L) probability of k others in service; an empty OLT (L = 0) means batch 1 surely
                w = float(k == 0) if L == 0 else math.exp(k * math.log(L) - L - math.lgamma(k + 1))
                a, b = self.olt(1 + k)
                pf += w * a
                dec += w * b
            self._cache[key] = (pf, dec)
        return self._cache[key]


class OltReporter:
    """The OLT's own mean energy rate over the last `minutes` of its traffic, at any moment t.

    The OLT serves every PON, so over that window it sees n ~ Poisson(arrivals in
    it) queries (Little's law), each meeting its own batch, 1 + Poisson(load at its
    time). The report is the mean of their rates: cheap for the OLT to keep, far
    less noisy than one query's rate (which under marginal accounting is ~100x
    higher when the query found the OLT idle), and at most `minutes` old. It rides
    on an answer (report: window, piggyback) or goes out in the OLT's broadcast.
    """

    def __init__(self, load_at, E: "Energy", curve: OltCurve, olt_tokens: float, minutes: float,
                 max_load: float, rng):
        """load_at: the OLT's true load at any times; olt_tokens: mean tokens of an OLT answer
        (for service time, Little's law); minutes: the averaging window; max_load: sizes the
        batch table; rng: this report stream's own random generator.
        """
        self.load_at, self.curve, self.tok, self.minutes, self.rng = load_at, curve, olt_tokens, minutes, rng
        self.kmax = max(int(max_load + 8 * math.sqrt(max_load) + 10), 80)   # rates hold at batch 64 beyond
        self.table = np.array([E.olt(1 + k) for k in range(self.kmax + 1)])
        self.log_b, self.log_tps = np.log(curve.b), np.log(curve.tps)

    def _arrivals_per_s(self, L: np.ndarray) -> np.ndarray:
        """Little's law, vectorized OltCurve.service_s at batch 1 + L."""
        b = np.clip(1 + L, self.curve.b[0], self.curve.b[-1])
        tps = np.exp(np.interp(np.log(b), self.log_b, self.log_tps))
        return L / (self.tok / (tps / b))

    def means_at(self, ts, chunk: int = 2000) -> list:
        """The report at each of the times ts."""
        ts, w, out = np.asarray(ts, dtype=float), self.minutes / 60, []
        for c in range(0, len(ts), chunk):
            t = ts[c:c + chunk]
            n = np.maximum(1, self.rng.poisson(self._arrivals_per_s(self.load_at(t - w / 2)) * self.minutes * 60))
            when = np.repeat(t, n) - w * self.rng.random(int(n.sum()))     # each arrival in the window
            k = np.minimum(self.rng.poisson(self.load_at(when)), self.kmax)  # the batch it met
            sums = np.add.reduceat(self.table[k], np.concatenate([[0], np.cumsum(n)[:-1]]), axis=0)
            out += [(float(pf), float(dec)) for pf, dec in sums / n[:, None]]
        return out

    def broadcasts(self, times, interval_s: float) -> list:
        """What a household knows at each arrival: the OLT's last broadcast before it."""
        dt = interval_s / 3600
        slots, which = np.unique(np.floor(np.asarray(times) / dt), return_inverse=True)
        sent = self.means_at(slots * dt)
        return [sent[j] for j in which]
