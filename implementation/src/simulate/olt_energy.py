"""What an OLT query really costs, and what the OLT can tell the tiers below."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from energy.three_tier import OltCurve


class Energy:
    """True energy rates per tier; the OLT's depend on its batch.

    Average accounting charges a query its share of the batch's energy; marginal charges
    only what it adds, since the OLT is on and serving other PONs anyway. The phone's and
    the ONU's figures are the same under both.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(
        self, curve: OltCurve, pub: dict, accounting: str = "average", speeds: dict | None = None
    ):
        """Set up the rates.

        Args:
            curve: The OLT's measured curve, at the chosen boundary.
            pub: The phone's and ONU's rates (energy.three_tier.published_rates).
            accounting: "average" or "marginal".
            speeds: The phone's and ONU's tokens per second (published_speeds), for seconds().
        """
        self.curve, self.pub, self.speeds = curve, pub, speeds
        self.olt = curve.marginal_rates if accounting == "marginal" else curve.rates
        self._cache: dict = {}

    # ==========================================
    # Rates and times
    # ==========================================
    def rates(self, tier: str, batch: int) -> tuple[float, float]:
        """Return a tier's true (J per prompt token, J per generated token).

        Args:
            tier: The tier's name.
            batch: The OLT batch the query met (ignored below the OLT).
        """
        if tier == "olt":
            return self.olt(batch)
        return self.pub[tier]["pf"], self.pub[tier]["dec"]

    def seconds(self, tier: str, batch: int, prompt_tokens: float, gen_tokens: float) -> float:
        """Return how long a tier takes to answer: compute time only (network time is milliseconds).

        Args:
            tier: The tier's name.
            batch: The OLT batch the query met.
            prompt_tokens: The prompt's length.
            gen_tokens: The answer's length.

        Returns:
            Seconds: the phone's prefill plus decode, the ONU's all-in speed, or the OLT's
            per-sequence speed in the batch it joined.
        """
        if tier == "olt":
            return self.curve.service_s(batch, gen_tokens)
        v = self.speeds[tier]
        return (prompt_tokens / v["pf"] if v["pf"] else 0.0) + gen_tokens / v["dec"]

    def expected_olt(self, load: float) -> tuple[float, float]:
        """Return the OLT's expected rates at a load, over the batch an arrival meets (1 + Poisson(load)).

        Args:
            load: The OLT's offered load, in queries in service.
        """
        key = round(load, 2)
        if key not in self._cache:
            mean_load = max(key, 0.0)
            kmax = int(mean_load + 8 * math.sqrt(mean_load) + 10)
            pf = dec = 0.0
            for k in range(kmax + 1):
                # Poisson(load) probability of k others in service; an empty OLT means batch 1
                w = (
                    float(k == 0)
                    if mean_load == 0
                    else math.exp(k * math.log(mean_load) - mean_load - math.lgamma(k + 1))
                )
                a, b = self.olt(1 + k)
                pf += w * a
                dec += w * b
            self._cache[key] = (pf, dec)
        return self._cache[key]


class OltReporter:
    """The OLT's own mean energy rate over the last few minutes of its traffic.

    Much steadier than one query's rate (which under marginal accounting is ~100x higher
    when the query found the OLT idle), and at most `minutes` old. It rides on an answer
    (piggyback) or goes out in the OLT's broadcast.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(
        self,
        load_at,
        prices: Energy,
        curve: OltCurve,
        olt_tokens: float,
        minutes: float,
        max_load: float,
        rng,
    ):
        """Set up the reporter.

        Args:
            load_at: The OLT's true load at any times.
            prices: True energy rates.
            curve: The OLT's measured curve.
            olt_tokens: Mean tokens of an OLT answer (for service time, Little's law).
            minutes: The averaging window.
            max_load: The run's highest load, to size the batch table.
            rng: This report stream's own random generator.
        """
        self.load_at, self.curve, self.tok, self.minutes, self.rng = (
            load_at,
            curve,
            olt_tokens,
            minutes,
            rng,
        )
        self.kmax = max(
            int(max_load + 8 * math.sqrt(max_load) + 10), 80
        )  # rates hold at batch 64 beyond
        self.table = np.array([prices.olt(1 + k) for k in range(self.kmax + 1)])
        self.log_b, self.log_tps = np.log(curve.b), np.log(curve.tps)

    # ==========================================
    # Reports
    # ==========================================
    def _arrivals_per_s(self, loads: np.ndarray) -> np.ndarray:
        """Return arrivals per second at these loads (Little's law, service at batch 1 + load)."""
        b = np.clip(1 + loads, self.curve.b[0], self.curve.b[-1])
        tps = np.exp(np.interp(np.log(b), self.log_b, self.log_tps))
        return loads / (self.tok / (tps / b))

    def means_at(self, ts, chunk: int = 2000) -> list:
        """Return the OLT's report at each of these times.

        Over the window the OLT sees n ~ Poisson(arrivals) queries, each meeting its own
        batch, 1 + Poisson(load at its time); the report is the mean of their rates.

        Args:
            ts: Times, in hours.
            chunk: How many times to compute at once.

        Returns:
            [(J/prompt token, J/generated token)] per time.
        """
        ts, w, out = np.asarray(ts, dtype=float), self.minutes / 60, []
        for c in range(0, len(ts), chunk):
            t = ts[c : c + chunk]
            n = np.maximum(
                1,
                self.rng.poisson(self._arrivals_per_s(self.load_at(t - w / 2)) * self.minutes * 60),
            )
            when = np.repeat(t, n) - w * self.rng.random(int(n.sum()))  # each arrival in the window
            k = np.minimum(self.rng.poisson(self.load_at(when)), self.kmax)  # the batch it met
            sums = np.add.reduceat(self.table[k], np.concatenate([[0], np.cumsum(n)[:-1]]), axis=0)
            out += [(float(pf), float(dec)) for pf, dec in sums / n[:, None]]
        return out

    def broadcasts(self, times, interval_s: float) -> list:
        """Return what a household knows at each arrival: the OLT's last broadcast before it.

        Args:
            times: Arrival times, in hours.
            interval_s: Seconds between broadcasts.
        """
        dt = interval_s / 3600
        slots, which = np.unique(np.floor(np.asarray(times) / dt), return_inverse=True)
        sent = self.means_at(slots * dt)
        return [sent[j] for j in which]
