"""What an OLT query really costs, and what the OLT can tell the tiers below."""

# pylint: disable=wrong-import-position

from __future__ import annotations

import math
import sys
from pathlib import Path

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
