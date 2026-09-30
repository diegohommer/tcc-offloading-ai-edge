"""The two ways of setting up the OLT's load, behind one interface.

TraceMode is the case study (a replayed trace with unforeseen surges); SyntheticMode is
the earlier exploration (BurstGPT's average day with an optional drift). simulate.py asks
whichever one it has the same questions.
"""

from __future__ import annotations

import collections
import statistics as st

import numpy as np

from cascade import build_stream
from olt_load import calibrate, load_noise, load_shape, load_shape_vec, Surges, TraceLoad


class TraceMode:
    """The case study: BurstGPT's hourly requests replayed day by day, plus surges and dips.

    The static policies are calibrated on the trace's training days only; every policy
    runs on the days after.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(self, settings, questions):
        """Load the trace, draw the surges and build the query stream.

        Args:
            settings: simulate.py's parsed arguments.
            questions: The question indices that can be asked.
        """
        s = settings
        self.settings = s
        self.trace = TraceLoad(s.load_trace, s.train_days)
        self.surges = Surges(
            s.surge_factor, s.surge_per_day, s.surge_hours, 24 * self.trace.days, s.seed
        )

        self.stream = build_stream(
            questions, s.days, s.per_day, s.seed, s.households, self.trace, s.household_shape
        )
        """The time-ordered queries arriving at the households' phones."""

        self.times = np.array([t for _, t, _ in self.stream])
        """Each query's arrival time, in hours."""

        self.multiplier = list(self.surges.mult(self.times))
        """The surges' load multiplier at each arrival."""

        self.relative_load = self.trace.rel
        """The OLT's load at a time, relative to the busiest hour."""

        self.average_day = [float(x) for x in self.trace.shape / self.trace.shape.max()]
        """The 24 hourly loads of the training days' average day, relative to the peak."""

    # ==========================================
    # Interface
    # ==========================================
    def load_at(self, ts, peak):
        """Return the OLT's true load at any times, for its own reports and broadcasts.

        Args:
            ts: Times, in hours.
            peak: The OLT's load at its busiest hour.
        """
        return peak * self.trace.rel_vec(ts) * self.surges.mult(ts)

    def schedule_key(self, t):
        """Return static_hour's cell for a time: (0 weekday / 1 weekend, hour of day)."""
        return self.trace.daytype(t), int(t) % 24

    def static_rates(self, peak, prices, loads):
        """Return the OLT rates the static policies ship with, from the training days alone.

        Args:
            peak: The OLT's load at its busiest hour.
            prices: True energy rates (olt_energy.Energy).
            loads: Unused: the trace mode calibrates on the training days, not this run.
        """
        del loads
        return calibrate(self.trace, peak, prices, self.settings.stale_factor, self.surges.mult)

    def describe(self):
        """Return the load, in one line of the printout."""
        s, trace = self.settings, self.trace
        return (
            f"{trace.days - trace.train_days} days of BurstGPT '{s.load_trace}' after "
            f"{trace.train_days} calibration days, surges x{s.surge_factor:g} "
            f"({len(self.surges.start)} events)"
        )

    def json_fields(self):
        """Return the (load_trace, surges) entries of the run's JSON."""
        s, trace = self.settings, self.trace
        load_trace = {
            "column": trace.column,
            "train_days": trace.train_days,
            "days": trace.days,
            "weekend_days_mod7": sorted(trace.weekend),
            "average_day": self.average_day,
        }
        surges = {
            "factor": s.surge_factor,
            "per_day": s.surge_per_day,
            "hours": s.surge_hours,
            "events_start_end_sign": self.surges.events(),
        }
        return load_trace, surges

    def tag(self):
        """Return the load's part of the output file's name."""
        s = self.settings
        return f"_trace-{s.load_trace}" + (
            f"_surge{s.surge_factor:g}" if s.surge_factor != 1 else ""
        )


class SyntheticMode:
    """The earlier exploration: every day is BurstGPT's average day, times an optional drift.

    The static policies are calibrated on the very load the run meets.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(self, settings, questions):
        """Build the query stream and the drift.

        Args:
            settings: simulate.py's parsed arguments.
            questions: The question indices that can be asked.
        """
        s = settings
        self.settings = s

        self.stream = build_stream(
            questions, s.days, s.per_day, s.seed, s.households, None, s.household_shape
        )
        """The time-ordered queries arriving at the households' phones."""

        self.times = np.array([t for _, t, _ in self.stream])
        """Each query's arrival time, in hours."""

        self.multiplier = load_noise(self.stream, s.load_sigma, s.load_tau, s.seed)
        """The drift's load multiplier at each arrival."""

        self.relative_load = load_shape
        """The OLT's load at a time, relative to the busiest hour."""

        self.average_day = [load_shape(h + 0.5) for h in range(24)]
        """The 24 hourly loads of the average day, relative to the peak."""

    # ==========================================
    # Interface
    # ==========================================
    def load_at(self, ts, peak):
        """Return the OLT's true load at any times, for its own reports and broadcasts.

        Args:
            ts: Times, in hours.
            peak: The OLT's load at its busiest hour.
        """
        return peak * load_shape_vec(ts) * np.interp(ts, self.times, self.multiplier)

    def schedule_key(self, t):
        """Return static_hour's cell for a time: every day is the same, so (0, hour of day)."""
        return 0, int(t) % 24

    def static_rates(self, peak, prices, loads):
        """Return the OLT rates the static policies ship with, from the load the run meets.

        Args:
            peak: Unused: the loads already include it.
            prices: True energy rates (olt_energy.Energy).
            loads: The OLT's true load at each arrival.
        """
        del peak
        stale = self.settings.stale_factor

        def day_average(scale):
            """Return the OLT's mean expected rate over the run, at `scale` times its load."""
            rs = [prices.expected_olt(load * scale) for load in loads]
            return st.mean(r[0] for r in rs), st.mean(r[1] for r in rs)

        rates = {
            "static_day": day_average(1.0),
            "stale_low": day_average(1 / stale),
            "stale_high": day_average(stale),
        }
        by_hour = collections.defaultdict(list)
        for (_, t, _), load in zip(self.stream, loads):
            by_hour[int(t) % 24].append(prices.expected_olt(load))
        rates["static_hour"] = {
            (0, h): (st.mean(r[0] for r in by_hour[h]), st.mean(r[1] for r in by_hour[h]))
            for h in range(24)
        }
        return rates

    def describe(self):
        """Return the load, in one line of the printout."""
        s = self.settings
        return f"{s.days} days, load drift sigma {s.load_sigma:g} (tau {s.load_tau:g} h)"

    def json_fields(self):
        """Return the (load_trace, surges) entries of the run's JSON: neither applies."""
        return None, None

    def tag(self):
        """Return the load's part of the output file's name: nothing to add."""
        return ""
