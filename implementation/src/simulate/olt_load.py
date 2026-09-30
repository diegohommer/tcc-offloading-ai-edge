"""The OLT's load over time: how many queries it has in service at any moment.

The OLT serves many PONs, so its load is set from outside, not by the cascade's own
queries: either BurstGPT's average day with a drift, or its hourly requests replayed day
by day with unforeseen surges and dips.
"""

from __future__ import annotations

import collections
import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # for the type hint only; never imported at run time
    from olt_energy import Energy

TRACES = Path(__file__).resolve().parents[2] / "data" / "load_traces"
"""Where the prepared load traces live (prepare_load_traces.py)."""

BURSTGPT = [
    768, 727, 469, 443, 318, 331, 180, 142, 165, 347, 710, 953,
    1318, 1283, 1551, 1109, 1283, 1170, 998, 1377, 1127, 940, 912, 877,
]  # fmt: skip
"""BurstGPT arrivals per hour of day (61 days of Azure OpenAI traffic): only its shape is used.

Typed in before data/load_traces/ was prepared, so it is not derived from it: it follows the
'all' column's average day (correlation 0.99, max shape difference 0.12). Only the synthetic
average day uses it; the case study replays data/load_traces/. Kept as is because every
run's JSON records it.
"""


# ==========================================
# Synthetic average day
# ==========================================
def load_shape(t_h: float) -> float:
    """Return the OLT's load at a time, relative to the busiest hour, on BurstGPT's average day.

    Each hour's value sits at its midpoint, linear in between, wrapping at midnight.

    Args:
        t_h: Time, in hours.
    """
    x = (t_h - 0.5) % 24
    h0 = int(x)
    f = x - h0
    peak = max(BURSTGPT)
    return ((1 - f) * BURSTGPT[h0] + f * BURSTGPT[(h0 + 1) % 24]) / peak


def load_shape_vec(ts) -> np.ndarray:
    """Return load_shape for an array of times.

    Args:
        ts: Times, in hours.
    """
    x = (np.asarray(ts, dtype=float) - 0.5) % 24
    h0 = x.astype(int)
    f = x - h0
    b = np.array(BURSTGPT, dtype=float)
    return ((1 - f) * b[h0] + f * b[(h0 + 1) % 24]) / b.max()


def load_noise(stream, sigma: float, tau_h: float, seed: int) -> list[float]:
    """Return a multiplier on the OLT's load at each arrival, mean 1: a slow random drift.

    The exponential of an Ornstein-Uhlenbeck process with stationary sd sigma and
    correlation time tau_h: busier and quieter days no configuration set in advance can
    follow.

    Args:
        stream: The query stream, [(question, time in hours, household)].
        sigma: The drift's log standard deviation (0 = the average day exactly).
        tau_h: The drift's correlation time, in hours.
        seed: Random seed (one path, shared by every load and policy).
    """
    if sigma == 0:
        return [1.0] * len(stream)
    rng = np.random.default_rng([seed, 7919])
    x, t0, out = rng.normal(0, sigma), stream[0][1], []
    for _, t, _ in stream:
        a = math.exp(-(t - t0) / tau_h)
        x = a * x + sigma * math.sqrt(1 - a * a) * rng.normal()
        t0 = t
        out.append(math.exp(x - sigma**2 / 2))
    return out


# ==========================================
# Replayed trace
# ==========================================
class TraceLoad:
    """The OLT's load replayed from a trace's hourly request counts (prepare_load_traces.py).

    The first train_days are what a schedule may be calibrated on; the simulation runs on
    the days after.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(self, column: str, train_days: int, path: Path = TRACES / "burstgpt_hourly.csv"):
        """Load the trace and summarize its training days.

        Args:
            column: "conversation", "api" or "all".
            train_days: Days a schedule may learn from; the rest are simulated.
            path: The prepared hourly counts.

        Raises:
            ValueError: When train_days leaves no day to simulate.
        """
        with open(path, encoding="utf-8") as f:
            self.c = np.array([float(r[column]) for r in csv.DictReader(f)]).reshape(-1, 24)
        self.column, self.train_days, self.days = column, train_days, len(self.c)
        if not 0 < train_days < self.days:
            raise ValueError(f"--train-days must be within 1..{self.days - 1}")
        train = self.c[:train_days]
        by_dow = [train[d::7].mean() for d in range(7)]

        self.weekend = set(int(d) for d in np.argsort(by_dow)[:2])
        """The two quietest days of the week (day index mod 7)."""

        self.shape = train.mean(axis=0)
        """The training days' average day, per hour."""

        self.flat = self.c.ravel() / self.shape.max()
        """Every hour of the trace, relative to the training days' busiest hour."""

    # ==========================================
    # Load at a time
    # ==========================================
    def rel(self, t_h: float) -> float:
        """Return the load at a time, relative to the training days' busiest hour.

        Args:
            t_h: Hours from the trace's start (linear between hour midpoints).
        """
        x = t_h - 0.5
        i0 = min(max(int(math.floor(x)), 0), len(self.flat) - 1)
        i1 = min(i0 + 1, len(self.flat) - 1)
        f = min(max(x - i0, 0.0), 1.0)
        return (1 - f) * self.flat[i0] + f * self.flat[i1]

    def rel_vec(self, ts) -> np.ndarray:
        """Return rel for an array of times."""
        return np.interp(np.asarray(ts, dtype=float), np.arange(len(self.flat)) + 0.5, self.flat)

    def daytype(self, t_h: float) -> int:
        """Return 0 on a weekday, 1 on a weekend day (the two quietest days of the week)."""
        return int(int(t_h // 24) % 7 in self.weekend)


def calibrate(
    trace: TraceLoad, peak: float, prices: Energy, stale_factor: float, mult=None
) -> dict:
    """Return the OLT rates the static policies ship with, from the training days alone.

    Each is the OLT's expected rate averaged over its period, weighted by arrivals, every 5
    minutes: the whole training period for the static ones, each (weekday/weekend, hour)
    cell for static_hour.

    Args:
        trace: The replayed trace.
        peak: The OLT's load at its busiest hour.
        prices: True energy rates (olt_energy.Energy).
        stale_factor: How far off the stale configurations' assumed load is (x and 1/x).
        mult: The surges' load multiplier: events on training days are part of the average.

    Returns:
        {"static_day": rates, "stale_low": ..., "stale_high": ..., "static_hour": {cell: rates}}.
    """
    ts = np.arange(0, 24 * trace.train_days, 1 / 12) + 1 / 24
    w = np.array([trace.rel(t) for t in ts])
    load = peak * w * (mult(ts) if mult is not None else 1.0)
    cells = collections.defaultdict(list)
    for n, t in enumerate(ts):
        cells[(trace.daytype(t), int(t) % 24)].append(n)

    def avg(idx, scale=1.0):
        """Return the arrival-weighted mean OLT rate over these 5-minute steps, at `scale` x the load."""
        idx = np.asarray(idx)
        r = np.array([prices.expected_olt(load[n] * scale) for n in idx])
        ww = w[idx] if w[idx].sum() > 0 else np.ones(len(idx))
        return float((r[:, 0] * ww).sum() / ww.sum()), float((r[:, 1] * ww).sum() / ww.sum())

    every = range(len(ts))
    return {
        "static_day": avg(every),
        "stale_low": avg(every, 1 / stale_factor),
        "stale_high": avg(every, stale_factor),
        "static_hour": {k: avg(v) for k, v in cells.items()},
    }


# ==========================================
# Unforeseen events
# ==========================================
class Surges:
    """Unforeseen events on the OLT's load, on top of the replayed trace.

    What no timetable knows about: a news surge, traffic moved from a neighbouring OLT, an
    outage elsewhere. Events arrive as a Poisson process and multiply the load by `factor`
    (a surge) or divide it (a dip), with equal probability; overlapping events compound.
    They depend on the seed alone, so every factor, load and policy meets the same ones.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(self, factor: float, per_day: float, hours: float, span_h: float, seed: int):
        """Draw the events.

        Args:
            factor: Load multiplier of an event (1 = none).
            per_day: Expected events a day.
            hours: Each event's length.
            span_h: Hours covered.
            seed: Random seed (which events).
        """
        rng = np.random.default_rng([seed, 4242])
        n = int(rng.poisson(per_day * span_h / 24))
        self.start = rng.uniform(0, span_h, n)
        self.sign = rng.choice([-1, 1], n)
        self.end = self.start + hours
        self.factor = factor
        # piecewise constant: the multiplier between consecutive event boundaries
        self.bounds = np.unique(np.concatenate([[-np.inf], self.start, self.end]))
        active = (self.start[None, :] <= self.bounds[:, None]) & (
            self.bounds[:, None] < self.end[None, :]
        )
        self.level = float(factor) ** (active * self.sign[None, :]).sum(axis=1)

    # ==========================================
    # Queries
    # ==========================================
    def mult(self, ts) -> np.ndarray:
        """Return the load multiplier at each of these times (1 outside events).

        Args:
            ts: Times, in hours.
        """
        ts = np.asarray(ts, dtype=float)
        if self.factor == 1:
            return np.ones_like(ts)
        return self.level[np.searchsorted(self.bounds, ts, side="right") - 1]

    def events(self) -> list:
        """Return the events drawn, as [start h, end h, +1 surge / -1 dip], for the run's JSON."""
        return [
            [round(float(s), 3), round(float(e), 3), int(g)]
            for s, e, g in zip(self.start, self.end, self.sign)
        ]
