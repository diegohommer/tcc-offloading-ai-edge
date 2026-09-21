"""The OLT's load over time: how many queries it has in service at any moment.

Part of the simulator (simulate.py). The OLT serves many PONs, so its load is
exogenous: set from outside, not by the cascade's own queries. Two sources:

    synthetic   BURSTGPT's average day (load_shape), optionally multiplied by a
                slow random drift (load_noise). Used by config/simulation.yaml.
    trace       BurstGPT's hourly request counts replayed day by day (TraceLoad),
                optionally with unforeseen surges and dips on top (Surges). Used
                by the case study, config/study.yaml.

calibrate() builds, from the trace's training days only, the OLT cost tables of
the policies that are configured in advance: static_day (one rate for the whole
day) and static_hour (one rate per hour of day, weekday or weekend).
"""
from __future__ import annotations

import collections
import csv
import math
from pathlib import Path

import numpy as np

TRACES = Path(__file__).resolve().parents[2] / "data" / "load_traces"

# BurstGPT arrivals per hour of day (61 days of regional Azure OpenAI traffic):
# only the daily SHAPE of the OLT's load is taken from it.
BURSTGPT = [768, 727, 469, 443, 318, 331, 180, 142, 165, 347, 710, 953,
            1318, 1283, 1551, 1109, 1283, 1170, 998, 1377, 1127, 940, 912, 877]


def load_shape(t_h: float) -> float:
    """OLT offered load at time t (hours), relative to the busiest hour; each
    hour's value sits at its midpoint, linear in between, wrapping at midnight."""
    x = (t_h - 0.5) % 24
    h0 = int(x)
    f = x - h0
    peak = max(BURSTGPT)
    return ((1 - f) * BURSTGPT[h0] + f * BURSTGPT[(h0 + 1) % 24]) / peak


def load_shape_vec(ts) -> np.ndarray:
    """load_shape for an array of times."""
    x = (np.asarray(ts, dtype=float) - 0.5) % 24
    h0 = x.astype(int)
    f = x - h0
    b = np.array(BURSTGPT, dtype=float)
    return ((1 - f) * b[h0] + f * b[(h0 + 1) % 24]) / b.max()


def load_noise(stream, sigma: float, tau_h: float, seed: int) -> list[float]:
    """Multiplier on the OLT's load at each arrival, mean 1.

    exp of an Ornstein-Uhlenbeck process with stationary sd sigma and
    correlation time tau_h hours: the real load drifts off BurstGPT's average
    day (busier and quieter days, surges) in a way no configuration set in
    advance can follow. sigma = 0 is the average day exactly.
    """
    if sigma == 0:
        return [1.0] * len(stream)
    rng = np.random.default_rng([seed, 7919])      # one path, shared by every load and policy
    x, t0, out = rng.normal(0, sigma), stream[0][1], []
    for _, t, _ in stream:
        a = math.exp(-(t - t0) / tau_h)
        x = a * x + sigma * math.sqrt(1 - a * a) * rng.normal()
        t0 = t
        out.append(math.exp(x - sigma ** 2 / 2))
    return out


class TraceLoad:
    """The OLT's load replayed from a trace's hourly request counts (prepare_load_traces.py).

    rel(t) is the load at t hours from the trace's start, relative to the busiest
    hour of the training days' average day, linear between hour midpoints like
    load_shape. The first train_days are what a schedule may be calibrated on;
    the simulation runs on the days after.
    """

    def __init__(self, column: str, train_days: int, path: Path = TRACES / "burstgpt_hourly.csv"):
        """column: 'conversation', 'api' or 'all' (data/load_traces/burstgpt_hourly.csv);
        train_days: days a schedule may learn from, the rest are simulated.
        """
        with open(path) as f:
            self.c = np.array([float(r[column]) for r in csv.DictReader(f)]).reshape(-1, 24)
        self.column, self.train_days, self.days = column, train_days, len(self.c)
        if not 0 < train_days < self.days:
            raise ValueError(f"--train-days must be within 1..{self.days - 1}")
        train = self.c[:train_days]
        by_dow = [train[d::7].mean() for d in range(7)]
        self.weekend = set(int(d) for d in np.argsort(by_dow)[:2])   # the two quietest days of the week
        self.shape = train.mean(axis=0)                               # training days' average day
        self.flat = self.c.ravel() / self.shape.max()

    def rel(self, t_h: float) -> float:
        """Load at t hours from the trace's start, relative to the training days' busiest hour."""
        x = t_h - 0.5
        i0 = min(max(int(math.floor(x)), 0), len(self.flat) - 1)
        i1 = min(i0 + 1, len(self.flat) - 1)
        f = min(max(x - i0, 0.0), 1.0)
        return (1 - f) * self.flat[i0] + f * self.flat[i1]

    def rel_vec(self, ts) -> np.ndarray:
        """rel for an array of times."""
        return np.interp(np.asarray(ts, dtype=float), np.arange(len(self.flat)) + 0.5, self.flat)

    def daytype(self, t_h: float) -> int:
        """0 on a weekday, 1 on a weekend day (the two quietest days of the week)."""
        return int(int(t_h // 24) % 7 in self.weekend)


def calibrate(trace: TraceLoad, peak: float, E: "Energy", stale_factor: float, mult=None) -> dict:
    """Static configurations and the schedule, from the training days alone.

    Each is the OLT's expected rate averaged over its period, weighted by
    arrivals (which follow the trace), every 5 minutes: the whole training period
    for the static ones, each (weekday/weekend, hour) cell for the schedule.
    mult is the surges' multiplier on the load: events on training days are
    part of the average the schedule learns.
    """
    ts = np.arange(0, 24 * trace.train_days, 1 / 12) + 1 / 24
    w = np.array([trace.rel(t) for t in ts])
    load = peak * w * (mult(ts) if mult is not None else 1.0)
    cells = collections.defaultdict(list)
    for n, t in enumerate(ts):
        cells[(trace.daytype(t), int(t) % 24)].append(n)

    def avg(idx, scale=1.0):
        """Arrival-weighted mean OLT rate over these 5-minute steps, at `scale` times the load."""
        idx = np.asarray(idx)
        r = np.array([E.expected_olt(load[n] * scale) for n in idx])
        ww = w[idx] if w[idx].sum() > 0 else np.ones(len(idx))
        return float((r[:, 0] * ww).sum() / ww.sum()), float((r[:, 1] * ww).sum() / ww.sum())

    every = range(len(ts))
    return {"static_day": avg(every), "stale_low": avg(every, 1 / stale_factor),
            "stale_high": avg(every, stale_factor),
            "static_hour": {k: avg(v) for k, v in cells.items()}}


class Surges:
    """Unforeseen events on the OLT's load, on top of the replayed trace.

    What no timetable or calendar knows about: a news surge, a big game running
    long, a neighbouring OLT's traffic moved here, an outage elsewhere. Events
    arrive as a Poisson process, per_day a day on average, each lasting `hours`;
    each multiplies the load by `factor` (a surge) or divides it by `factor` (a
    dip), with equal probability, and overlapping events compound. factor 1 =
    none. The events depend on the seed alone, so every factor, load and policy
    meets the same ones. They fall on training days too, so the schedule learns
    an average that includes them. The cascade's own arrivals are unchanged: the
    events are other traffic at the OLT.
    """

    def __init__(self, factor: float, per_day: float, hours: float, span_h: float, seed: int):
        """factor: load multiplier of an event (1 = none); per_day: expected events a day;
        hours: each event's length; span_h: hours covered; seed: which events.
        """
        rng = np.random.default_rng([seed, 4242])
        n = int(rng.poisson(per_day * span_h / 24))
        self.start = rng.uniform(0, span_h, n)
        self.sign = rng.choice([-1, 1], n)
        self.end = self.start + hours
        self.factor = factor
        # piecewise constant: the multiplier between consecutive event boundaries
        self.bounds = np.unique(np.concatenate([[-np.inf], self.start, self.end]))
        active = (self.start[None, :] <= self.bounds[:, None]) & (self.bounds[:, None] < self.end[None, :])
        self.level = float(factor) ** (active * self.sign[None, :]).sum(axis=1)

    def mult(self, ts) -> np.ndarray:
        """The load multiplier at each of these times (1 outside events)."""
        ts = np.asarray(ts, dtype=float)
        if self.factor == 1:
            return np.ones_like(ts)
        return self.level[np.searchsorted(self.bounds, ts, side="right") - 1]

    def events(self) -> list:
        """The events drawn, as [start h, end h, +1 surge / -1 dip], for the run's JSON."""
        return [[round(float(s), 3), round(float(e), 3), int(g)] for s, e, g in zip(self.start, self.end, self.sign)]
