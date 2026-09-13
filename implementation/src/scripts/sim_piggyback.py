#!/usr/bin/env python3
"""Three-tier cascade (user -> ONU -> OLT) with piggybacked energy reports.

RecServe still decides WHETHER a query escalates, unchanged: the beta-quantile
of each tier's own confidence history. This adds one thing. Every answer
travelling back down carries a small piggyback packet in which each tier that
worked on the query reports the energy rates it just ran at (J per prompt token,
J per generated token) and how many tokens it generated. No extra messages: the
packet rides on a response that was coming back anyway.

From those packets a tier learns what the tiers above it cost right now, and
uses it for the two decisions RecServe leaves implicit:

    on arrival     run the query here, or forward it straight to a tier above?
    on escalation  send it to the next tier up, or skip to a later one?

Both pick the lowest expected cost to completion,

    C(j) = E_j + p_j * C(j+1)        E_j = J_pf_j * prompt + J_dec_j * tokens_j

where p_j is how often a query that reached j went further. The skip margin
delta keeps the default (run here / next tier up) unless the alternative is at
least delta cheaper, as a fraction of the default's cost.

Nine policies. The last seven share that rule and differ ONLY in which OLT
rate they see, so the comparison isolates what the OLT's information is worth:

    stepwise    plain RecServe, the baseline
    skip_onu    plain RecServe on a two-tier chain, user -> OLT: the ONU is never
                used. The control for "is the gain just from dropping the ONU?"
    static      the OLT's day-average rate, fixed -- a config shipped once, calibrated
                on exactly the load it then meets (the best a single fixed rate can do)
    stale_low   the same, but calibrated when load was --stale-factor times lower
    stale_high  ... or --stale-factor times higher: a config that has gone stale
    schedule    a time-of-day table: the OLT's mean rate in each hour of the day,
                calibrated on the load it then meets. Knows the typical day, not today
    piggyback   learned from the packets on the household's own answers (EWMA of the
                rates reported)
    broadcast   the OLT's own mean rate over its last --report-window minutes, sent to
                every ONU on the PON every --broadcast-interval-s seconds on the
                downstream multicast channel: every household knows it, asked or not
    oracle      the true expected rate at this moment -- the upper bound on live information

--shared-stats pools the answer lengths and escalation rates every policy's rule
also needs across households (they describe the questions, not the household), so
a household decides from its first query; its own devices' rates it knows, and
only the OLT's has to reach it.

With --load-sigma 0 the load IS the typical day, so the schedule is the oracle
at hourly resolution; only a load that drifts from it (--load-sigma > 0) can
separate a live signal from a well-kept schedule.

REAL LOAD AND MANY HOUSEHOLDS (--load-trace, --households)
  --load-trace replays a public trace's hourly request counts as the OLT's load
  (data/load_traces/, written by prepare_load_traces.py): 'conversation' is
  BurstGPT's people-chatting traffic, the case study's load; 'all' adds its API
  traffic, a burstier stress case. The static configurations and the schedule
  are calibrated on the first --train-days only, and every policy runs on the
  days after them, which nobody knew in advance. The schedule then knows the
  hour of day and weekday/weekend. The cascade's own arrivals follow the trace too.
  --households splits those arrivals over that many households, each with its own
  user device and ONU, so each learns the OLT's cost from its own answers alone;
  --per-day is then per household. RecServe's confidence thresholds stay one per
  tier, shared by all households (a threshold calibrated on the tier's population).
  --surge-factor adds unforeseen events to the replayed load, surges and dips no
  timetable or calendar knows about (class Surges): the unpredictable traffic.

MARGINAL ACCOUNTING charges no tier for being on: the OLT's card figure net of
idle (and, at the whole-system boundary, without the idle-machines share: x 2.20),
the ONU's all-in figure minus its board's idle draw (0.61 instead of 1.11 J per
token), the user's SoC figure as is (energy_tests.md §8.3, §8.5).

WHAT THE NUMBERS REST ON (energy_tests.md §3-§8)
  - Answers: the 1,319-question zero-shot GSM8K collection, every tier.
  - Energy: user and ONU from published measurements, constant (batch-1
    devices); the OLT from the measured L4 batch curve, converted to the
    whole-system boundary (x2.47) unless --boundary says otherwise.
  - OLT load is exogenous: the OLT serves many PONs, so the cascade's own
    queries are a small part of its traffic. Offered load (mean queries in
    service) follows BurstGPT's hourly shape, scaled so the busiest hour has
    --peak-loads. An arriving query finds a Poisson number of others in service
    (M/G/inf, arrivals see time averages), so its batch is 1 + Poisson(load):
    the rate any single packet reports is noisy.
  - Results are compared at equal accuracy: skipping to a more accurate tier
    changes accuracy too, so J/query is read off the stepwise frontier at the
    same accuracy.

SETTINGS: every default is in config/sim_piggyback.yaml, which says what each
setting does and where its value comes from; a flag overrides one for one run.

Usage:
    python src/scripts/sim_piggyback.py                      # config/sim_piggyback.yaml as is
    python src/scripts/sim_piggyback.py --boundary gpu       # GPU-card OLT, for comparison
    python src/scripts/sim_piggyback.py --onu-scale 0.2      # a 5x more efficient ONU
    python src/scripts/sim_piggyback.py --load-sigma 0.5 --days 14    # load drifting off the average day
    python src/scripts/sim_piggyback.py --accounting marginal         # every tier pays what a query adds
    python src/scripts/sim_piggyback.py --load-trace conversation --households 40 --per-day 50
Writes results/energy_tests/sim_piggyback_<tag>_<UTC>.csv (one row per load, beta,
policy) and .json (the same plus per-hour detail and the inputs used).
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # implementation/src
from energy.three_tier import (RESULTS, TIERS, OltCurve, load_answers,  # noqa: E402
                               olt_factor, olt_runs, published_rates)

TOP = len(TIERS) - 1
TRACES = Path(__file__).resolve().parents[2] / "data" / "load_traces"
POLICIES = ("stepwise", "skip_onu", "static", "stale_low", "stale_high", "schedule", "piggyback", "broadcast",
            "oracle")
FIXED = ("stepwise", "skip_onu")                  # fixed chains: no energy information at all
CONFIG = Path(__file__).resolve().parents[2] / "config" / "sim_piggyback.yaml"
STATIC = ("static", "stale_low", "stale_high")   # fixed OLT rate, shipped as configuration

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
        x = t_h - 0.5
        i0 = min(max(int(math.floor(x)), 0), len(self.flat) - 1)
        i1 = min(i0 + 1, len(self.flat) - 1)
        f = min(max(x - i0, 0.0), 1.0)
        return (1 - f) * self.flat[i0] + f * self.flat[i1]

    def rel_vec(self, ts) -> np.ndarray:
        """rel for an array of times."""
        return np.interp(np.asarray(ts, dtype=float), np.arange(len(self.flat)) + 0.5, self.flat)

    def daytype(self, t_h: float) -> int:
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
        idx = np.asarray(idx)
        r = np.array([E.expected_olt(load[n] * scale) for n in idx])
        ww = w[idx] if w[idx].sum() > 0 else np.ones(len(idx))
        return float((r[:, 0] * ww).sum() / ww.sum()), float((r[:, 1] * ww).sum() / ww.sum())

    every = range(len(ts))
    return {"static": avg(every), "stale_low": avg(every, 1 / stale_factor),
            "stale_high": avg(every, stale_factor),
            "schedule": {k: avg(v) for k, v in cells.items()}}


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
        ts = np.asarray(ts, dtype=float)
        if self.factor == 1:
            return np.ones_like(ts)
        return self.level[np.searchsorted(self.bounds, ts, side="right") - 1]

    def events(self) -> list:
        return [[round(float(s), 3), round(float(e), 3), int(g)] for s, e, g in zip(self.start, self.end, self.sign)]


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


class Energy:
    """True energy rates per tier; the OLT's depend on its batch.

    accounting 'average' charges a query its share of the batch's energy, the
    measured J per token at that batch. 'marginal' charges the energy it adds to
    the network: the OLT is on, and serving other PONs, whether or not this
    query goes there (energy_tests.md §8.3). The user and ONU figures are the
    same under both.
    """

    def __init__(self, curve: OltCurve, pub: dict, accounting: str = "average"):
        self.curve, self.pub = curve, pub
        self.olt = curve.marginal_rates if accounting == "marginal" else curve.rates
        self._cache: dict = {}

    def rates(self, tier: str, batch: int) -> tuple[float, float]:
        if tier == "olt":
            return self.olt(batch)
        return self.pub[tier]["pf"], self.pub[tier]["dec"]

    def expected_olt(self, load: float) -> tuple[float, float]:
        """E[rates] over the batch an arrival meets, 1 + Poisson(load)."""
        key = round(load, 2)
        if key not in self._cache:
            L = max(key, 0.0)
            kmax = int(L + 8 * math.sqrt(L) + 10)
            pf = dec = 0.0
            for k in range(kmax + 1):
                w = math.exp(-L) if L == 0 else math.exp(k * math.log(L) - L - math.lgamma(k + 1))
                a, b = self.olt(1 + k)
                pf += w * a
                dec += w * b
            self._cache[key] = (pf, dec)
        return self._cache[key]


class Window:
    """RecServe's confidence history: the last n values, kept sorted for quantiles."""

    def __init__(self, n: int):
        self.n, self.q, self.s = n, collections.deque(), []

    def __len__(self):
        return len(self.q)

    def add(self, x: float) -> None:
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
    """What one tier knows about itself and the tiers above it -- from packets only."""

    def __init__(self, alpha: float, warmup: int, rate_alpha: float | None = None, pool: "Learned | None" = None):
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
        d[k] = x if k not in d else (1 - self.alpha) * d[k] + self.alpha * x

    def ready(self, tiers) -> bool:
        return all(self.n[t] >= self.warmup for t in tiers)


def cost_to_completion(j: int, prompt: float, rates: dict, L: Learned) -> float:
    """Expected joules from handing the query to tier j until it is answered."""
    total, reach = 0.0, 1.0
    for k in range(j, TOP + 1):
        t = TIERS[k]
        jpf, jdec = rates[t]
        total += reach * (jpf * prompt + jdec * L.tokens[t])
        if k < TOP:
            reach *= L.p_on[t]
    return total


def on_arrival(i: int, prompt: float, rates: dict, L: Learned, delta: float) -> int:
    """Run here (returns i), or forward straight to a tier above."""
    if i == TOP:
        return i
    costs = {k: cost_to_completion(k, prompt, rates, L) for k in range(i + 1, TOP + 1)}
    jpf, jdec = rates[TIERS[i]]
    run_here = jpf * prompt + jdec * L.tokens[TIERS[i]] + L.p_on[TIERS[i]] * min(costs.values())
    best = min(costs, key=costs.get)
    return best if run_here - costs[best] > delta * run_here else i


def on_escalation(i: int, prompt: float, rates: dict, L: Learned, delta: float) -> int:
    """Next tier up, or skip to a later one."""
    costs = {k: cost_to_completion(k, prompt, rates, L) for k in range(i + 1, TOP + 1)}
    best = min(costs, key=costs.get)
    return best if costs[i + 1] - costs[best] > delta * costs[i + 1] else i + 1


def run(rec, stream, batches, loads, beta, policy, args, E, static_rates, reports=None, sched_key=None, bcast=None):
    history = {t: Window(args.window) for t in TIERS}   # RecServe's thresholds: one per tier, shared by every household
    homes: dict[int, dict[str, Learned]] = {}           # per household, its tiers that can decide
    pool = ({t: Learned(args.alpha, args.warmup, args.rate_alpha) for t in TIERS[:TOP]}
            if args.shared_stats else None)             # question statistics pooled across households
    energy = correct = forwarded = skipped = 0.0
    rate_err = []
    final_at = collections.Counter()
    hourly = [[0.0, 0, 0, 0, 0] for _ in range(24)]      # joules, queries, final at user/onu/olt

    for n_q, ((qi, t_h, hh), b, load) in enumerate(zip(stream, batches, loads)):   # load: the OLT's true offered load now
        learners = homes.get(hh)
        if learners is None:
            learners = homes[hh] = {t: Learned(args.alpha, args.warmup, args.rate_alpha, pool[t] if pool else None)
                                    for t in TIERS[:TOP]}
        prompt = rec[qi]["user"]["tp"]
        packet: dict[str, tuple[float, float, int]] = {}
        spent = 0.0
        i = 0
        rates = None
        while True:
            t = TIERS[i]
            L = learners.get(t)
            decide = policy not in FIXED and L is not None and L.ready(TIERS[i:])
            if decide:
                if pool:    # a household knows its own devices' rates; only the OLT's has to reach it
                    rates = {u: E.rates(u, 1) for u in TIERS[i:TOP]}
                    rates["olt"] = L.rates.get("olt")
                else:
                    rates = {u: L.rates[u] for u in TIERS[i:]}
                if policy in STATIC:
                    rates["olt"] = static_rates[policy]
                elif policy == "schedule":
                    rates["olt"] = static_rates["schedule"][sched_key(t_h)]
                elif policy == "oracle":
                    rates["olt"] = E.expected_olt(load)
                elif policy == "broadcast":
                    rates["olt"] = bcast[n_q]
                if policy in ("piggyback", "broadcast") and rates["olt"] is not None:
                    true_dec = E.expected_olt(load)[1]
                    rate_err.append(abs(rates["olt"][1] - true_dec) / true_dec)
                decide = rates["olt"] is not None       # piggyback: this household has not heard the OLT yet
            if decide:
                j = on_arrival(i, prompt, rates, L, args.delta)
                if j != i:
                    forwarded += 1
                    i = j
                    continue

            d = rec[qi][t]
            jpf, jdec = E.rates(t, b)
            spent += jpf * d["tp"] + jdec * d["tg"]
            packet[t] = (*(reports[n_q] if reports and t == "olt" else (jpf, jdec)), d["tg"])

            hist = history[t]
            escalate = i < TOP and len(hist) > 1 and d["conf"] < hist.quantile(beta)
            hist.add(d["conf"])
            if not escalate:
                correct += d["correct"]
                final_at[t] += 1
                break
            nxt = on_escalation(i, prompt, rates, L, args.delta) if decide else i + 1
            if policy == "skip_onu":
                nxt = TOP
            skipped += nxt > i + 1
            i = nxt

        energy += spent
        hr = hourly[int(t_h) % 24]
        hr[0] += spent
        hr[1] += 1
        hr[2 + TIERS.index(t)] += 1
        # The answer travels back down. Every deciding tier at or below where it
        # was produced reads the packet for itself and the tiers above it. The ONU
        # relays everything above the user, so it hears reports even for queries
        # whose inference it skipped.
        final = t
        for tier, Lt in learners.items():
            if TIERS.index(tier) <= TIERS.index(final):
                Lt.absorb({u: v for u, v in packet.items() if TIERS.index(u) >= TIERS.index(tier)}, final)

    n = len(stream)
    return {
        "accuracy": correct / n,
        "J_per_query": energy / n,
        "forwarded_on_arrival": forwarded / n,
        "skipped_on_escalation": skipped / n,
        "olt_rate_error": st.mean(rate_err) if rate_err else float("nan"),
        **{f"final_{t}": final_at[t] / n for t in TIERS},
    }, [{"hour": h, "J_per_query": hr[0] / hr[1] if hr[1] else None, "queries": hr[1],
         **{f"final_{t}": (hr[2 + k] / hr[1] if hr[1] else None) for k, t in enumerate(TIERS)}}
        for h, hr in enumerate(hourly)]


def build_stream(indices, days, per_day, seed, households=1, trace=None):
    """Time-ordered arrivals at the user tier: (question, hour, household).

    Without a trace every day is BurstGPT's average day: hour h gets a share of
    per_day x households proportional to BURSTGPT. With one, arrivals follow the
    trace over its days after training: a Poisson number per hour, proportional
    to that hour's requests, per_day x households a day on average. Each arrival
    belongs to a household drawn at random.
    """
    rng = random.Random(seed)
    home = (lambda: rng.randrange(households)) if households > 1 else (lambda: 0)
    out = []
    if trace is None:
        total = sum(BURSTGPT)
        for day in range(days):
            times = []
            for h in range(24):
                times += [24 * day + h + rng.random()
                          for _ in range(round(per_day * households * BURSTGPT[h] / total))]
            out += [(rng.choice(indices), t, home()) for t in sorted(times)]
        return out
    nprng = np.random.default_rng([seed, 31])
    test = trace.c[trace.train_days:]
    per_count = per_day * households * len(test) / test.sum()
    for d, day in enumerate(test):
        for h, n in enumerate(day):
            t0 = 24 * (trace.train_days + d) + h
            for t in sorted(t0 + nprng.random(nprng.poisson(per_count * n))):
                out.append((rng.choice(indices), float(t), home()))
    return out


def iso_accuracy(rows: list[dict]) -> None:
    """Add, to each row, stepwise's J/query at the same accuracy and the saving against it."""
    step = sorted((r["accuracy"], r["J_per_query"]) for r in rows if r["policy"] == "stepwise")
    xs, ys = [a for a, _ in step], [j for _, j in step]
    for r in rows:
        a = r["accuracy"]
        if xs[0] <= a <= xs[-1]:
            ref = float(np.interp(a, xs, ys))
            r["J_stepwise_same_accuracy"] = ref
            r["saving_same_accuracy"] = 1 - r["J_per_query"] / ref
        else:
            r["J_stepwise_same_accuracy"] = float("nan")
            r["saving_same_accuracy"] = float("nan")


def frontier(rows: list[dict], targets: list[float]) -> dict:
    """One policy's J/query at each target accuracy, along its beta sweep.

    Only Pareto points are kept (no other beta is both more accurate and
    cheaper), then J is interpolated between them. It is the cost of reaching
    AT LEAST that accuracy: a target below the policy's range gets its least
    accurate point (it cannot be made less accurate, but does not need to be);
    a target above the range gets None.
    """
    pareto, best = [], float("inf")
    for a, j in sorted(((r["accuracy"], r["J_per_query"]) for r in rows), reverse=True):
        if j < best:
            pareto.append((a, j))
            best = j
    pareto.sort()
    xs, ys = [a for a, _ in pareto], [j for _, j in pareto]
    return {f"{t:.2f}": (None if t > xs[-1] else ys[0] if t < xs[0] else float(np.interp(t, xs, ys)))
            for t in targets}


def load_config(path: Path) -> dict:
    """The settings file's values, flattened into argparse defaults (lists become comma strings).

    A file may say `extends: <file>`: that file's values first, this one's on top.
    """
    with open(path) as f:
        groups = yaml.safe_load(f)
    base = groups.pop("extends", None)
    flat = load_config(path.parent / base) if base else {}
    for items in groups.values():
        for k, v in items.items():
            flat[k] = ",".join(str(x) for x in v) if isinstance(v, list) else v
    return flat


def main() -> int:
    # Every default comes from config/sim_piggyback.yaml (what each setting means and
    # where its value comes from is written there); a flag overrides it for one run.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=CONFIG, help="settings file (default config/sim_piggyback.yaml)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[pre])
    ap.add_argument("--betas", help="comma-separated RecServe beta values")
    ap.add_argument("--peak-loads", help="comma-separated OLT loads at the busiest hour, in queries in service")
    ap.add_argument("--boundary", choices=("system", "system-low", "gpu"), help="where the OLT's energy is counted (§7)")
    ap.add_argument("--confidence", choices=("mean", "min"), help="exp(mean logprob), RecServe's, or exp(min logprob)")
    ap.add_argument("--onu-scale", type=float, help="multiply the ONU's J per generated token")
    ap.add_argument("--accounting", choices=("average", "marginal"), help="share of the batch, or what a query adds")
    ap.add_argument("--load-sigma", type=float, help="log-sd of the synthetic load's drift (0 = none)")
    ap.add_argument("--load-tau", type=float, help="correlation time of that drift, hours")
    ap.add_argument("--report", choices=("query", "window"), help="the OLT packet's rate: this batch's, or its recent mean")
    ap.add_argument("--report-window", type=float, help="minutes the OLT averages over (--report window)")
    ap.add_argument("--stale-factor", type=float, help="how far off the stale configs' assumed load is (x and 1/x)")
    ap.add_argument("--delta", type=float, help="relative skip margin")
    ap.add_argument("--window", type=int, help="RecServe's confidence history per tier")
    ap.add_argument("--alpha", type=float, help="EWMA weight for learned answer lengths and escalation rates")
    ap.add_argument("--rate-alpha", type=float, help="EWMA weight for reported energy rates (unset: --alpha)")
    ap.add_argument("--warmup", type=int, help="reports per tier before a household decides")
    ap.add_argument("--days", type=int, help="days simulated on the synthetic average day")
    ap.add_argument("--per-day", type=int, help="queries per day at the user tier, per household")
    ap.add_argument("--households", type=int, help="households sharing the OLT, each learning on its own")
    ap.add_argument("--load-trace", choices=("conversation", "api", "all"),
                    help="replay BurstGPT's hourly requests as the OLT's load (data/load_traces/)")
    ap.add_argument("--train-days", type=int, help="trace days the configs and the schedule are calibrated on")
    ap.add_argument("--surge-factor", type=float, help="load multiplier of an unforeseen event (1 = none)")
    ap.add_argument("--surge-per-day", type=float, help="expected unforeseen events a day")
    ap.add_argument("--surge-hours", type=float, help="length of an unforeseen event, hours")
    ap.add_argument("--broadcast-interval-s", type=float, help="seconds between the OLT's broadcasts")
    ap.add_argument("--shared-stats", action=argparse.BooleanOptionalAction,
                    help="pool answer lengths and escalation rates across households")
    ap.add_argument("--policies", help="comma-separated subset; must include stepwise")
    ap.add_argument("--acc-targets", help="comma-separated accuracies the frontiers are read at")
    ap.add_argument("--no-hourly", action="store_true", help="leave the per-hour detail out of the JSON")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", type=Path, help="CSV path (default results/energy_tests/sim_piggyback_<tag>_<UTC>.csv)")
    config = pre.parse_known_args()[0].config
    defaults = load_config(config)
    dests = {a.dest for a in ap._actions} - {"help", "config", "out"}
    if set(defaults) != dests:
        print(f"{config}: missing {sorted(dests - set(defaults))}, unknown {sorted(set(defaults) - dests)}",
              file=sys.stderr)
        return 1
    ap.set_defaults(**defaults)
    args = ap.parse_args()
    for a in ap._actions:                       # argparse does not check defaults against choices
        v = getattr(args, a.dest, None)
        if a.choices and v is not None and v not in a.choices:
            print(f"{config}: {a.dest} = {v!r}, expected one of {a.choices}", file=sys.stderr)
            return 1

    recs, models, sources = load_answers()
    key = "cmean" if args.confidence == "mean" else "cmin"
    rec: dict = collections.defaultdict(dict)
    for t in TIERS:
        for r in recs[t]:
            rec[r["index"]][t] = {**r, "conf": r[key]}
    rec = {i: v for i, v in rec.items() if set(TIERS) <= set(v)}
    acc = {t: st.mean(rec[i][t]["correct"] for i in rec) for t in TIERS}
    if not all(acc[a] <= acc[b] for a, b in zip(TIERS, TIERS[1:])):
        print(f"ladder is not accuracy-monotonic: {acc} -- skips are unsafe here", file=sys.stderr)
        return 1

    policies = args.policies.split(",")
    if "stepwise" not in policies or not set(policies) <= set(POLICIES):
        print(f"--policies must include stepwise and come from {POLICIES}", file=sys.stderr)
        return 1
    if args.load_trace and args.load_sigma:
        print("--load-trace replays a real load; --load-sigma drifts the synthetic average day", file=sys.stderr)
        return 1
    pub = published_rates(args.accounting)
    pub["onu"]["dec"] *= args.onu_scale
    factor = olt_factor(args.boundary, args.accounting)
    curve = OltCurve(olt_runs()[1]["batch_curve"], factor)
    E = Energy(curve, pub, args.accounting)
    trace = TraceLoad(args.load_trace, args.train_days) if args.load_trace else None
    if args.surge_factor != 1 and not trace:
        print("--surge-factor adds events to a replayed trace: use it with --load-trace", file=sys.stderr)
        return 1
    surges = Surges(args.surge_factor, args.surge_per_day, args.surge_hours, 24 * trace.days, args.seed) if trace else None
    stream = build_stream(sorted(rec), args.days, args.per_day, args.seed, args.households, trace)
    times = np.array([t for _, t, _ in stream])
    drift = list(surges.mult(times)) if trace else load_noise(stream, args.load_sigma, args.load_tau, args.seed)
    rel = trace.rel if trace else load_shape

    def load_at(ts, peak):
        """The OLT's true load at any times, for its own reports and broadcasts."""
        if trace:
            return peak * trace.rel_vec(ts) * surges.mult(ts)
        return peak * load_shape_vec(ts) * np.interp(ts, times, drift)
    shape24 = [float(x) for x in trace.shape / trace.shape.max()] if trace else [load_shape(h + 0.5) for h in range(24)]
    sched_key = (lambda t: (trace.daytype(t), int(t) % 24)) if trace else (lambda t: (0, int(t) % 24))
    betas = [float(b) for b in args.betas.split(",")]
    targets = [float(t) for t in args.acc_targets.split(",")]
    peaks = [float(p) for p in args.peak_loads.split(",")]
    mean_tok = {t: (st.mean(rec[i][t]["tp"] for i in rec), st.mean(rec[i][t]["tg"] for i in rec)) for t in TIERS}
    fixed = {t: pub[t]["pf"] * mean_tok[t][0] + pub[t]["dec"] * mean_tok[t][1] for t in ("user", "onu")}

    print(f"answers: {', '.join(sources)}")
    span = (f"{trace.days - trace.train_days} days of BurstGPT '{args.load_trace}' after {trace.train_days} "
            f"calibration days, surges x{args.surge_factor:g} ({len(surges.start)} events)"
            if trace else f"{args.days} days, load drift sigma {args.load_sigma:g} (tau {args.load_tau:g} h)")
    print(f"n={len(rec)} queries, {len(stream)} arrivals over {span}; {args.households} household(s) x "
          f"{args.per_day}/day; confidence exp({args.confidence}); "
          f"OLT boundary {args.boundary} (x{factor:.2f}), {args.accounting} accounting; OLT reports "
          f"{'its ' + format(args.report_window, 'g') + '-min mean' if args.report == 'window' else 'per query'}; "
          f"delta={args.delta}, window={args.window}")
    print("standalone accuracy: " + "  ".join(f"{t} {acc[t]:.3f}" for t in TIERS))
    print(f"per query: user {fixed['user']:.1f} J, ONU {fixed['onu']:.1f} J (fixed)\n")

    rows, hourly, configs, frontiers = [], [], [], []
    for peak in peaks:
        rng = np.random.default_rng([args.seed, int(peak * 1000)])   # same batches for every beta and policy
        loads = [peak * rel(t) * m for (_, t, _), m in zip(stream, drift)]
        batches = [1 + int(k) for k in rng.poisson(loads)]
        if trace:
            static_rates = calibrate(trace, peak, E, args.stale_factor, surges.mult)
        else:
            def day_average(scale):
                rs = [E.expected_olt(L * scale) for L in loads]
                return st.mean(r[0] for r in rs), st.mean(r[1] for r in rs)
            static_rates = {"static": day_average(1.0), "stale_low": day_average(1 / args.stale_factor),
                            "stale_high": day_average(args.stale_factor)}
            by_hour = collections.defaultdict(list)
            for (_, t, _), L in zip(stream, loads):
                by_hour[int(t) % 24].append(E.expected_olt(L))
            static_rates["schedule"] = {(0, h): (st.mean(r[0] for r in by_hour[h]), st.mean(r[1] for r in by_hour[h]))
                                        for h in range(24)}
        def reporter(stream_id):
            return OltReporter(lambda ts: load_at(ts, peak), E, curve, mean_tok["olt"][1], args.report_window,
                               max(loads), np.random.default_rng([args.seed, int(peak * 1000), stream_id]))
        reports = (reporter(2).means_at(times)
                   if args.report == "window" and "piggyback" in policies else None)
        bcast = reporter(3).broadcasts(times, args.broadcast_interval_s) if "broadcast" in policies else None
        olt_q = [E.rates("olt", b)[0] * rec[qi]["olt"]["tp"] + E.rates("olt", b)[1] * rec[qi]["olt"]["tg"]
                 for (qi, _, _), b in zip(stream, batches)]
        # Little's law: load = arrival rate x service time, service at the peak's batch.
        svc = curve.service_s(1 + peak, mean_tok["olt"][1])
        cfg = {
            "peak_load": peak,
            "peak_arrivals_per_h": peak / svc * 3600,
            "trough_load": peak * min(shape24),
            "max_load_in_service": max(loads),                     # the curve is measured to batch 64
            "share_arrivals_over_64": float(np.mean(np.array(loads) > 63)),
            "olt_service_s_at_peak": svc,
            "olt_J_per_query_mean": st.mean(olt_q),
            "olt_J_per_query_peak_hour": E.expected_olt(peak)[0] * mean_tok["olt"][0] + E.expected_olt(peak)[1] * mean_tok["olt"][1],
            "olt_J_per_query_trough_hour": (lambda r: r[0] * mean_tok["olt"][0] + r[1] * mean_tok["olt"][1])(
                E.expected_olt(peak * min(shape24))),
            "olt_hourly_J_per_query": [(lambda r: r[0] * mean_tok["olt"][0] + r[1] * mean_tok["olt"][1])(
                E.expected_olt(peak * shape24[h])) for h in range(24)],
        }
        configs.append(cfg)
        print(f"peak load {peak:g}: ~{cfg['peak_arrivals_per_h']:,.0f} OLT queries/h at peak; OLT J/query "
              f"{cfg['olt_J_per_query_trough_hour']:.0f} (trough) to {cfg['olt_J_per_query_peak_hour']:.0f} (peak), "
              f"all-OLT mean {cfg['olt_J_per_query_mean']:.0f}")

        block = []
        for beta in betas:
            for policy in policies:
                m, hr = run(rec, stream, batches, loads, beta, policy, args, E, static_rates, reports, sched_key, bcast)
                block.append({"peak_load": peak, "beta": beta, "policy": policy, **m})
                if not args.no_hourly:
                    hourly.append({"peak_load": peak, "beta": beta, "policy": policy, "hours": hr})
        iso_accuracy(block)
        rows += block
        for p in policies:
            frontiers.append({"peak_load": peak, "policy": p,
                              "J_at_accuracy": frontier([r for r in block if r["policy"] == p], targets)})

    print(f"\n{'load':>5} {'beta':>5} {'policy':>10} {'acc':>7} {'J/query':>8} {'saving':>7} {'fwd':>6} {'skip':>6} "
          f"{'rate err':>8} | " + " ".join(f"{t:>6}" for t in TIERS))
    for r in rows:
        print(f"{r['peak_load']:5g} {r['beta']:5.1f} {r['policy']:>10} {r['accuracy']:7.4f} {r['J_per_query']:8.1f} "
              f"{r['saving_same_accuracy']:7.1%} {r['forwarded_on_arrival']:6.1%} {r['skipped_on_escalation']:6.1%} "
              f"{r['olt_rate_error']:8.1%} | " + " ".join(f"{r['final_' + t]:6.1%}" for t in TIERS))
        if r["policy"] == "oracle" and r["beta"] == betas[-1]:
            print()

    print("median saving at equal accuracy, across beta:")
    for peak in peaks:
        med = {p: np.nanmedian([r["saving_same_accuracy"] for r in rows
                                if r["peak_load"] == peak and r["policy"] == p] or [float("nan")])
               for p in policies if p != "stepwise"}
        print(f"  peak load {peak:5g}: " + "  ".join(f"{p} {v:6.1%}" for p, v in med.items()))

    print("\nJ/query at equal accuracy, each policy's own frontier (- = accuracy out of its range):")
    print(f"{'load':>5} {'acc':>5} " + " ".join(f"{p:>10}" for p in policies))
    for peak in peaks:
        for a in [f"{t:.2f}" for t in targets if round(t, 2) in (0.70, 0.80)]:
            vals = {f["policy"]: f["J_at_accuracy"][a] for f in frontiers if f["peak_load"] == peak}
            print(f"{peak:5g} {a:>5} " + " ".join(f"{vals[p]:10.1f}" if vals[p] is not None else f"{'-':>10}"
                                                  for p in policies))

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = (args.boundary + (f"_onu{args.onu_scale:g}" if args.onu_scale != 1 else "")
           + (f"_trace-{args.load_trace}" if trace else "")
           + (f"_surge{args.surge_factor:g}" if trace and args.surge_factor != 1 else "")
           + ("_shared" if args.shared_stats else "")
           + (f"_hh{args.households}x{args.per_day}" if args.households > 1 else "")
           + (f"_sigma{args.load_sigma:g}" if args.load_sigma else "")
           + ("_marginal" if args.accounting == "marginal" else "")
           + ("_window" if args.report == "window" else "")
           + (f"_ra{args.rate_alpha:g}" if args.rate_alpha is not None else ""))
    out = args.out or RESULTS / f"sim_piggyback_{tag}_{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "answers": sources,
                   "models": models, "olt_factor": factor, "accuracy": acc,
                   "fixed_J_per_query": fixed, "mean_tokens": mean_tok, "burstgpt": BURSTGPT,
                   "load_trace": ({"column": trace.column, "train_days": trace.train_days, "days": trace.days,
                                   "weekend_days_mod7": sorted(trace.weekend), "average_day": shape24}
                                  if trace else None),
                   "surges": ({"factor": args.surge_factor, "per_day": args.surge_per_day, "hours": args.surge_hours,
                               "events_start_end_sign": surges.events()} if trace else None),
                   "configs": configs, "rows": rows, "frontiers": frontiers, "hourly": hourly}, f, indent=1)
    print(f"wrote {out} and {out.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
