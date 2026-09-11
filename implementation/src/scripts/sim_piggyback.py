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

Seven policies. The last five share that rule and differ ONLY in which OLT
rate they see, so the comparison isolates what the packets are worth:

    stepwise    plain RecServe, the baseline
    skip_onu    plain RecServe on a two-tier chain, user -> OLT: the ONU is never
                used. The control for "is the gain just from dropping the ONU?"
    static      the OLT's day-average rate, fixed -- a config shipped once, calibrated
                on exactly the load it then meets (the best a fixed config can do)
    stale_low   the same, but calibrated when load was --stale-factor times lower
    stale_high  ... or --stale-factor times higher: a config that has gone stale
    piggyback   learned from the packets (EWMA of the rates reported)
    oracle      the true expected rate at this moment -- the upper bound on live information

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

Usage:
    python src/scripts/sim_piggyback.py                      # defaults below
    python src/scripts/sim_piggyback.py --boundary gpu       # GPU-card OLT, for comparison
    python src/scripts/sim_piggyback.py --onu-scale 0.2      # a 5x more efficient ONU
Writes results/energy_tests/sim_piggyback_<UTC>.csv (one row per load, beta,
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # implementation/src
from energy.three_tier import (RESULTS, TIERS, OltCurve, load_answers,  # noqa: E402
                               olt_factor, olt_runs, published_rates)

TOP = len(TIERS) - 1
POLICIES = ("stepwise", "skip_onu", "static", "stale_low", "stale_high", "piggyback", "oracle")
ACC_TARGETS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)   # where the policies' frontiers are compared
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


class Energy:
    """True energy rates per tier; the OLT's depend on its batch."""

    def __init__(self, curve: OltCurve, pub: dict):
        self.curve, self.pub = curve, pub
        self._cache: dict = {}

    def rates(self, tier: str, batch: int) -> tuple[float, float]:
        if tier == "olt":
            return self.curve.rates(batch)
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
                a, b = self.curve.rates(1 + k)
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

    def __init__(self, alpha: float, warmup: int):
        self.alpha, self.warmup = alpha, warmup
        self.rates: dict[str, tuple[float, float]] = {}   # EWMA of reported (pf, dec)
        self.tokens: dict[str, float] = {}                 # EWMA tokens generated there
        self.p_on: dict[str, float] = {}                   # EWMA P(query went beyond it)
        self.n: collections.Counter = collections.Counter()

    def absorb(self, packet: dict, final: str) -> None:
        for tier, (jpf, jdec, gen) in packet.items():
            went_on = float(TIERS.index(final) > TIERS.index(tier))
            self._ewma(self.tokens, tier, gen)
            self._ewma(self.p_on, tier, went_on)
            old = self.rates.get(tier)
            self.rates[tier] = (jpf, jdec) if old is None else (
                (1 - self.alpha) * old[0] + self.alpha * jpf, (1 - self.alpha) * old[1] + self.alpha * jdec)
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


def run(rec, stream, batches, loads, beta, policy, args, E, static_rates):
    history = {t: Window(args.window) for t in TIERS}
    learners = {t: Learned(args.alpha, args.warmup) for t in TIERS[:TOP]}  # only tiers that can decide
    energy = correct = forwarded = skipped = 0.0
    rate_err = []
    final_at = collections.Counter()
    hourly = [[0.0, 0, 0, 0, 0] for _ in range(24)]      # joules, queries, final at user/onu/olt

    for (qi, t_h), b, load in zip(stream, batches, loads):
        prompt = rec[qi]["user"]["tp"]
        packet: dict[str, tuple[float, float, int]] = {}
        spent = 0.0
        i = 0
        rates = None
        while True:
            t = TIERS[i]
            L = learners.get(t)
            decide = policy not in ("stepwise", "skip_onu") and L is not None and L.ready(TIERS[i:])
            if decide:
                rates = {u: L.rates[u] for u in TIERS[i:]}
                if policy in STATIC:
                    rates["olt"] = static_rates[policy]
                elif policy == "oracle":
                    rates["olt"] = E.expected_olt(load)
                else:
                    true_dec = E.expected_olt(load)[1]
                    rate_err.append(abs(rates["olt"][1] - true_dec) / true_dec)
                j = on_arrival(i, prompt, rates, L, args.delta)
                if j != i:
                    forwarded += 1
                    i = j
                    continue

            d = rec[qi][t]
            jpf, jdec = E.rates(t, b)
            spent += jpf * d["tp"] + jdec * d["tg"]
            packet[t] = (jpf, jdec, d["tg"])

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


def build_stream(indices, days, per_day, seed):
    """Time-ordered arrivals at the user tier, hour h getting a share proportional to BurstGPT."""
    rng = random.Random(seed)
    total = sum(BURSTGPT)
    out = []
    for day in range(days):
        times = []
        for h in range(24):
            times += [24 * day + h + rng.random() for _ in range(round(per_day * BURSTGPT[h] / total))]
        out += [(rng.choice(indices), t) for t in sorted(times)]
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


def frontier(rows: list[dict]) -> dict:
    """One policy's J/query at each target accuracy, along its beta sweep.

    Only Pareto points are kept (no other beta is both more accurate and
    cheaper), then J is interpolated between them; a target outside the
    policy's accuracy range gets None.
    """
    pareto, best = [], float("inf")
    for a, j in sorted(((r["accuracy"], r["J_per_query"]) for r in rows), reverse=True):
        if j < best:
            pareto.append((a, j))
            best = j
    pareto.sort()
    xs, ys = [a for a, _ in pareto], [j for _, j in pareto]
    return {f"{t:.2f}": (float(np.interp(t, xs, ys)) if xs[0] <= t <= xs[-1] else None) for t in ACC_TARGETS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--betas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--peak-loads", default="0.5,1,2,4,8,16,32",
                    help="OLT offered load at the busiest hour: mean queries in service")
    ap.add_argument("--boundary", choices=("system", "system-low", "gpu"), default="system",
                    help="OLT energy boundary (energy_tests.md §7): whole system at PUE 1.54 (default), at 1.09, or GPU card")
    ap.add_argument("--confidence", choices=("mean", "min"), default="mean",
                    help="exp(mean logprob), RecServe's definition; or exp(min logprob)")
    ap.add_argument("--onu-scale", type=float, default=1.0,
                    help="multiply the ONU's J per generated token (1.11, whole board, idle included): "
                         "a sensitivity for a more efficient ONU accelerator")
    ap.add_argument("--stale-factor", type=float, default=4.0,
                    help="how far off the stale configs' assumed load is (x and 1/x)")
    ap.add_argument("--delta", type=float, default=0.0, help="relative skip margin")
    ap.add_argument("--window", type=int, default=1000,
                    help="confidence history per tier (RecServe paper recommends 300-1000)")
    ap.add_argument("--alpha", type=float, default=0.05, help="EWMA weight for learned stats")
    ap.add_argument("--warmup", type=int, default=20, help="reports per tier before deciding")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--per-day", type=int, default=2000, help="queries per day at the user tier")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, help="CSV path (default results/energy_tests/sim_piggyback_<UTC>.csv)")
    args = ap.parse_args()

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

    pub = published_rates()
    pub["onu"]["dec"] *= args.onu_scale
    factor = olt_factor(args.boundary)
    curve = OltCurve(olt_runs()[1]["batch_curve"], factor)
    E = Energy(curve, pub)
    stream = build_stream(sorted(rec), args.days, args.per_day, args.seed)
    betas = [float(b) for b in args.betas.split(",")]
    peaks = [float(p) for p in args.peak_loads.split(",")]
    mean_tok = {t: (st.mean(rec[i][t]["tp"] for i in rec), st.mean(rec[i][t]["tg"] for i in rec)) for t in TIERS}
    fixed = {t: pub[t]["pf"] * mean_tok[t][0] + pub[t]["dec"] * mean_tok[t][1] for t in ("user", "onu")}

    print(f"answers: {', '.join(sources)}")
    print(f"n={len(rec)} queries, {len(stream)} arrivals over {args.days} days; confidence exp({args.confidence}); "
          f"OLT boundary {args.boundary} (x{factor:.2f}); delta={args.delta}, window={args.window}")
    print("standalone accuracy: " + "  ".join(f"{t} {acc[t]:.3f}" for t in TIERS))
    print(f"per query: user {fixed['user']:.1f} J, ONU {fixed['onu']:.1f} J (fixed)\n")

    rows, hourly, configs, frontiers = [], [], [], []
    for peak in peaks:
        rng = np.random.default_rng([args.seed, int(peak * 1000)])   # same batches for every beta and policy
        loads = [peak * load_shape(t) for _, t in stream]
        batches = [1 + int(k) for k in rng.poisson(loads)]
        def day_average(scale):
            rs = [E.expected_olt(L * scale) for L in loads]
            return st.mean(r[0] for r in rs), st.mean(r[1] for r in rs)
        static_rates = {"static": day_average(1.0), "stale_low": day_average(1 / args.stale_factor),
                        "stale_high": day_average(args.stale_factor)}
        olt_q = [curve.rates(b)[0] * rec[qi]["olt"]["tp"] + curve.rates(b)[1] * rec[qi]["olt"]["tg"]
                 for (qi, _), b in zip(stream, batches)]
        # Little's law: load = arrival rate x service time, service at the peak's batch.
        svc = curve.service_s(1 + peak, mean_tok["olt"][1])
        cfg = {
            "peak_load": peak,
            "peak_arrivals_per_h": peak / svc * 3600,
            "trough_load": peak * min(BURSTGPT) / max(BURSTGPT),
            "olt_service_s_at_peak": svc,
            "olt_J_per_query_mean": st.mean(olt_q),
            "olt_J_per_query_peak_hour": E.expected_olt(peak)[0] * mean_tok["olt"][0] + E.expected_olt(peak)[1] * mean_tok["olt"][1],
            "olt_J_per_query_trough_hour": (lambda r: r[0] * mean_tok["olt"][0] + r[1] * mean_tok["olt"][1])(
                E.expected_olt(peak * min(BURSTGPT) / max(BURSTGPT))),
            "olt_hourly_J_per_query": [(lambda r: r[0] * mean_tok["olt"][0] + r[1] * mean_tok["olt"][1])(
                E.expected_olt(peak * load_shape(h + 0.5))) for h in range(24)],
        }
        configs.append(cfg)
        print(f"peak load {peak:g}: ~{cfg['peak_arrivals_per_h']:,.0f} OLT queries/h at peak; OLT J/query "
              f"{cfg['olt_J_per_query_trough_hour']:.0f} (trough) to {cfg['olt_J_per_query_peak_hour']:.0f} (peak), "
              f"all-OLT mean {cfg['olt_J_per_query_mean']:.0f}")

        block = []
        for beta in betas:
            for policy in POLICIES:
                m, hr = run(rec, stream, batches, loads, beta, policy, args, E, static_rates)
                block.append({"peak_load": peak, "beta": beta, "policy": policy, **m})
                hourly.append({"peak_load": peak, "beta": beta, "policy": policy, "hours": hr})
        iso_accuracy(block)
        rows += block
        for p in POLICIES:
            frontiers.append({"peak_load": peak, "policy": p,
                              "J_at_accuracy": frontier([r for r in block if r["policy"] == p])})

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
               for p in POLICIES[1:]}
        print(f"  peak load {peak:5g}: " + "  ".join(f"{p} {v:6.1%}" for p, v in med.items()))

    print("\nJ/query at equal accuracy, each policy's own frontier (- = accuracy out of its range):")
    print(f"{'load':>5} {'acc':>5} " + " ".join(f"{p:>10}" for p in POLICIES))
    for peak in peaks:
        for a in ("0.70", "0.80"):
            vals = {f["policy"]: f["J_at_accuracy"][a] for f in frontiers if f["peak_load"] == peak}
            print(f"{peak:5g} {a:>5} " + " ".join(f"{vals[p]:10.1f}" if vals[p] is not None else f"{'-':>10}"
                                                  for p in POLICIES))

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = args.boundary + (f"_onu{args.onu_scale:g}" if args.onu_scale != 1 else "")
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
                   "configs": configs, "rows": rows, "frontiers": frontiers, "hourly": hourly}, f, indent=1)
    print(f"wrote {out} and {out.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
