#!/usr/bin/env python3
"""The simulator: households' queries through the phone -> ONU -> OLT cascade, under one or more policies.

ROLE IN THE PIPELINE
    energy/three_tier.py (tier prices, answers)  +  data/load_traces/ (OLT load)
        -> simulate/simulate.py  ->  results/study/ (via run_study.sh) or results/adhoc/

WHAT IS SIMULATED
    Queries arrive at households' phones. RecServe decides WHETHER a query
    escalates, unchanged: a tier passes a query on when its answer's confidence is
    below the beta-quantile of the tier's recent confidences (routing.Window). The
    energy-aware policies also decide WHERE it goes, at two moments:

        on arrival     run the query here, or forward it straight to a tier above?
        on escalation  send it to the next tier up, or skip to a later one?

    Both pick the lowest expected cost to completion (routing.cost_to_completion),

        C(j) = E_j + p_j * C(j+1)        E_j = J_pf_j * prompt + J_dec_j * tokens_j

    where p_j is how often a query that reached j went further. The phone's and
    the ONU's costs are fixed; the OLT's falls as its batch grows, so it depends on
    the OLT's load right now. The policies differ ONLY in where they get it from:

        recserve         RecServe as published, always one tier up: the baseline
        recserve_no_onu  RecServe on phone -> OLT only (control: is it just dropping the ONU?)
        static_day       one fixed OLT rate for the whole day, calibrated in advance
        static_hour      one OLT rate per hour of day, weekday or weekend: a timetable
        broadcast        the OLT's own 5-minute mean, broadcast on the PON's downstream
                         to every ONU every --broadcast-interval-s: the proposal
        oracle           the true expected rate right now: the ceiling for any live signal
        piggyback        the OLT's rate heard only on the household's own answers (ablation)
        stale_low/high   static_day calibrated at 1/4 or 4x the real load (a stale config)

    Every answer is replayed from the recorded collection (every tier answered all
    1,319 GSM8K test questions), so no model runs here. Accuracy and energy per
    query come out per (OLT load, beta, policy); policies are compared at EQUAL
    ACCURACY (frontier.py), because skipping to a better tier changes accuracy too.

MODULES
    olt_load.py    the OLT's load over time: BurstGPT replayed (TraceLoad), unforeseen
                   surges (Surges), or a synthetic average day with drift; the
                   static_day / static_hour tables (calibrate)
    olt_energy.py  true energy rates (Energy) and the OLT's report (OltReporter)
    routing.py     RecServe's window and the energy-aware routing rule
    cascade.py     one run of one policy over the query stream (run, build_stream)
    frontier.py    J per query at equal accuracy
    simulate.py    this file: settings, the loop over loads, betas and policies, output

TWO KINDS OF LOAD
    --load-trace conversation|all   BurstGPT's hourly requests replayed as the OLT's
        load: static_day and static_hour are calibrated on its first --train-days,
        and every policy runs on the days after. --surge-factor adds unforeseen
        surges and dips. --households splits the cascade's own queries over that
        many households (--per-day each). This is the case study (config/study.yaml).
    no trace (config/simulation.yaml)   every day is BurstGPT's average day,
        optionally times a drift (--load-sigma, --load-tau); the static tables are
        then calibrated on the load they meet. Kept from the exploration (§8.2-8.4).

ACCOUNTING
    average   a query pays its share of the OLT batch's energy (Google's per-prompt view)
    marginal  a query pays only what it adds; no tier is charged for being on
              (the case study's primary accounting, energy_tests.md §8.3, §8.5)

SETTINGS
    Every default is in config/simulation.yaml, which says what each setting does
    and where its value comes from; config/study.yaml extends it for the case
    study; a flag overrides one value for one run.

Usage:
    python src/simulate/simulate.py                                   # config/simulation.yaml (~45 s)
    python src/simulate/simulate.py --config config/study.yaml --seed 7 --surge-factor 3   # one study run (~10 min)
    python src/simulate/simulate.py --config config/study.yaml \
        --policies recserve,static_day,static_hour,broadcast,oracle --peak-loads 8
Writes results/adhoc/sim_<tag>_<UTC>.csv (one row per load, beta, policy) and .json
(the same plus each policy's accuracy-energy frontier, the settings and the inputs),
or wherever --out says.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # implementation/src
from energy.three_tier import (ROOT, TIERS, OltCurve, load_answers,  # noqa: E402
                               olt_factor, olt_reference, published_rates, published_speeds)
from cascade import POLICIES, build_stream, run  # noqa: E402
from frontier import frontier, iso_accuracy  # noqa: E402
from olt_energy import Energy, OltReporter  # noqa: E402
from olt_load import BURSTGPT, Surges, TraceLoad, calibrate, load_noise, load_shape, load_shape_vec  # noqa: E402

CONFIG = ROOT / "config" / "simulation.yaml"
ADHOC = ROOT / "results" / "adhoc"               # default output of a one-off run


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
    # Every default comes from config/simulation.yaml (what each setting means and
    # where its value comes from is written there); a flag overrides it for one run.
    """Read the settings, build the query stream and the OLT's load, run every policy at every
    OLT peak load and beta, print the tables and write the CSV and JSON.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=CONFIG, help="settings file (default config/sim_piggyback.yaml)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[pre])
    ap.add_argument("--betas", help="comma-separated RecServe beta values")
    ap.add_argument("--peak-loads", help="comma-separated OLT loads at the busiest hour, in queries in service")
    ap.add_argument("--boundary", choices=("system", "system-low", "gpu"), help="where the OLT's energy is counted (§7)")
    ap.add_argument("--confidence", choices=("mean", "min"), help="exp(mean logprob), RecServe's, or exp(min logprob)")
    ap.add_argument("--onu-scale", type=float, help="multiply the ONU's J per generated token")
    ap.add_argument("--olt-scale", type=float, help="multiply the OLT's energy after the boundary conversion")
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
    ap.add_argument("--household-shape", choices=("trace", "flat"),
                    help="households' queries over the day: like the OLT's load, or flat")
    ap.add_argument("--load-trace", choices=("conversation", "api", "all"),
                    help="replay BurstGPT's hourly requests as the OLT's load (data/load_traces/)")
    ap.add_argument("--train-days", type=int, help="trace days the configs and the schedule are calibrated on")
    ap.add_argument("--surge-factor", type=float, help="load multiplier of an unforeseen event (1 = none)")
    ap.add_argument("--surge-per-day", type=float, help="expected unforeseen events a day")
    ap.add_argument("--surge-hours", type=float, help="length of an unforeseen event, hours")
    ap.add_argument("--broadcast-interval-s", type=float, help="seconds between the OLT's broadcasts")
    ap.add_argument("--shared-stats", action=argparse.BooleanOptionalAction,
                    help="pool answer lengths and escalation rates across households")
    ap.add_argument("--policies", help="comma-separated subset; must include recserve")
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
    if "recserve" not in policies or not set(policies) <= set(POLICIES):
        print(f"--policies must include recserve and come from {POLICIES}", file=sys.stderr)
        return 1
    if args.load_trace and args.load_sigma:
        print("--load-trace replays a real load; --load-sigma drifts the synthetic average day", file=sys.stderr)
        return 1
    pub = published_rates(args.accounting)
    pub["onu"]["dec"] *= args.onu_scale
    factor = olt_factor(args.boundary, args.accounting) * args.olt_scale
    curve = OltCurve(olt_reference()["batch_curve"], factor)
    E = Energy(curve, pub, args.accounting, published_speeds())
    trace = TraceLoad(args.load_trace, args.train_days) if args.load_trace else None
    if args.household_shape == "flat" and not trace:
        print("--household-shape flat applies to a replayed trace: use it with --load-trace", file=sys.stderr)
        return 1
    if args.surge_factor != 1 and not trace:
        print("--surge-factor adds events to a replayed trace: use it with --load-trace", file=sys.stderr)
        return 1
    surges = Surges(args.surge_factor, args.surge_per_day, args.surge_hours, 24 * trace.days, args.seed) if trace else None
    stream = build_stream(sorted(rec), args.days, args.per_day, args.seed, args.households, trace,
                          args.household_shape)
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
                """Synthetic mode: the OLT's mean expected rate over the run, at `scale` times its load."""
                rs = [E.expected_olt(L * scale) for L in loads]
                return st.mean(r[0] for r in rs), st.mean(r[1] for r in rs)
            static_rates = {"static_day": day_average(1.0), "stale_low": day_average(1 / args.stale_factor),
                            "stale_high": day_average(args.stale_factor)}
            by_hour = collections.defaultdict(list)
            for (_, t, _), L in zip(stream, loads):
                by_hour[int(t) % 24].append(E.expected_olt(L))
            static_rates["static_hour"] = {(0, h): (st.mean(r[0] for r in by_hour[h]), st.mean(r[1] for r in by_hour[h]))
                                        for h in range(24)}
        def reporter(stream_id):
            """The OLT's own-mean reporter at this peak load; stream_id keeps each report stream's randomness separate."""
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
               for p in policies if p != "recserve"}
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
           + (f"_olt{args.olt_scale:g}" if args.olt_scale != 1 else "")
           + ("_flat" if args.household_shape == "flat" else "")
           + (f"_trace-{args.load_trace}" if trace else "")
           + (f"_surge{args.surge_factor:g}" if trace and args.surge_factor != 1 else "")
           + ("_shared" if args.shared_stats else "")
           + (f"_hh{args.households}x{args.per_day}" if args.households > 1 else "")
           + (f"_sigma{args.load_sigma:g}" if args.load_sigma else "")
           + ("_marginal" if args.accounting == "marginal" else "")
           + ("_window" if args.report == "window" else "")
           + (f"_ra{args.rate_alpha:g}" if args.rate_alpha is not None else ""))
    out = args.out or ADHOC / f"sim_{tag}_{stamp}.csv"
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
