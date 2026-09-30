#!/usr/bin/env python3
"""Simulate households' queries through the phone -> ONU -> OLT cascade, under each policy.

Every answer is replayed from the recorded collection, so no model runs here. Results come
out per OLT peak load, RecServe beta and policy, and policies are compared at equal
accuracy. Usage and the meaning of every setting: README.md and config/simulation.yaml.
"""

# pylint: disable=wrong-import-position

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from cascade import POLICIES, run, RunSetup
from energy.three_tier import (
    load_answers,
    olt_factor,
    olt_reference,
    OltCurve,
    published_rates,
    published_speeds,
    ROOT,
    TIERS,
)
from frontier import frontier, iso_accuracy
from load_modes import SyntheticMode, TraceMode
from olt_energy import Energy, OltReporter
from olt_load import BURSTGPT

CONFIG = ROOT / "config" / "simulation.yaml"
"""The default settings file."""

ADHOC = ROOT / "results" / "adhoc"
"""Where a one-off run writes when --out is not given."""


def main() -> int:
    """Run every policy at every OLT peak load and beta, print the tables and write the results.

    The steps, each a function below: read_settings, answers_by_question,
    check_combinations, build_prices; then, per OLT peak load, prepare_peak, olt_summary
    and run_policies; finally print_tables and write_results.

    Returns:
        The process exit code: 0, or 1 when the settings are unusable.
    """
    # --- Settings and inputs ---
    s = read_settings()
    if s is None:
        return 1
    answers, accuracy, models, sources = answers_by_question(s.confidence)
    if answers is None:
        return 1
    policies = s.policies.split(",")
    if not check_combinations(s, policies):
        return 1
    published, factor, curve, prices = build_prices(s)
    mode = (TraceMode if s.load_trace else SyntheticMode)(s, sorted(answers))
    betas = [float(b) for b in s.betas.split(",")]
    targets = [float(t) for t in s.acc_targets.split(",")]
    peaks = [float(p) for p in s.peak_loads.split(",")]
    # each tier's mean prompt and answer length, and the phone's and ONU's fixed J per query
    mean_tokens = {
        t: (
            st.mean(answers[q][t]["tp"] for q in answers),
            st.mean(answers[q][t]["tg"] for q in answers),
        )
        for t in TIERS
    }
    fixed = {
        t: published[t]["pf"] * mean_tokens[t][0] + published[t]["dec"] * mean_tokens[t][1]
        for t in ("user", "onu")
    }
    print_header(s, sources, answers, mode, factor, accuracy, fixed)

    # --- Every OLT peak load, beta and policy ---
    rows, hourly, configs, frontiers = [], [], [], []
    for peak in peaks:
        setup = prepare_peak(peak, s, mode, answers, prices, curve, mean_tokens, policies)
        configs.append(olt_summary(peak, mode, setup, prices, curve, mean_tokens))
        print_peak_line(configs[-1])
        block, block_hourly = run_policies(setup, peak, betas, policies)
        iso_accuracy(block)  # adds each row's saving against RecServe at equal accuracy
        rows += block
        hourly += block_hourly
        for p in policies:
            frontiers.append(
                {
                    "peak_load": peak,
                    "policy": p,
                    "J_at_accuracy": frontier([r for r in block if r["policy"] == p], targets),
                }
            )

    # --- Output ---
    print_tables(rows, betas, peaks, policies, frontiers, targets)
    write_results(
        s,
        mode,
        rows,
        hourly,
        configs,
        frontiers,
        sources,
        models,
        factor,
        accuracy,
        fixed,
        mean_tokens,
    )
    return 0


# ==========================================
# Settings and inputs
# ==========================================
def load_config(path: Path) -> dict:
    """Read a settings file, flattened into argparse defaults (lists become comma strings).

    A file may say `extends: <file>`: that file's values come first, this one's on top.

    Args:
        path: The settings file.

    Returns:
        {setting: value}.
    """
    with open(path, encoding="utf-8") as f:
        groups = yaml.safe_load(f)
    base = groups.pop("extends", None)
    flat = load_config(path.parent / base) if base else {}
    for items in groups.values():
        for k, v in items.items():
            flat[k] = ",".join(str(x) for x in v) if isinstance(v, list) else v
    return flat


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the command line; every flag's default comes from the settings file.

    Returns:
        (the parser that only reads --config, the full parser).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config", type=Path, default=CONFIG, help="settings file (default config/simulation.yaml)"
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[pre]
    )
    add = ap.add_argument
    add("--betas", help="comma-separated RecServe beta values")
    add("--peak-loads", help="comma-separated OLT loads at the busiest hour, in queries in service")
    add(
        "--boundary",
        choices=("system", "system-low", "gpu"),
        help="where the OLT's energy is counted",
    )
    add(
        "--confidence",
        choices=("mean", "min"),
        help="exp(mean logprob), RecServe's, or exp(min logprob)",
    )
    add("--onu-scale", type=float, help="multiply the ONU's J per generated token")
    add("--olt-scale", type=float, help="multiply the OLT's energy after the boundary conversion")
    add(
        "--accounting",
        choices=("average", "marginal"),
        help="share of the batch, or what a query adds",
    )
    add("--load-sigma", type=float, help="log-sd of the synthetic load's drift (0 = none)")
    add("--load-tau", type=float, help="correlation time of that drift, hours")
    add(
        "--report",
        choices=("query", "window"),
        help="the OLT packet's rate: this batch's, or its recent mean",
    )
    add("--report-window", type=float, help="minutes the OLT averages over (--report window)")
    add(
        "--stale-factor",
        type=float,
        help="how far off the stale configs' assumed load is (x and 1/x)",
    )
    add("--delta", type=float, help="relative skip margin")
    add("--window", type=int, help="RecServe's confidence history per tier")
    add("--alpha", type=float, help="EWMA weight for learned answer lengths and escalation rates")
    add("--rate-alpha", type=float, help="EWMA weight for reported energy rates (unset: --alpha)")
    add("--warmup", type=int, help="reports per tier before a household decides")
    add("--days", type=int, help="days simulated on the synthetic average day")
    add("--per-day", type=int, help="queries per day at the user tier, per household")
    add("--households", type=int, help="households sharing the OLT, each learning on its own")
    add(
        "--household-shape",
        choices=("trace", "flat"),
        help="households' queries over the day: like the OLT's load, or flat",
    )
    add(
        "--load-trace",
        choices=("conversation", "api", "all"),
        help="replay BurstGPT's hourly requests as the OLT's load (data/load_traces/)",
    )
    add("--train-days", type=int, help="trace days the configs and the schedule are calibrated on")
    add("--surge-factor", type=float, help="load multiplier of an unforeseen event (1 = none)")
    add("--surge-per-day", type=float, help="expected unforeseen events a day")
    add("--surge-hours", type=float, help="length of an unforeseen event, hours")
    add("--broadcast-interval-s", type=float, help="seconds between the OLT's broadcasts")
    add(
        "--shared-stats",
        action=argparse.BooleanOptionalAction,
        help="pool answer lengths and escalation rates across households",
    )
    add("--policies", help="comma-separated subset; must include recserve")
    add("--acc-targets", help="comma-separated accuracies the frontiers are read at")
    add("--no-hourly", action="store_true", help="leave the per-hour detail out of the JSON")
    add("--seed", type=int)
    add("--out", type=Path, help="CSV path (default results/adhoc/sim_<tag>_<UTC>.csv)")
    return pre, ap


def read_settings():
    """Read the settings file's values, then any flags on top.

    Returns:
        The parsed settings, or None (after printing why) when the file is missing a
        setting, has an unknown one, or a value is not among a setting's choices.
    """
    # argparse has no public list of its options, hence _actions
    # pylint: disable=protected-access
    pre, ap = build_parser()
    config = pre.parse_known_args()[0].config
    defaults = load_config(config)
    dests = {a.dest for a in ap._actions} - {"help", "config", "out"}
    if set(defaults) != dests:
        print(
            f"{config}: missing {sorted(dests - set(defaults))}, unknown {sorted(set(defaults) - dests)}",
            file=sys.stderr,
        )
        return None
    ap.set_defaults(**defaults)
    settings = ap.parse_args()
    for a in ap._actions:  # argparse does not check defaults against choices
        v = getattr(settings, a.dest, None)
        if a.choices and v is not None and v not in a.choices:
            print(f"{config}: {a.dest} = {v!r}, expected one of {a.choices}", file=sys.stderr)
            return None
    return settings


def answers_by_question(confidence: str):
    """Load every tier's recorded answers, keyed by question.

    Each record gets a "conf" field, the confidence the cascade routes on. Only questions
    every tier answered are kept, and the tiers must get more accurate going up, or
    skipping one would be unsafe.

    Args:
        confidence: "mean" (exp of the mean token log-probability) or "min".

    Returns:
        (answers[question][tier], each tier's accuracy, models, source files), or four
        Nones (after printing why) when the tiers are not in accuracy order.
    """
    records, models, sources = load_answers()
    key = "cmean" if confidence == "mean" else "cmin"
    answers: dict = collections.defaultdict(dict)
    for t in TIERS:
        for r in records[t]:
            answers[r["index"]][t] = {**r, "conf": r[key]}
    answers = {q: v for q, v in answers.items() if set(TIERS) <= set(v)}
    accuracy = {t: st.mean(answers[q][t]["correct"] for q in answers) for t in TIERS}
    if not all(accuracy[a] <= accuracy[b] for a, b in zip(TIERS, TIERS[1:])):
        print(
            f"ladder is not accuracy-monotonic: {accuracy} -- skips are unsafe here",
            file=sys.stderr,
        )
        return None, None, None, None
    return answers, accuracy, models, sources


def check_combinations(s, policies) -> bool:
    """Check that the settings can go together.

    Args:
        s: The parsed settings.
        policies: The policies to run.

    Returns:
        True, or False (after printing why).
    """
    if "recserve" not in policies or not set(policies) <= set(POLICIES):
        print(f"--policies must include recserve and come from {POLICIES}", file=sys.stderr)
        return False
    if s.load_trace and s.load_sigma:
        print(
            "--load-trace replays a real load; --load-sigma drifts the synthetic average day",
            file=sys.stderr,
        )
        return False
    if s.household_shape == "flat" and not s.load_trace:
        print(
            "--household-shape flat applies to a replayed trace: use it with --load-trace",
            file=sys.stderr,
        )
        return False
    if s.surge_factor != 1 and not s.load_trace:
        print(
            "--surge-factor adds events to a replayed trace: use it with --load-trace",
            file=sys.stderr,
        )
        return False
    return True


def build_prices(s):
    """Build the true energy rates.

    Args:
        s: The parsed settings.

    Returns:
        (the phone's and ONU's published rates, the OLT's boundary factor, the OLT's curve,
        the true rates as an olt_energy.Energy).
    """
    published = published_rates(s.accounting)
    published["onu"]["dec"] *= s.onu_scale
    factor = olt_factor(s.boundary, s.accounting) * s.olt_scale
    curve = OltCurve(olt_reference()["batch_curve"], factor)
    return published, factor, curve, Energy(curve, published, s.accounting, published_speeds())


# ==========================================
# One OLT peak load
# ==========================================
def prepare_peak(peak, s, mode, answers, prices, curve, mean_tokens, policies) -> RunSetup:
    """Build what every run at this OLT peak load shares, so the runs differ only by policy.

    Args:
        peak: The OLT's load at its busiest hour.
        s: The parsed settings.
        mode: The load mode (TraceMode or SyntheticMode).
        answers: answers[question][tier].
        prices: True energy rates.
        curve: The OLT's measured curve.
        mean_tokens: Each tier's mean (prompt, answer) length.
        policies: The policies to run.

    Returns:
        The RunSetup: the same batches, static rates and OLT reports for every policy.
    """
    rng = np.random.default_rng([s.seed, int(peak * 1000)])
    loads = [peak * mode.relative_load(t) * m for (_, t, _), m in zip(mode.stream, mode.multiplier)]
    batches = [1 + int(k) for k in rng.poisson(loads)]  # the OLT batch each query would meet
    static_rates = mode.static_rates(peak, prices, loads)

    def reporter(stream_id):
        """Return the OLT's own-mean reporter; stream_id keeps each stream's randomness apart."""
        return OltReporter(
            lambda ts: mode.load_at(ts, peak),
            prices,
            curve,
            mean_tokens["olt"][1],
            s.report_window,
            max(loads),
            np.random.default_rng([s.seed, int(peak * 1000), stream_id]),
        )

    olt_reports = (
        reporter(2).means_at(mode.times)
        if s.report == "window" and "piggyback" in policies
        else None
    )
    broadcasts = (
        reporter(3).broadcasts(mode.times, s.broadcast_interval_s)
        if "broadcast" in policies
        else None
    )
    return RunSetup(
        answers=answers,
        stream=mode.stream,
        batches=batches,
        loads=loads,
        prices=prices,
        static_rates=static_rates,
        settings=s,
        olt_reports=olt_reports,
        schedule_key=mode.schedule_key,
        broadcasts=broadcasts,
    )


def olt_summary(peak, mode, setup, prices, curve, mean_tokens) -> dict:
    """Summarize what the OLT costs over the day at this peak load, for the printout and the JSON.

    Args:
        peak: The OLT's load at its busiest hour.
        mode: The load mode.
        setup: This peak load's RunSetup.
        prices: True energy rates.
        curve: The OLT's measured curve.
        mean_tokens: Each tier's mean (prompt, answer) length.

    Returns:
        The summary, one entry of the JSON's "configs".
    """

    def per_query(rates):
        """Return J per query at the OLT's mean prompt and answer length."""
        return rates[0] * mean_tokens["olt"][0] + rates[1] * mean_tokens["olt"][1]

    olt_query_joules = [
        prices.rates("olt", b)[0] * setup.answers[q]["olt"]["tp"]
        + prices.rates("olt", b)[1] * setup.answers[q]["olt"]["tg"]
        for (q, _, _), b in zip(setup.stream, setup.batches)
    ]
    service_s = curve.service_s(
        1 + peak, mean_tokens["olt"][1]
    )  # Little's law: load = arrivals x time
    return {
        "peak_load": peak,
        "peak_arrivals_per_h": peak / service_s * 3600,
        "trough_load": peak * min(mode.average_day),
        "max_load_in_service": max(setup.loads),  # the curve is measured to batch 64
        "share_arrivals_over_64": float(np.mean(np.array(setup.loads) > 63)),
        "olt_service_s_at_peak": service_s,
        "olt_J_per_query_mean": st.mean(olt_query_joules),
        "olt_J_per_query_peak_hour": prices.expected_olt(peak)[0] * mean_tokens["olt"][0]
        + prices.expected_olt(peak)[1] * mean_tokens["olt"][1],
        "olt_J_per_query_trough_hour": per_query(prices.expected_olt(peak * min(mode.average_day))),
        "olt_hourly_J_per_query": [
            per_query(prices.expected_olt(peak * mode.average_day[h])) for h in range(24)
        ],
    }


def run_policies(setup: RunSetup, peak, betas, policies):
    """Run every beta and policy at this peak load.

    Args:
        setup: This peak load's RunSetup.
        peak: The OLT's load at its busiest hour.
        betas: RecServe's beta values.
        policies: The policies to run.

    Returns:
        (result rows, their per-hour detail).
    """
    block, block_hourly = [], []
    for beta in betas:
        for policy in policies:
            metrics, hours = run(setup, beta, policy)
            block.append({"peak_load": peak, "beta": beta, "policy": policy, **metrics})
            if not setup.settings.no_hourly:
                block_hourly.append(
                    {"peak_load": peak, "beta": beta, "policy": policy, "hours": hours}
                )
    return block, block_hourly


# ==========================================
# Printout and files
# ==========================================
def print_header(s, sources, answers, mode, factor, accuracy, fixed) -> None:
    """Print the run's settings and inputs, before any policy runs."""
    reports = f"its {s.report_window:g}-min mean" if s.report == "window" else "per query"
    print(f"answers: {', '.join(sources)}")
    print(
        f"n={len(answers)} queries, {len(mode.stream)} arrivals over {mode.describe()}; "
        f"{s.households} household(s) x {s.per_day}/day; confidence exp({s.confidence}); "
        f"OLT boundary {s.boundary} (x{factor:.2f}), {s.accounting} accounting; OLT reports "
        f"{reports}; delta={s.delta}, window={s.window}"
    )
    print("standalone accuracy: " + "  ".join(f"{t} {accuracy[t]:.3f}" for t in TIERS))
    print(f"per query: user {fixed['user']:.1f} J, ONU {fixed['onu']:.1f} J (fixed)\n")


def print_peak_line(cfg) -> None:
    """Print one line per OLT peak load: its traffic and what an OLT query costs over the day."""
    print(
        f"peak load {cfg['peak_load']:g}: ~{cfg['peak_arrivals_per_h']:,.0f} OLT queries/h at peak; "
        f"OLT J/query {cfg['olt_J_per_query_trough_hour']:.0f} (trough) to "
        f"{cfg['olt_J_per_query_peak_hour']:.0f} (peak), all-OLT mean {cfg['olt_J_per_query_mean']:.0f}"
    )


def print_tables(rows, betas, peaks, policies, frontiers, targets) -> None:
    """Print every result row, the median saving per peak load, and J per query at equal accuracy."""
    # --- Every row ---
    print(
        f"\n{'load':>5} {'beta':>5} {'policy':>10} {'acc':>7} {'J/query':>8} {'saving':>7} {'fwd':>6} "
        f"{'skip':>6} {'rate err':>8} | " + " ".join(f"{t:>6}" for t in TIERS)
    )
    for r in rows:
        print(
            f"{r['peak_load']:5g} {r['beta']:5.1f} {r['policy']:>10} {r['accuracy']:7.4f} "
            f"{r['J_per_query']:8.1f} {r['saving_same_accuracy']:7.1%} {r['forwarded_on_arrival']:6.1%} "
            f"{r['skipped_on_escalation']:6.1%} {r['olt_rate_error']:8.1%} | "
            + " ".join(f"{r['final_' + t]:6.1%}" for t in TIERS)
        )
        if r["policy"] == "oracle" and r["beta"] == betas[-1]:
            print()

    # --- Median saving ---
    print("median saving at equal accuracy, across beta:")
    for peak in peaks:
        med = {
            p: np.nanmedian(
                [
                    r["saving_same_accuracy"]
                    for r in rows
                    if r["peak_load"] == peak and r["policy"] == p
                ]
                or [float("nan")]
            )
            for p in policies
            if p != "recserve"
        }
        print(f"  peak load {peak:5g}: " + "  ".join(f"{p} {v:6.1%}" for p, v in med.items()))

    # --- J per query at equal accuracy ---
    print(
        "\nJ/query at equal accuracy, each policy's own frontier (- = accuracy out of its range):"
    )
    print(f"{'load':>5} {'acc':>5} " + " ".join(f"{p:>10}" for p in policies))
    for peak in peaks:
        for a in [f"{t:.2f}" for t in targets if round(t, 2) in (0.70, 0.80)]:
            vals = {f["policy"]: f["J_at_accuracy"][a] for f in frontiers if f["peak_load"] == peak}
            print(
                f"{peak:5g} {a:>5} "
                + " ".join(
                    f"{vals[p]:10.1f}" if vals[p] is not None else f"{'-':>10}" for p in policies
                )
            )


def output_path(s, mode) -> Path:
    """Return --out, or results/adhoc/sim_<tag>_<UTC>.csv with a tag naming every non-default setting."""
    if s.out:
        return s.out
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = (
        s.boundary
        + (f"_onu{s.onu_scale:g}" if s.onu_scale != 1 else "")
        + (f"_olt{s.olt_scale:g}" if s.olt_scale != 1 else "")
        + ("_flat" if s.household_shape == "flat" else "")
        + mode.tag()
        + ("_shared" if s.shared_stats else "")
        + (f"_hh{s.households}x{s.per_day}" if s.households > 1 else "")
        + (f"_sigma{s.load_sigma:g}" if s.load_sigma else "")
        + ("_marginal" if s.accounting == "marginal" else "")
        + ("_window" if s.report == "window" else "")
        + (f"_ra{s.rate_alpha:g}" if s.rate_alpha is not None else "")
    )
    return ADHOC / f"sim_{tag}_{stamp}.csv"


def write_results(
    s, mode, rows, hourly, configs, frontiers, sources, models, factor, accuracy, fixed, mean_tokens
) -> None:
    """Write the CSV (one row per load, beta, policy) and the JSON (plus settings, inputs, frontiers)."""
    out = output_path(s, mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    load_trace, surges = mode.json_fields()
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {k: str(v) for k, v in vars(s).items()},
                "answers": sources,
                "models": models,
                "olt_factor": factor,
                "accuracy": accuracy,
                "fixed_J_per_query": fixed,
                "mean_tokens": mean_tokens,
                "burstgpt": BURSTGPT,
                "load_trace": load_trace,
                "surges": surges,
                "configs": configs,
                "rows": rows,
                "frontiers": frontiers,
                "hourly": hourly,
            },
            f,
            indent=1,
        )
    print(f"wrote {out} and {out.with_suffix('.json').name}")


if __name__ == "__main__":
    raise SystemExit(main())
