#!/usr/bin/env python3
"""Simulate households' queries through the phone -> ONU -> OLT cascade, under each policy.

Every answer is replayed from the recorded collection, so no model runs here. Results come
out per OLT size, RecServe beta and policy, and policies are compared at equal
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
from cascade import calibrate, POLICIES, run, RunSetup
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
from households import Households
from olt_energy import Energy

CONFIG = ROOT / "config" / "simulation.yaml"
"""The default settings file."""

ADHOC = ROOT / "results" / "adhoc"
"""Where a one-off run writes when --out is not given."""


def main() -> int:
    """Run every policy at every OLT size and beta, print the tables and write the results.

    The steps, each a function below: read_settings, answers_by_question,
    check_combinations, build_prices; then, per OLT size, prepare_population, olt_summary
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
    if not check_combinations(policies):
        return 1
    published, factor, curve, prices = build_prices(s)
    betas = [float(b) for b in s.betas.split(",")]
    targets = [float(t) for t in s.acc_targets.split(",")]
    sizes = [int(n) for n in s.subscribers.split(",")]
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
    print_header(s, sources, answers, factor, accuracy, fixed)

    # --- Every population, beta and policy ---
    rows, hourly, configs, frontiers = [], [], [], []
    for subs in sizes:
        setup, homes = prepare_population(subs, s, answers, curve, prices)
        configs.append(olt_summary(subs, homes, setup))
        print(f"{homes.describe()}; {len(setup.stream):,} queries simulated")
        block, block_hourly = run_policies(setup, subs, betas, policies)
        iso_accuracy(block)  # adds each row's saving against RecServe at equal accuracy
        rows += block
        hourly += block_hourly
        for p in policies:
            frontiers.append(
                {
                    "subscribers": subs,
                    "policy": p,
                    "J_at_accuracy": frontier([r for r in block if r["policy"] == p], targets),
                }
            )

    # --- Output ---
    print_tables(rows, betas, sizes, policies, frontiers, targets)
    write_results(
        s,
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
    add("--subscribers", help="comma-separated household counts on the OLT")
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
    add("--users-per-home", type=float, help="weekly-active LLM users in one household")
    add("--per-user-day", type=float, help="messages one active user sends a day")
    add("--test-days", type=int, help="days simulated")
    add("--calibration-days", type=int, help="days before them the timetable is observed over")
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


def check_combinations(policies) -> bool:
    """Check that the settings can go together.

    Args:
        policies: The policies to run.

    Returns:
        True, or False (after printing why).
    """
    if "recserve" not in policies or not set(policies) <= set(POLICIES):
        print(f"--policies must include recserve and come from {POLICIES}", file=sys.stderr)
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
# One population
# ==========================================
def prepare_population(subscribers, settings, answers, curve, prices):
    """Build the stream one population runs on, and the tables the static policies ship with.

    The calibration days only exist so the static tables have a month to be observed over;
    the test days follow them, on a different seed so they are not the same month twice.

    Args:
        subscribers: Households on the OLT.
        settings: The parsed settings.
        answers: answers[question][tier].
        curve: The OLT's measured curve.
        prices: True energy rates.

    Returns:
        (the RunSetup every policy at this size shares, the Households that made it).
    """
    homes = Households(subscribers, settings.users_per_home, settings.per_user_day)
    questions = sorted(answers)

    def schedule_key(hour):
        """Return static_hour's cell for a time: (0 weekday / 1 weekend, hour of day)."""
        return (0 if homes.weekday(int(hour // 24)) else 1, int(hour) % 24)

    def observed_tables(population):
        """Return the static tables observed over the calibration days of this population."""
        watched = RunSetup(
            answers=answers,
            stream=population.stream(questions, settings.calibration_days, 0, settings.seed),
            curve=curve,
            prices=prices,
            static_rates={},
            settings=settings,
            schedule_key=schedule_key,
        )
        return calibrate(watched, middle_beta)

    # --- The tables: observed once, at the middle of the beta sweep ---
    betas = sorted(float(beta) for beta in settings.betas.split(","))
    middle_beta = betas[len(betas) // 2]
    static_rates = observed_tables(homes)
    # the stale pair ship static_day as observed on a population stale_factor off the real one
    for name, scale in (
        ("stale_low", 1 / settings.stale_factor),
        ("stale_high", settings.stale_factor),
    ):
        if name in settings.policies.split(","):
            other = Households(
                max(round(subscribers * scale), 1),
                settings.users_per_home,
                settings.per_user_day,
            )
            static_rates[name] = observed_tables(other).get("static_day")

    setup = RunSetup(
        answers=answers,
        stream=homes.stream(
            questions, settings.test_days, settings.calibration_days, settings.seed + 1
        ),
        curve=curve,
        prices=prices,
        static_rates=static_rates,
        settings=settings,
        schedule_key=schedule_key,
    )
    return setup, homes


def olt_summary(subs, homes, setup) -> dict:
    """Summarize the population and what it sends, for the printout and the JSON.

    What the OLT then costs is no longer something to summarize here: it depends on the
    policy, so each run reports its own mean batch.

    Args:
        subs: Households on the OLT.
        homes: The Households that made the stream.
        setup: This size's RunSetup.

    Returns:
        The summary, one entry of the JSON's "configs".
    """
    return {
        "subscribers": subs,
        "users_per_home": homes.users,
        "messages_per_user_day": homes.per_user_day,
        "messages_per_day": homes.messages_per_day,
        "bursts_per_day": homes.sessions_per_day,
        "queries_simulated": len(setup.stream),
        "weekend_days_mod7": sorted(homes.weekend),
    }


def run_policies(setup: RunSetup, subs, betas, policies):
    """Run every beta and policy at this OLT size.

    Args:
        setup: This size's RunSetup.
        subs: Households on the OLT.
        betas: RecServe's beta values.
        policies: The policies to run.

    Returns:
        (result rows, their per-hour detail).
    """
    block, block_hourly = [], []
    for beta in betas:
        for policy in policies:
            metrics, hours, _ = run(setup, beta, policy)
            block.append({"subscribers": subs, "beta": beta, "policy": policy, **metrics})
            if not setup.settings.no_hourly:
                block_hourly.append(
                    {"subscribers": subs, "beta": beta, "policy": policy, "hours": hours}
                )
    return block, block_hourly


# ==========================================
# Printout and files
# ==========================================
def print_header(s, sources, answers, factor, accuracy, fixed) -> None:
    """Print the run's settings and inputs, before any policy runs."""
    reports = f"its {s.report_window:g}-min mean" if s.report == "window" else "per query"
    print(f"answers: {', '.join(sources)}")
    print(
        f"n={len(answers)} questions; {s.test_days} days after {s.calibration_days} "
        f"calibration days; confidence exp({s.confidence}); "
        f"OLT boundary {s.boundary} (x{factor:.2f}), {s.accounting} accounting; OLT reports "
        f"{reports}; delta={s.delta}, window={s.window}"
    )
    print("standalone accuracy: " + "  ".join(f"{t} {accuracy[t]:.3f}" for t in TIERS))
    print(f"per query: user {fixed['user']:.1f} J, ONU {fixed['onu']:.1f} J (fixed)\n")


def print_tables(rows, betas, sizes, policies, frontiers, targets) -> None:
    """Print every result row, the median saving per OLT size, and J per query at equal accuracy."""
    # --- Every row ---
    print(
        f"\n{'homes':>7} {'beta':>5} {'policy':>10} {'acc':>7} {'J/query':>8} {'saving':>7} {'fwd':>6} "
        f"{'skip':>6} {'rate err':>8} {'batch':>6} {'queued':>6} | "
        + " ".join(f"{t:>6}" for t in TIERS)
    )
    for r in rows:
        print(
            f"{r['subscribers']:7d} {r['beta']:5.1f} {r['policy']:>10} {r['accuracy']:7.4f} "
            f"{r['J_per_query']:8.1f} {r['saving_same_accuracy']:7.1%} {r['forwarded_on_arrival']:6.1%} "
            f"{r['skipped_on_escalation']:6.1%} {r['olt_rate_error']:8.1%} "
            f"{r['mean_olt_batch']:6.1f} {r['olt_queued']:6.1%} | "
            + " ".join(f"{r['final_' + t]:6.1%}" for t in TIERS)
        )
        if r["policy"] == "oracle" and r["beta"] == betas[-1]:
            print()

    # --- Median saving ---
    print("median saving at equal accuracy, across beta:")
    for subs in sizes:
        med = {
            p: np.nanmedian(
                [
                    r["saving_same_accuracy"]
                    for r in rows
                    if r["subscribers"] == subs and r["policy"] == p
                ]
                or [float("nan")]
            )
            for p in policies
            if p != "recserve"
        }
        print(f"  {subs:7,} households: " + "  ".join(f"{p} {v:6.1%}" for p, v in med.items()))

    # --- J per query at equal accuracy ---
    print(
        "\nJ/query at equal accuracy, each policy's own frontier (- = accuracy out of its range):"
    )
    print(f"{'homes':>7} {'acc':>5} " + " ".join(f"{p:>10}" for p in policies))
    for subs in sizes:
        for a in [f"{t:.2f}" for t in targets if round(t, 2) in (0.70, 0.80)]:
            vals = {
                f["policy"]: f["J_at_accuracy"][a] for f in frontiers if f["subscribers"] == subs
            }
            print(
                f"{subs:7d} {a:>5} "
                + " ".join(
                    f"{vals[p]:10.1f}" if vals[p] is not None else f"{'-':>10}" for p in policies
                )
            )


def output_path(s) -> Path:
    """Return --out, or results/adhoc/sim_<tag>_<UTC>.csv with a tag naming every non-default setting."""
    if s.out:
        return s.out
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tag = (
        s.boundary
        + (f"_onu{s.onu_scale:g}" if s.onu_scale != 1 else "")
        + (f"_olt{s.olt_scale:g}" if s.olt_scale != 1 else "")
        + ("_shared" if s.shared_stats else "")
        + (f"_u{s.users_per_home:g}x{s.per_user_day:g}" if s.users_per_home != 1 else "")
        + ("_marginal" if s.accounting == "marginal" else "")
        + ("_window" if s.report == "window" else "")
        + (f"_ra{s.rate_alpha:g}" if s.rate_alpha is not None else "")
    )
    return ADHOC / f"sim_{tag}_{stamp}.csv"


def write_results(
    s, rows, hourly, configs, frontiers, sources, models, factor, accuracy, fixed, mean_tokens
) -> None:
    """Write the CSV (one row per load, beta, policy) and the JSON (plus settings, inputs, frontiers)."""
    out = output_path(s)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
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
