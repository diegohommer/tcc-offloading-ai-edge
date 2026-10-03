#!/usr/bin/env python3
"""Write the case study's tables from the simulation runs: results/study/SUMMARY.md.

Every number is read at equal accuracy (simulate/frontier.py): each policy's J per query at
the target accuracy, mean over seeds. A dagger marks a population at which more than 1% of
the OLT's arrivals found all its slots busy and waited.
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from energy.three_tier import STUDY

NAMES = {
    "recserve": "RecServe",
    "recserve_no_onu": "RecServe w/o ONU",
    "static_day": "static/day",
    "static_hour": "static/hour",
    "broadcast": "broadcast",
    "piggyback": "piggyback",
    "oracle": "oracle",
}
"""Display name of each policy (simulate/cascade.py explains each)."""

COMPARE = [
    ("static_day", "recserve"),
    ("static_hour", "recserve"),
    ("broadcast", "recserve"),
    ("broadcast", "static_day"),
    ("broadcast", "static_hour"),
    ("piggyback", "static_hour"),
    ("oracle", "broadcast"),
]
"""(policy, baseline) pairs in the full tables: the saving of the first over the second."""

ORDER = ["main", "average", "onu0.5", "onu0.2", "olt1.07", "olt1.75", "users2", "perhousehold"]
"""Scenarios in run_study.sh's order."""

COLUMNS = ("accuracy", "latency_s_mean", "pon_MB_per_1k_queries", "comm_MB_per_1k_queries")
"""Per-run columns shown beside the energy, read at the same frontier point."""


# ==========================================
# Reading the runs
# ==========================================
def saving(new, reference):
    """Return the energy `new` saves over `reference` (0.25 = 25% less), or None if either is missing."""
    return None if new is None or reference is None else 1 - new / reference


def at(run, subscribers, accuracy):
    """Return every policy's J per query at this accuracy and population.

    Args:
        run: One run's JSON.
        subscribers: Households on the OLT.
        accuracy: The target accuracy, as a string ("0.80").
    """
    return {
        frontier["policy"]: frontier["J_at_accuracy"][accuracy]
        for frontier in run["frontiers"]
        if frontier["subscribers"] == subscribers
    }


def at_column(run, subscribers, accuracy, column):
    """Return any per-run column, read at the same frontier point as J per query.

    Args:
        run: One run's JSON.
        subscribers: Households on the OLT.
        accuracy: The target accuracy, as a string ("0.80").
        column: The result column to read.

    Returns:
        {policy: value, or None when the target is above the policy's range}.
    """
    target = float(accuracy)
    rows = [row for row in run["rows"] if row["subscribers"] == subscribers]
    values = {}
    for policy in {row["policy"] for row in rows}:
        points, cheapest = [], float("inf")
        for row in sorted(
            (row for row in rows if row["policy"] == policy),
            key=lambda row: (row["accuracy"], row["J_per_query"]),
            reverse=True,
        ):
            if row["J_per_query"] < cheapest:
                points.append(row)
                cheapest = row["J_per_query"]
        points.sort(key=lambda row: row["accuracy"])
        accuracies = [row["accuracy"] for row in points]
        readings = [row[column] for row in points]
        values[policy] = (
            None
            if target > accuracies[-1]
            else (
                readings[0]
                if target < accuracies[0]
                else float(np.interp(target, accuracies, readings))
            )
        )
    return values


def queued(runs, subscribers):
    """Return whether over 1% of the OLT's arrivals waited for a slot, in any run at this size."""
    return any(
        row["subscribers"] == subscribers and row["olt_queued"] > 0.01
        for run in runs
        for row in run["rows"]
    )


def describe(run):
    """Return one line saying what a scenario is."""
    args = run["args"]
    olt_scale = float(args["olt_scale"])
    return (
        f"{args['accounting']} accounting; ONU {run['fixed_J_per_query']['onu']:.0f} J; "
        + (f"OLT energy x{olt_scale:g}; " if olt_scale != 1 else "")
        + f"{args['users_per_home']} active user(s) x {args['per_user_day']} messages/day per household; "
        + f"question statistics {'shared' if args['shared_stats'] == 'True' else 'per household'}"
    )


# ==========================================
# Formatting
# ==========================================
def present(values):
    """Return the values that are not None."""
    return [value for value in values if value is not None]


def percent(savings, spread=True):
    """Format the mean of per-seed savings, with (min..max) across seeds when asked."""
    savings = present(savings)
    if not savings:
        return "-"
    mean = statistics.mean(savings)
    if spread and len(savings) > 1:
        return f"{mean:+.1%} ({min(savings):+.0%}..{max(savings):+.0%})"
    return f"{mean:+.1%}"


def mean_of(values, decimals):
    """Format the mean over seeds with this many decimals."""
    values = present(values)
    return f"{statistics.mean(values):.{decimals}f}" if values else "-"


def header(columns):
    """Return a markdown table's header and rule."""
    return ["| " + " | ".join(columns) + " |", "|---" * len(columns) + "|"]


# ==========================================
# Tables
# ==========================================
def headline(runs, populations):
    """Return the headline tables: the main scenario at 0.80 accuracy, then every scenario's saving."""
    main_runs = runs["main"]
    sizes = [f"{size:,} homes" + (" †" if queued(main_runs, size) else "") for size in populations]
    lines = [f"## Headline — {describe(main_runs[0])}; 0.80 accuracy", ""]

    lines += ["### Saving over RecServe: static/day / static/hour / broadcast", ""]
    lines += header(["", *sizes])
    cells = []
    for size in populations:
        per_seed = [at(run, size, "0.80") for run in main_runs]
        cells.append(
            " / ".join(
                percent(
                    [saving(energy.get(policy), energy["recserve"]) for energy in per_seed], False
                )
                for policy in ("static_day", "static_hour", "broadcast")
            )
        )
    lines += ["| saving | " + " | ".join(cells) + " |", ""]

    lines += ["### Saving over static/hour: broadcast (oracle) · piggyback", ""]
    lines += header(["", *sizes])
    cells = []
    for size in populations:
        per_seed = [at(run, size, "0.80") for run in main_runs]
        broadcast, oracle, piggyback = (
            percent(
                [saving(energy.get(policy), energy.get("static_hour")) for energy in per_seed],
                False,
            )
            for policy in ("broadcast", "oracle", "piggyback")
        )
        cells.append(f"{broadcast} ({oracle}) · {piggyback}")
    lines += ["| saving | " + " | ".join(cells) + " |", ""]

    lines += ["### Every scenario: broadcast's saving over RecServe · over static/hour", ""]
    lines += header(["scenario", *(f"{size:,} homes" for size in populations)])
    for scenario in [name for name in ORDER if name in runs]:
        cells = []
        for size in populations:
            per_seed = [at(run, size, "0.80") for run in runs[scenario]]
            over_recserve = percent(
                [saving(energy.get("broadcast"), energy["recserve"]) for energy in per_seed], False
            )
            over_timetable = percent(
                [saving(energy.get("broadcast"), energy.get("static_hour")) for energy in per_seed],
                False,
            )
            cells.append(
                f"{over_recserve} · {over_timetable}"
                + (" †" if queued(runs[scenario], size) else "")
            )
        lines.append(f"| `{scenario}` | " + " | ".join(cells) + " |")
    return lines + [""]


def scenario_tables(scenario, runs, populations):
    """Return one scenario's full tables: energy and savings, then the other columns."""
    policies = [policy for policy in NAMES if policy in runs[0]["args"]["policies"].split(",")]
    pairs = [(new, old) for new, old in COMPARE if new in policies and old in policies]
    lines = [f"## `{scenario}` — {describe(runs[0])} ({len(runs)} seeds)", ""]
    lines += header(
        ["homes", "acc"]
        + [NAMES[policy] for policy in policies]
        + [f"{NAMES[new]} vs {NAMES[old]}" for new, old in pairs]
    )
    for size in populations:
        for accuracy in ("0.70", "0.80"):
            per_seed = [at(run, size, accuracy) for run in runs]
            lines.append(
                f"| {size:,}{' †' if queued(runs, size) else ''} | {accuracy} | "
                + " | ".join(
                    mean_of([energy.get(policy) for energy in per_seed], 1) for policy in policies
                )
                + " | "
                + " | ".join(
                    percent([saving(energy.get(new), energy.get(old)) for energy in per_seed])
                    for new, old in pairs
                )
                + " |"
            )
    lines += [
        "",
        "At 0.80 accuracy, read at the same frontier point as the energy: accuracy delivered "
        "· mean latency, s · PON traffic, MB per 1,000 queries · RecServe's communication "
        "burden, MB per 1,000 queries.",
        "",
    ]
    lines += header(["homes", *(NAMES[policy] for policy in policies)])
    for size in populations:
        readings = {
            column: [at_column(run, size, "0.80", column) for run in runs] for column in COLUMNS
        }
        cells = [
            " · ".join(
                [
                    mean_of([reading.get(policy) for reading in readings["accuracy"]], 3),
                    mean_of([reading.get(policy) for reading in readings["latency_s_mean"]], 1)
                    + " s",
                    mean_of(
                        [reading.get(policy) for reading in readings["pon_MB_per_1k_queries"]], 2
                    ),
                    mean_of(
                        [reading.get(policy) for reading in readings["comm_MB_per_1k_queries"]], 2
                    ),
                ]
            )
            for policy in policies
        ]
        lines.append(f"| {size:,} | " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> int:
    """Group the runs by scenario, write SUMMARY.md and print it.

    Returns:
        The process exit code: 0, or 1 when there are no runs.
    """
    runs = collections.defaultdict(list)
    for path in sorted(STUDY.glob("study_*_seed*.json")):
        with open(path, encoding="utf-8") as file:
            runs[path.stem[len("study_") :].rsplit("_seed", 1)[0]].append(json.load(file))
    if not runs:
        print(f"no runs in {STUDY}", file=sys.stderr)
        return 1
    populations = sorted(
        {
            frontier["subscribers"]
            for run in next(iter(runs.values()))
            for frontier in run["frontiers"]
        }
    )

    lines = [
        "# Case study — summary",
        "",
        "Generated by `src/analyze/summarize_study.py` from `src/simulate/run_study.sh`. J per "
        "query at the target accuracy (each policy's frontier), mean over seeds; savings per "
        "seed, mean (min..max). Positive = the first policy uses less energy. † = over 1% of "
        "the OLT's arrivals waited for one of its 64 slots.",
        "",
    ]
    if "main" in runs:
        lines += headline(runs, populations)
    for scenario in [name for name in ORDER if name in runs] + sorted(set(runs) - set(ORDER)):
        lines += scenario_tables(scenario, runs[scenario], populations)

    out = STUDY / "SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
