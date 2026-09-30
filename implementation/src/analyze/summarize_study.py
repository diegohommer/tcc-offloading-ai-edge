#!/usr/bin/env python3
"""Write the case study's tables from the simulation runs: results/study/SUMMARY.md.

Every number is read at equal accuracy (simulate/frontier.py): each policy's J per query at
the target accuracy, mean over seeds. A dagger marks a load at which the OLT's load exceeded
the measured batch range (64 in service) for over 1% of arrivals: indicative only.
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from energy.three_tier import STUDY

DIR = STUDY
"""results/study/: the runs read, and where SUMMARY.md is written."""

# display name of each policy in the tables (simulate/cascade.py explains each)
NAMES = {
    "recserve": "RecServe",
    "recserve_no_onu": "RecServe w/o ONU",
    "static_day": "static/day",
    "static_hour": "static/hour",
    "broadcast": "broadcast",
    "piggyback": "piggyback",
    "oracle": "oracle",
}
# pairs compared in the full tables: (policy, baseline) -> saving of the first over the second
COMPARE = [
    ("static_day", "recserve"),
    ("static_hour", "recserve"),
    ("broadcast", "recserve"),
    ("broadcast", "static_day"),
    ("broadcast", "static_hour"),
    ("piggyback", "static_hour"),
    ("oracle", "static_hour"),
]
MAIN = ["main_surge1", "main_surge1.5", "main_surge2", "main_surge3", "main_surge5"]
SENSITIVITY = (
    "average",
    "onu0.5",
    "onu0.2",
    "olt1.07",
    "olt1.75",
    "flat",
    "hh1x2000",
    "hh10x200",
    "hh100x20",
    "perhousehold",
)
ORDER = MAIN + ["alltraffic"] + [f"{s}_surge{f}" for s in SENSITIVITY for f in (1, 3)]


def saving(new, ref):
    """Return the energy saved by `new` relative to `ref` (0.25 = 25% less), or None if either is missing."""
    return None if new is None or ref is None else 1 - new / ref


def pct(xs, spread=True):
    """Format the mean of per-seed savings as a percentage.

    Args:
        xs: One saving per seed (None where missing).
        spread: Whether to add (min..max) across seeds.
    """
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    m = st.mean(xs)
    return f"{m:+.1%} ({min(xs):+.0%}..{max(xs):+.0%})" if spread and len(xs) > 1 else f"{m:+.1%}"


def joules(xs):
    """Format the mean over seeds with one decimal: for J per query and seconds."""
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):.1f}" if xs else "-"


def fmt2(xs):
    """Format the mean over seeds with two decimals: for traffic in MB."""
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):.2f}" if xs else "-"


def fmt3(xs):
    """Format the mean over seeds with three decimals: for confidences, which live near 0.9."""
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):.3f}" if xs else "-"


def at(d, peak, acc):
    """Return every policy's J per query at this accuracy and OLT peak load.

    Args:
        d: One run's JSON.
        peak: The OLT peak load.
        acc: The target accuracy, as a string ("0.80").
    """
    return {f["policy"]: f["J_at_accuracy"][acc] for f in d["frontiers"] if f["peak_load"] == peak}


def at_column(d, peak, acc, column):
    """Return any per-run column, read at the same operating point as J per query.

    Every quantity (delivered accuracy, latency, traffic, confidence) is read along the
    same energy frontier points as J per query (simulate/frontier.py), so all columns
    describe the same runs.

    Args:
        d: One run's JSON.
        peak: The OLT peak load.
        acc: The target accuracy, as a string ("0.80").
        column: The result column to read.

    Returns:
        {policy: value, or None when the target is above the policy's range}.
    """
    out = {}
    t = float(acc)
    for policy in {r["policy"] for r in d["rows"] if r["peak_load"] == peak}:
        rows = sorted(
            (r for r in d["rows"] if r["peak_load"] == peak and r["policy"] == policy),
            key=lambda r: (r["accuracy"], r["J_per_query"]),
            reverse=True,
        )
        pts, best = [], float("inf")
        for r in rows:
            if r["J_per_query"] < best:
                pts.append(r)
                best = r["J_per_query"]
        pts.sort(key=lambda r: r["accuracy"])
        xs, ys = [r["accuracy"] for r in pts], [r[column] for r in pts]
        out[policy] = None if t > xs[-1] else ys[0] if t < xs[0] else float(np.interp(t, xs, ys))
    return out


def dagger(ds, peak):
    """Return whether the OLT's load exceeded the measured batch range (64) for over 1% of arrivals."""
    return any(
        c["peak_load"] == peak and c.get("share_arrivals_over_64", 0) > 0.01
        for d in ds
        for c in d["configs"]
    )


def describe(d):
    """Return one line saying what a scenario is: traffic, accounting, ONU and OLT cost, households."""
    a = d["args"]
    surge = float(a["surge_factor"])
    traffic = f"{a['load_trace']} traffic" + (
        " as recorded" if surge == 1 else f", surges and dips x{surge:g}"
    )
    olt = float(a.get("olt_scale", 1))
    return (
        f"{traffic}; {a['accounting']} accounting; ONU {d['fixed_J_per_query']['onu']:.0f} J; "
        + (f"OLT energy x{olt:g}; " if olt != 1 else "")
        + f"{a['households']} household(s) x {a['per_day']}/day"
        + (", flat over the day" if a.get("household_shape") == "flat" else "")
        + f"; question statistics {'shared' if a['shared_stats'] == 'True' else 'per household'}"
    )


def main() -> int:
    """Group the runs by scenario, write SUMMARY.md and print it.

    Returns:
        The process exit code: 0, or 1 when there are no runs.
    """
    # --- Read the runs ---
    runs = collections.defaultdict(list)
    for p in sorted(DIR.glob("study_*_seed*.json")):
        with open(p, encoding="utf-8") as f:
            runs[p.stem[len("study_") :].rsplit("_seed", 1)[0]].append(json.load(f))
    if not runs:
        print(f"no runs in {DIR}", file=sys.stderr)
        return 1
    peaks = sorted({f["peak_load"] for d in next(iter(runs.values())) for f in d["frontiers"]})
    lines = [
        "# Case study — summary",
        "",
        "Generated by `src/analyze/summarize_study.py` from `src/simulate/run_study.sh`. "
        "J per query at the target "
        "accuracy (each policy's frontier), mean over seeds; savings per seed, mean (min..max). Positive = "
        "the first policy uses less energy. † = the OLT's load exceeded the measured batch range (64 in "
        "service) for over 1% of arrivals: indicative only.",
        "",
    ]

    # --- Headline tables ---
    main_rows = [s for s in MAIN if s in runs]
    if main_rows:
        lines += [
            f"## Headline — {describe(runs[main_rows[0]][0]).split('; ', 1)[1]}, 0.80 accuracy",
            "",
            "### Saving over RecServe: static/day / static/hour / broadcast",
            "",
            "| traffic | " + " | ".join(f"load {p:g}" for p in peaks) + " |",
            "|---|" + "---|" * len(peaks),
        ]
        for s in main_rows:
            ds, f = runs[s], float(runs[s][0]["args"]["surge_factor"])
            cells = []
            for p in peaks:
                per_seed = [at(d, p, "0.80") for d in ds]
                cells.append(
                    f"{pct([saving(j.get('static_day'), j['recserve']) for j in per_seed], False)} / "
                    f"{pct([saving(j['static_hour'], j['recserve']) for j in per_seed], False)} / "
                    f"{pct([saving(j['broadcast'], j['recserve']) for j in per_seed], False)}"
                    + (" †" if dagger(ds, p) else "")
                )
            lines.append(
                f"| {'predictable' if f == 1 else f'surprises x{f:g}'} | "
                + " | ".join(cells)
                + " |"
            )
        lines += [
            "",
            "### Saving over static/hour (the timetable): "
            "broadcast (oracle ceiling) · piggyback on answers only",
            "",
            "| traffic | " + " | ".join(f"load {p:g}" for p in peaks) + " |",
            "|---|" + "---|" * len(peaks),
        ]
        for s in main_rows:
            ds, f = runs[s], float(runs[s][0]["args"]["surge_factor"])
            cells = []
            for p in peaks:
                per_seed = [at(d, p, "0.80") for d in ds]
                cells.append(
                    f"{pct([saving(j['broadcast'], j['static_hour']) for j in per_seed], False)} "
                    f"({pct([saving(j['oracle'], j['static_hour']) for j in per_seed], False)}) · "
                    f"{pct([saving(j['piggyback'], j['static_hour']) for j in per_seed], False)}"
                    + (" †" if dagger(ds, p) else "")
                )
            lines.append(
                f"| {'predictable' if f == 1 else f'surprises x{f:g}'} | "
                + " | ".join(cells)
                + " |"
            )
        lines += [
            "",
            "### At 0.80 accuracy: mean latency (s) · PON traffic (MB per 1,000 queries), "
            "RecServe / static/hour / broadcast",
            "",
            "| traffic | " + " | ".join(f"load {p:g}" for p in peaks) + " |",
            "|---|" + "---|" * len(peaks),
        ]
        for s in main_rows:
            ds, f = runs[s], float(runs[s][0]["args"]["surge_factor"])
            cells = []
            for p in peaks:
                lat = [at_column(d, p, "0.80", "latency_s_mean") for d in ds]
                pon = [at_column(d, p, "0.80", "pon_MB_per_1k_queries") for d in ds]
                cells.append(
                    " / ".join(
                        f"{joules([x.get(q) for x in lat])} s · {fmt2([x.get(q) for x in pon])}"
                        for q in ("recserve", "static_hour", "broadcast")
                    )
                )
            lines.append(
                f"| {'predictable' if f == 1 else f'surprises x{f:g}'} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")

    # --- One full table per scenario ---
    for s in [s for s in ORDER if s in runs] + sorted(set(runs) - set(ORDER)):
        ds = runs[s]
        pols = [p for p in NAMES if p in ds[0]["args"]["policies"].split(",")]
        lines += [
            f"## `{s}` — {describe(ds[0])} ({len(ds)} seeds)",
            "",
            "| load | acc | "
            + " | ".join(NAMES[p] for p in pols)
            + " | "
            + " | ".join(f"{NAMES[a]} vs {NAMES[b]}" for a, b in COMPARE)
            + " |",
            "|---|---|" + "---|" * (len(pols) + len(COMPARE)),
        ]
        for p in peaks:
            for acc in ("0.70", "0.80"):
                per_seed = [at(d, p, acc) for d in ds]
                lines.append(
                    f"| {p:g}{' †' if dagger(ds, p) else ''} | {acc} | "
                    + " | ".join(joules([j.get(q) for j in per_seed]) for q in pols)
                    + " | "
                    + " | ".join(
                        pct([saving(j.get(a), j.get(b)) for j in per_seed]) for a, b in COMPARE
                    )
                    + " |"
                )
        lines += [
            "",
            "At 0.80 accuracy, per policy, read at the same frontier point as the energy: "
            "accuracy delivered (above 0.80 when even the policy's least accurate point is) · "
            "mean latency, s · PON traffic, MB per 1,000 queries · RecServe's communication "
            "burden, MB per 1,000 queries · confidence of the delivered answer.",
            "",
            "| load | " + " | ".join(NAMES[q] for q in pols) + " |",
            "|---|" + "---|" * len(pols),
        ]
        for p in peaks:
            cols = {
                c: [at_column(d, p, "0.80", c) for d in ds]
                for c in (
                    "accuracy",
                    "latency_s_mean",
                    "pon_MB_per_1k_queries",
                    "comm_MB_per_1k_queries",
                    "mean_confidence",
                )
            }
            cells = [
                " · ".join(
                    [
                        fmt3([x.get(q) for x in cols["accuracy"]]),
                        f"{joules([x.get(q) for x in cols['latency_s_mean']])} s",
                        fmt2([x.get(q) for x in cols["pon_MB_per_1k_queries"]]),
                        fmt2([x.get(q) for x in cols["comm_MB_per_1k_queries"]]),
                        fmt3([x.get(q) for x in cols["mean_confidence"]]),
                    ]
                )
                for q in pols
            ]
            lines.append(f"| {p:g}{' †' if dagger(ds, p) else ''} | " + " | ".join(cells) + " |")
        lines.append("")

    # --- Write ---
    out = DIR / "SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
