#!/usr/bin/env python3
"""Plot the case study, from the traffic to the results: results/study/figures/.

Every figure follows one standard, so any of them can go into the thesis as it is:
- drawn at the thesis's text width (15 cm), so LaTeX includes it at width=\\textwidth with
  no scaling, and its 9 pt text prints at 9 pt beside the 12 pt body;
- set in Times (Nimbus Roman, STIX for symbols), the body's typeface;
- no title inside the figure: the caption carries it, and panels are marked (a), (b);
- written as a vector PDF for the thesis and a 300 dpi PNG for slides and previews;
- its numbers written beside it in data/<name>.csv, one row per plotted point, in the
  same columns for every figure (DATA_COLUMNS);
- every result is a mean over the seeds 7, 8 and 9, read at equal accuracy
  (simulate/frontier.py) as SUMMARY.md reports it. Readings at 0.80 accuracy carry the
  seeds' range as error bars; the per-β curves, whose range is a few joules, carry it in
  the data only.

README.md beside the figures lists each one with a draft caption and its data file.
"""

# pylint: disable=wrong-import-position,ungrouped-imports

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulate"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_continuous_batching import compare_run  # noqa: E402
from energy.three_tier import olt_reference, RESULTS, STUDY  # noqa: E402
from households import Households  # noqa: E402
from summarize_study import at_column  # noqa: E402

FIGURES = STUDY / "figures"
"""Where the figures are written."""

DATA = FIGURES / "data"
"""Where each figure's numbers are written."""

SEEDS = (7, 8, 9)
SIZES = (1000, 2000, 5000, 10000, 20000)
SIZE = 10000
"""The household count the per-β figures show."""

TARGET = "0.80"
"""The accuracy every policy is read at."""

# ==========================================
# The standard
# ==========================================
TEXT_WIDTH = 426.79 / 72.27
"""The thesis's \\textwidth, in inches."""

HEIGHT = 2.7
"""Height of a figure, in inches; TRAFFIC_HEIGHT for the three stacked weeks."""

TRAFFIC_HEIGHT = 3.9

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,  # TrueType in the PDF: text stays text, and fonts embed
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

# Categorical slots of the validated default palette, each entity always in its own colour
# and marker. Some slots sit below 3:1 against white, so every series also has a legend
# entry and a marker of its own, and reads in greyscale print too.
SERIES = {
    "recserve": ("RecServe", "#2a78d6", "o"),
    "static_hour_self": ("Timetable", "#1baf7a", "^"),
    "broadcast": ("Broadcast", "#eda100", "D"),
}
SCENARIOS = {
    "main": ("No drift", "#2a78d6", "o"),
    "burst_week": ("Mild drift", "#eb6834", "s"),
    "burst_all": ("Strong drift", "#1baf7a", "^"),
}
TIERS = {"user": ("Phone", "#e87ba4"), "onu": ("ONU", "#008300"), "olt": ("OLT", "#4a3aa7")}
ALONE = {"user_alone": "Phone alone", "onu_alone": "ONU alone", "olt_alone": "OLT alone"}
"""The single-tier runs (results/study/alone_seed*.json), each tier answering every query."""

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BETA = r"$\beta$ (escalation quantile)"
ENERGY = "Energy per query (J)"
ACCURACY = "Accuracy"

DATA_COLUMNS = (
    "panel",
    "scenario",
    "households",
    "series",
    "x_name",
    "x",
    "y_name",
    "y",
    "y_low",
    "y_high",
)
"""Every figure's data file has these columns; a cell that does not apply is left empty.
y is a mean over the seeds and y_low, y_high their range, when the point has seeds."""

CAPTIONS = {
    "traffic_week": (
        "Messages sent per hour over one test week by 10,000 households, in each traffic "
        "scenario. With no drift every day follows BurstGPT's average day. Mild drift adds "
        "BurstGPT's variation within each week, surges of about 25% lasting about 3 hours; "
        "strong drift adds all of its variation around the average day, swings of about "
        "45 to 60% lasting about 12 hours, whole weeks included."
    ),
    "beta_knob": (
        "Accuracy (a) and energy per query (b) against RecServe's escalation quantile "
        "$\\beta$, with no drift and 10,000 households. Dotted lines: each tier answering "
        "every query alone."
    ),
    "equal_accuracy": (
        "Energy per query against accuracy, one marker per $\\beta$ from 0.1 to 0.9, with "
        "no drift and 10,000 households. Circles: each policy read at 0.80 accuracy. "
        "Crosses: each tier answering every query alone."
    ),
    "tiers": (
        "Share of queries answered at each tier against $\\beta$, under RecServe (a) and "
        "the broadcast (b), with no drift and 10,000 households."
    ),
    "batching_validation": (
        "(a) Energy per generated token on one L4 GPU, from the static batch sweep the "
        "simulator uses and from continuous batching. (b) Energy the simulator predicts "
        "against what the GPU measured, over 15 Poisson runs."
    ),
    "saving_over_recserve": (
        "Energy saved over RecServe at 0.80 accuracy, with no drift, by number of "
        "households. Bars: range over three seeds."
    ),
    "broadcast_over_timetable": (
        "Energy the broadcast saves over the timetable at 0.80 accuracy, by number of "
        "households, with no, mild and strong drift. Bars: range over three seeds."
    ),
    "drift": (
        "Energy per query at 0.80 accuracy with no, mild and strong drift, 10,000 "
        "households. "
        "Bars: range over three seeds."
    ),
    "bandwidth": (
        "(a) RecServe's communication burden, the bytes carried between tiers, against "
        "accuracy. (b) Change in energy and in communication burden against RecServe at "
        "0.80 accuracy. No drift, 10,000 households; bars: range over three seeds."
    ),
}
"""Draft caption of each figure, written to README.md beside them."""


# ==========================================
# Reading the runs
# ==========================================
def load_runs(scenario: str) -> list[dict]:
    """Return one scenario's run JSONs, one per seed."""
    return [
        json.loads((STUDY / f"study_{scenario}_seed{seed}.json").read_text(encoding="utf-8"))
        for seed in SEEDS
    ]


def spread(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, lowest, highest) of the seeds' values."""
    return statistics.mean(values), min(values), max(values)


def per_beta(runs: list[dict], policy: str, size: int, column: str) -> tuple[list, list]:
    """Return (betas, the column's (mean, low, high) over the seeds at each beta)."""
    by_beta: dict[float, list[float]] = {}
    for run in runs:
        for row in run["rows"]:
            if row["policy"] == policy and row["subscribers"] == size:
                by_beta.setdefault(row["beta"], []).append(row[column])
    betas = sorted(by_beta)
    return betas, [spread(by_beta[beta]) for beta in betas]


def at_accuracy(runs: list[dict], policy: str, size: int) -> tuple[float, float, float]:
    """Return a policy's J per query at the target accuracy, (mean, low, high) over seeds."""
    return spread(
        [
            entry["J_at_accuracy"][TARGET]
            for run in runs
            for entry in run["frontiers"]
            if entry["subscribers"] == size and entry["policy"] == policy
        ]
    )


def savings(runs: list[dict], policy: str, reference: str) -> list:
    """Return (mean, low, high) of a policy's saving over a reference, per size, across seeds."""
    out = []
    for size in SIZES:
        per_seed = []
        for run in runs:
            energy = {
                entry["policy"]: entry["J_at_accuracy"][TARGET]
                for entry in run["frontiers"]
                if entry["subscribers"] == size
            }
            if energy.get(policy) and energy.get(reference):
                per_seed.append(100 * (1 - energy[policy] / energy[reference]))
        out.append(spread(per_seed))
    return out


def tier_alone(policy: str, column: str) -> tuple[float, float, float]:
    """Return one tier's column when it answers every query, (mean, low, high) over seeds."""
    return spread(
        [
            row[column]
            for seed in SEEDS
            for row in json.loads((STUDY / f"alone_seed{seed}.json").read_text(encoding="utf-8"))[
                "rows"
            ]
            if row["policy"] == policy and row["subscribers"] == SIZE
        ]
    )


# ==========================================
# Drawing helpers
# ==========================================
def new_figure(columns: int = 1, height: float = HEIGHT, rows: int = 1, **shared):
    """Return a figure at the text width and its panels, as a flat array."""
    figure, panels = plt.subplots(rows, columns, figsize=(TEXT_WIDTH, height), **shared)
    return figure, np.atleast_1d(panels).ravel()


def style(axes, xlabel: str, ylabel: str, panel: str = "") -> None:
    """Give a chart recessive axes and grid, its labels, and its panel letter if any."""
    if panel:
        axes.set_title(panel, loc="left", color=INK, pad=6)
    axes.set_xlabel(xlabel, color=INK)
    axes.set_ylabel(ylabel, color=INK)
    axes.grid(color=GRID, linewidth=0.5)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    axes.tick_params(colors=INK)


def line(axes, xs, ys, name: str, color: str, marker: str, **extra) -> None:
    """Draw one series: a line with white-ringed markers."""
    axes.plot(
        xs,
        ys,
        color=color,
        marker=marker,
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=name,
        **extra,
    )


def ranged_line(axes, xs, points, name: str, color: str, marker: str) -> None:
    """Draw one series of (mean, low, high) points, the seeds' range as error bars."""
    axes.errorbar(
        xs,
        [point[0] for point in points],
        yerr=[[point[0] - point[1] for point in points], [point[2] - point[0] for point in points]],
        color=color,
        marker=marker,
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        capsize=2.5,
        elinewidth=0.8,
        label=name,
    )


def bar_errors(points) -> dict:
    """Return the error-bar arguments of bars drawn at the means of (mean, low, high) points."""
    return {
        "yerr": [
            [point[0] - point[1] for point in points],
            [point[2] - point[0] for point in points],
        ],
        "error_kw": {"ecolor": INK, "capsize": 2.5, "elinewidth": 0.8},
    }


def reference_line(axes, value: float, text: str, x: float, align: str = "left") -> None:
    """Draw a dotted horizontal reference, named just above it at x."""
    axes.axhline(value, color=MUTED, linewidth=0.7, linestyle=":")
    axes.annotate(
        text,
        (x, value),
        xytext=(0, 2),
        textcoords="offset points",
        ha=align,
        fontsize=7.5,
        color=MUTED,
    )


def legend(axes, where: str = "upper left", **extra) -> None:
    """Add the legend, kept for every chart of two or more series."""
    axes.legend(loc=where, frameon=False, labelcolor=INK, **extra)


def households_axis(axes) -> None:
    """Put the household counts on a log axis."""
    axes.set_xscale("log")
    axes.set_xticks(SIZES, [f"{size:,}" for size in SIZES])
    axes.minorticks_off()
    axes.set_xlim(800, 25000)


def point(**cells) -> dict:
    """Return one data row, every column present."""
    return {column: cells.get(column, "") for column in DATA_COLUMNS}


def ranged(name: str, points, **cells) -> list[dict]:
    """Return data rows for (x, (mean, low, high)) pairs of one series."""
    return [
        point(series=name, x=x, y=mean, y_low=low, y_high=high, **cells)
        for x, (mean, low, high) in points
    ]


def save(figure, name: str, rows: list[dict], top: float = 1.0) -> Path:
    """Write a figure as PDF and PNG, and its numbers as CSV; close it.

    Args:
        figure: The figure.
        name: Its file name, without suffix.
        rows: Its data rows (point()).
        top: The fraction of the height the panels may use, leaving the rest to a legend.
    """
    figure.tight_layout(pad=0.4, rect=(0, 0, 1, top))
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    with open(DATA / f"{name}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DATA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (f"{value:.6g}" if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )
    return FIGURES / f"{name}.pdf"


# ==========================================
# 1. The traffic
# ==========================================
def plot_traffic() -> Path:
    """Messages per hour over one test week, at 10,000 households, in each scenario."""
    questions = list(range(1319))
    figure, panels = new_figure(height=TRAFFIC_HEIGHT, rows=3, sharex=True, sharey=True)
    drift = {"main": (0.0, 12.3), "burst_week": (0.235, 3.18), "burst_all": (0.457, 12.3)}
    rows = []
    for axes, (scenario, (sigma, hours)) in zip(panels, drift.items()):
        homes = Households(SIZE, 1.0, 3.6, burst_sigma=sigma, burst_hours=hours)
        times = np.array([hour for _, hour, _ in homes.stream(questions, 7, 28, seed=8)]) - 28 * 24
        counts = np.bincount(times.astype(int), minlength=7 * 24)[: 7 * 24]
        name, color, _ = SCENARIOS[scenario]
        style(axes, "", "")
        axes.plot(np.arange(7 * 24) / 24, counts, color=color, linewidth=1)
        axes.annotate(
            name,
            (0.005, 0.95),
            xycoords="axes fraction",
            va="top",
            fontsize=8,
            color=INK,
        )
        rows += [
            point(
                scenario=scenario,
                households=SIZE,
                series=name,
                x_name="hour_of_week",
                x=hour,
                y_name="messages_per_hour",
                y=int(count),
            )
            for hour, count in enumerate(counts)
        ]
    panels[-1].set_xticks(range(8))
    panels[-1].set_xlim(0, 7)
    panels[-1].set_ylim(0, 1.25 * max(row["y"] for row in rows))
    panels[-1].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    panels[-1].set_xlabel("Day of the test week", color=INK)
    panels[1].set_ylabel("Messages per hour", color=INK)
    return save(figure, "traffic_week", rows)


# ==========================================
# 2. The β knob
# ==========================================
def plot_beta_knob() -> Path:
    """How β moves accuracy and energy, for each policy, beside each tier alone."""
    runs = load_runs("main")
    figure, panels = new_figure(columns=2)
    style(panels[0], BETA, ACCURACY, "(a)")
    style(panels[1], BETA, ENERGY, "(b)")
    rows = []
    for policy, (name, color, marker) in SERIES.items():
        for axes, column, y_name in (
            (panels[0], "accuracy", "accuracy"),
            (panels[1], "J_per_query", "J_per_query"),
        ):
            betas, points = per_beta(runs, policy, SIZE, column)
            line(axes, betas, [value[0] for value in points], name, color, marker)
            rows += ranged(
                name,
                zip(betas, points),
                panel="a" if axes is panels[0] else "b",
                scenario="main",
                households=SIZE,
                x_name="beta",
                y_name=y_name,
            )
    for policy, name in ALONE.items():
        for axes, column in ((panels[0], "accuracy"), (panels[1], "J_per_query")):
            mean, low, high = tier_alone(policy, column)
            # named where no curve passes: the OLT's line at the low betas, the others' at the high
            if policy == "olt_alone":
                reference_line(axes, mean, name, 0.1)
            else:
                reference_line(axes, mean, name, 0.9, "right")
            rows.append(
                point(
                    panel="a" if axes is panels[0] else "b",
                    scenario="main",
                    households=SIZE,
                    series=name,
                    y_name=column,
                    y=mean,
                    y_low=low,
                    y_high=high,
                )
            )
    for axes in panels:
        axes.set_xticks([round(0.1 * step, 1) for step in range(1, 10)])
    legend(panels[1], "upper left", bbox_to_anchor=(0.0, 0.97))
    return save(figure, "beta_knob", rows)


# ==========================================
# 3. Reading every policy at the same accuracy
# ==========================================
READING_LABELS = {"static_hour_self": (-8, 4, "right")}
"""Where each reading's value is written, so the closest ones do not collide."""

READING_DEFAULT = (8, -6, "left")


def plot_equal_accuracy() -> Path:
    """Energy against accuracy, where each policy's curve crosses 0.80, and each tier alone."""
    runs = load_runs("main")
    figure, (axes,) = new_figure()
    style(axes, f"{ACCURACY} (each marker one $\\beta$, 0.1 to 0.9)", ENERGY)
    rows = []
    for policy, name in ALONE.items():
        accuracy = tier_alone(policy, "accuracy")[0]
        mean, low, high = tier_alone(policy, "J_per_query")
        axes.plot(
            [accuracy], [mean], marker="X", markersize=6, color=MUTED, linestyle="none", zorder=4
        )
        axes.annotate(
            f"{name}, {mean:.0f} J",
            (accuracy, mean),
            xytext={"user_alone": (7, 0), "onu_alone": (4, 6), "olt_alone": (-4, 6)}[policy],
            textcoords="offset points",
            ha="right" if policy == "olt_alone" else "left",
            va="center" if policy == "user_alone" else "baseline",
            fontsize=7.5,
            color=MUTED,
        )
        rows.append(
            point(
                scenario="main",
                households=SIZE,
                series=name,
                x_name="accuracy",
                x=accuracy,
                y_name="J_per_query",
                y=mean,
                y_low=low,
                y_high=high,
            )
        )
    axes.axvline(float(TARGET), color=MUTED, linewidth=0.8, linestyle="--")
    for policy, (name, color, marker) in SERIES.items():
        _, accuracy = per_beta(runs, policy, SIZE, "accuracy")
        _, joules = per_beta(runs, policy, SIZE, "J_per_query")
        line(
            axes,
            [value[0] for value in accuracy],
            [value[0] for value in joules],
            name,
            color,
            marker,
        )
        rows += ranged(
            name,
            zip([value[0] for value in accuracy], joules),
            scenario="main",
            households=SIZE,
            x_name="accuracy",
            y_name="J_per_query",
        )
        reading = at_accuracy(runs, policy, SIZE)
        axes.plot(
            [float(TARGET)],
            [reading[0]],
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=INK,
            markeredgewidth=0.9,
        )
        dx, dy, align = READING_LABELS.get(policy, READING_DEFAULT)
        axes.annotate(
            f"{reading[0]:.0f} J",
            (float(TARGET), reading[0]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=8,
            color=INK,
        )
        rows += ranged(
            f"{name} at {TARGET}",
            [(float(TARGET), reading)],
            scenario="main",
            households=SIZE,
            x_name="accuracy",
            y_name="J_per_query",
        )
    axes.set_xlim(0.44, 0.95)
    legend(axes)
    return save(figure, "equal_accuracy", rows)


# ==========================================
# 4. Where queries are answered
# ==========================================
def plot_tiers() -> Path:
    """Share of queries answered at each tier, by β, under RecServe and the broadcast."""
    runs = load_runs("main")
    figure, panels = new_figure(columns=2, sharey=True)
    rows = []
    for axes, policy, letter in zip(panels, ("recserve", "broadcast"), "ab"):
        style(
            axes,
            BETA,
            "Share of queries (%)" if letter == "a" else "",
            f"({letter}) {SERIES[policy][0]}",
        )
        bottom = None
        for tier, (name, color) in TIERS.items():
            betas, points = per_beta(runs, policy, SIZE, f"final_{tier}")
            share = np.array([value[0] for value in points]) * 100
            axes.bar(
                betas,
                share,
                width=0.08,
                bottom=bottom,
                color=color,
                label=name,
                edgecolor="white",
                linewidth=0.6,
            )
            for beta, value, base in zip(betas, share, bottom if bottom is not None else 0 * share):
                if value >= 10:
                    axes.annotate(
                        f"{value:.0f}",
                        (beta, base + value / 2),
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if tier != "user" else INK,
                    )
            rows += ranged(
                name,
                [(beta, tuple(100 * part for part in value)) for beta, value in zip(betas, points)],
                panel=letter,
                scenario="main",
                households=SIZE,
                x_name="beta",
                y_name=f"share_answered_{tier}_percent",
            )
            bottom = share if bottom is None else bottom + share
        axes.set_ylim(0, 100)
        axes.set_xticks([round(0.1 * step, 1) for step in range(1, 10)])
        axes.grid(axis="x", visible=False)
    handles, names = panels[0].get_legend_handles_labels()
    figure.legend(
        handles,
        names,
        loc="upper center",
        ncol=3,
        frameon=False,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 1.0),
    )
    return save(figure, "tiers", rows, top=0.9)


# ==========================================
# 5. The batching model against the GPU
# ==========================================
def plot_batching_validation() -> Path:
    """The static sweep against continuous batching, and the simulator against the GPU."""
    report = json.loads(sorted(RESULTS.glob("gpu_energy_continuous_*.json"))[-1].read_text())
    static = {row["batch"]: row for row in olt_reference()["batch_curve"]}
    figure, panels = new_figure(columns=2)
    style(panels[0], "Requests generating together", "J per generated token (GPU)", "(a)")
    levels = [row["concurrency"] for row in report["fixed"]]
    measured = [row["J_per_generated_token"] for row in report["fixed"]]
    expected = [
        static[level]["decode_J_per_output_token"]
        + static[level]["prefill_J_per_input_token"]
        * row["prompt_tokens"]
        / row["generated_tokens"]
        for level, row in zip(levels, report["fixed"])
    ]
    line(panels[0], levels, expected, "Static sweep (simulator)", "#2a78d6", "o")
    line(panels[0], levels, measured, "Continuous batching", "#eb6834", "s", linestyle="--")
    panels[0].set_xscale("log")
    panels[0].set_yscale("log")
    panels[0].set_xticks(levels[::2] + [64], [str(level) for level in levels[::2] + [64]])
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    panels[0].set_yticks(ticks, [f"{tick:g}" for tick in ticks])
    panels[0].minorticks_off()
    legend(panels[0], "upper right")
    rows = [
        point(
            panel="a",
            series=series,
            x_name="concurrency",
            x=level,
            y_name="J_per_generated_token",
            y=value,
        )
        for series, values in (
            ("Static sweep (simulator)", expected),
            ("Continuous batching", measured),
        )
        for level, value in zip(levels, values)
    ]

    style(panels[1], "Energy the GPU spent (kJ)", "Energy the simulator predicts (kJ)", "(b)")
    for key, name, color, marker in (
        ("gross", "Gross, average accounting", "#2a78d6", "o"),
        ("net", "Net of idle, marginal", "#eb6834", "s"),
    ):
        pairs = [compare_run(run, report["idle_power_W"])[key] for run in report["poisson"]]
        panels[1].scatter(
            [pair[0] / 1000 for pair in pairs],
            [pair[1] / 1000 for pair in pairs],
            s=16,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.6,
            label=name,
            zorder=3,
        )
        rows += [
            point(
                panel="b",
                series=name,
                x_name="measured_kJ",
                x=pair[0] / 1000,
                y_name="predicted_kJ",
                y=pair[1] / 1000,
            )
            for pair in pairs
        ]
    limits = (4, 25)
    panels[1].plot(limits, limits, color=MUTED, linewidth=0.8, linestyle=":", label="Exact")
    panels[1].fill_between(
        limits,
        [0.9 * x for x in limits],
        [1.1 * x for x in limits],
        color=GRID,
        alpha=0.7,
        linewidth=0,
        label="±10% criterion",
    )
    panels[1].set_xlim(*limits)
    panels[1].set_ylim(*limits)
    legend(panels[1])
    return save(figure, "batching_validation", rows)


# ==========================================
# 6. The results
# ==========================================
def plot_savings_over_recserve() -> Path:
    """Each energy-aware policy's saving over RecServe, by population, no drift."""
    runs = load_runs("main")
    figure, (axes,) = new_figure()
    style(axes, "Households on the OLT", f"Energy saved over RecServe at {TARGET} (%)")
    rows = []
    for policy in ("static_hour_self", "broadcast"):
        name, color, marker = SERIES[policy]
        points = savings(runs, policy, "recserve")
        ranged_line(axes, SIZES, points, name, color, marker)
        rows += ranged(
            name,
            zip(SIZES, points),
            scenario="main",
            x_name="households",
            y_name="saving_over_recserve_percent",
        )
    households_axis(axes)
    legend(axes)
    return save(figure, "saving_over_recserve", rows)


def plot_broadcast_over_timetable() -> Path:
    """The broadcast's saving over the timetable, by population and drift."""
    figure, (axes,) = new_figure()
    style(axes, "Households on the OLT", f"Broadcast's saving over the timetable at {TARGET} (%)")
    axes.axhline(0, color=MUTED, linewidth=0.6)
    rows = []
    for scenario, (name, color, marker) in SCENARIOS.items():
        points = savings(load_runs(scenario), "broadcast", "static_hour_self")
        ranged_line(axes, SIZES, points, name, color, marker)
        rows += ranged(
            name,
            zip(SIZES, points),
            scenario=scenario,
            x_name="households",
            y_name="saving_over_timetable_percent",
        )
    households_axis(axes)
    legend(axes)
    return save(figure, "broadcast_over_timetable", rows)


def plot_drift() -> Path:
    """Each policy's energy at 0.80 accuracy as the drift grows, at a fixed population."""
    figure, (axes,) = new_figure()
    style(axes, "", f"{ENERGY} at {TARGET} accuracy")
    width = 0.26
    rows = []
    for slot, (policy, (name, color, _)) in enumerate(SERIES.items()):
        points = [at_accuracy(load_runs(scenario), policy, SIZE) for scenario in SCENARIOS]
        positions = np.arange(len(SCENARIOS)) + (slot - 1) * width
        axes.bar(
            positions,
            [value[0] for value in points],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=name,
            **bar_errors(points),
        )
        for position, value in zip(positions, points):
            axes.annotate(
                f"{value[0]:.0f}",
                (position, value[2]),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK,
            )
        rows += [
            point(
                scenario=scenario,
                households=SIZE,
                series=name,
                x_name="scenario",
                x=SCENARIOS[scenario][0],
                y_name="J_per_query_at_target",
                y=value[0],
                y_low=value[1],
                y_high=value[2],
            )
            for scenario, value in zip(SCENARIOS, points)
        ]
    axes.set_xticks(range(len(SCENARIOS)), [name for name, _, _ in SCENARIOS.values()])
    axes.grid(axis="x", visible=False)
    legend(axes, "upper right", ncol=3)
    axes.set_ylim(0, 260)
    return save(figure, "drift", rows)


def plot_bandwidth() -> Path:
    """RecServe's communication burden against our policies', next to the energy each spends."""
    runs = load_runs("main")
    figure, panels = new_figure(columns=2)
    style(
        panels[0],
        f"{ACCURACY} (each marker one $\\beta$)",
        "MB between tiers per 1,000 queries",
        "(a)",
    )
    rows = []
    for policy, (name, color, marker) in SERIES.items():
        _, accuracy = per_beta(runs, policy, SIZE, "accuracy")
        _, megabytes = per_beta(runs, policy, SIZE, "comm_MB_per_1k_queries")
        line(
            panels[0],
            [value[0] for value in accuracy],
            [value[0] for value in megabytes],
            name,
            color,
            marker,
        )
        rows += ranged(
            name,
            zip([value[0] for value in accuracy], megabytes),
            panel="a",
            scenario="main",
            households=SIZE,
            x_name="accuracy",
            y_name="comm_MB_per_1k_queries",
        )
    panels[0].axvline(float(TARGET), color=MUTED, linewidth=0.8, linestyle="--")
    legend(panels[0])

    style(panels[1], "", f"Change against RecServe at {TARGET} (%)", "(b)")
    panels[1].axhline(0, color=MUTED, linewidth=0.6)
    measures = (
        ("Energy per query", "J_per_query"),
        ("Communication burden", "comm_MB_per_1k_queries"),
    )
    width = 0.36
    for slot, policy in enumerate(("static_hour_self", "broadcast")):
        name, color, _ = SERIES[policy]
        points = []
        for _, column in measures:
            per_seed = []
            for run in runs:
                values = at_column(run, SIZE, TARGET, column)
                per_seed.append(100 * (values[policy] / values["recserve"] - 1))
            points.append(spread(per_seed))
        positions = np.arange(len(measures)) + (slot - 0.5) * width
        panels[1].bar(
            positions,
            [value[0] for value in points],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=name,
            **bar_errors(points),
        )
        for position, value in zip(positions, points):
            panels[1].annotate(
                f"{value[0]:+.0f}%",
                (position, value[2] if value[0] >= 0 else value[1]),
                xytext=(0, 2 if value[0] >= 0 else -2),
                textcoords="offset points",
                ha="center",
                va="bottom" if value[0] >= 0 else "top",
                fontsize=7.5,
                color=INK,
            )
        rows += [
            point(
                panel="b",
                scenario="main",
                households=SIZE,
                series=name,
                x_name="measure",
                x=column,
                y_name="change_against_recserve_percent",
                y=value[0],
                y_low=value[1],
                y_high=value[2],
            )
            for (_, column), value in zip(measures, points)
        ]
    panels[1].set_xticks(range(len(measures)), [label for label, _ in measures])
    panels[1].set_ylim(-52, 18)
    panels[1].grid(axis="x", visible=False)
    legend(panels[1], "lower right")
    return save(figure, "bandwidth", rows)


# ==========================================
# Index
# ==========================================
def write_index() -> Path:
    """Write README.md beside the figures: each one's files and draft caption."""
    lines = [
        "# Case-study figures",
        "",
        "Written by `src/analyze/plot_study.py`; do not edit by hand. Each figure is drawn at",
        "the thesis's text width (include it with `width=\\textwidth`), as a vector PDF and a",
        "300 dpi PNG, with the numbers behind it in `data/<name>.csv`.",
        "",
        "Every data file has the same columns: "
        + ", ".join(f"`{name}`" for name in DATA_COLUMNS)
        + ". `y` is a mean over seeds 7, 8 and 9 and `y_low`, `y_high` their range, when the",
        "point has seeds; a cell that does not apply is empty.",
        "",
    ]
    for name, caption in CAPTIONS.items():
        lines += [
            f"## {name}",
            "",
            f"`{name}.pdf` · `{name}.png` · `data/{name}.csv`",
            "",
            caption,
            "",
        ]
    path = FIGURES / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    """Draw every figure, write its data and the index, and print where each went.

    Returns:
        The process exit code.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    for draw in (
        plot_traffic,
        plot_beta_knob,
        plot_equal_accuracy,
        plot_tiers,
        plot_batching_validation,
        plot_savings_over_recserve,
        plot_broadcast_over_timetable,
        plot_drift,
        plot_bandwidth,
    ):
        print(f"wrote {draw()}")
    print(f"wrote {write_index()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
