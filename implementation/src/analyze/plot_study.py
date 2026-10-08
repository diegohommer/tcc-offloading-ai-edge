#!/usr/bin/env python3
"""Plot the case study, from the traffic to the results: results/study/figures/*.png.

Every point of a result is a mean over the seeds 7, 8 and 9, and every saving is read at
equal accuracy (simulate/frontier.py), as SUMMARY.md reports it.
"""

# pylint: disable=wrong-import-position,ungrouped-imports

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib
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

SEEDS = (7, 8, 9)
SIZES = (1000, 2000, 5000, 10000, 20000)
SIZE = 10000
"""The household count the per-beta figures show."""

# Categorical slots of the validated default palette, in fixed order. Some sit below 3:1
# against the surface, so every series also carries a direct label and its own marker.
SERIES = {
    "recserve": ("RecServe", "#2a78d6", "o"),
    "static_hour_self": ("Timetable", "#1baf7a", "^"),
    "broadcast": ("Broadcast", "#eda100", "D"),
}
SCENARIOS = {
    "main": ("No drift", "#2a78d6", "o"),
    "burst_week": ("Drift within a week", "#eb6834", "s"),
    "burst_all": ("All of BurstGPT's drift", "#1baf7a", "^"),
}
TIERS = {"user": ("Phone", "#e87ba4"), "onu": ("ONU", "#008300"), "olt": ("OLT", "#4a3aa7")}
BETA = "β (escalation quantile)"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ==========================================
# Reading the runs
# ==========================================
def load_runs(scenario: str) -> list[dict]:
    """Return one scenario's run JSONs, one per seed."""
    return [
        json.loads((STUDY / f"study_{scenario}_seed{seed}.json").read_text(encoding="utf-8"))
        for seed in SEEDS
    ]


def per_beta(runs: list[dict], policy: str, size: int, column: str) -> tuple[list, list]:
    """Return (betas, the column's mean over the seeds at each beta) for one policy and size."""
    by_beta: dict[float, list[float]] = {}
    for run in runs:
        for row in run["rows"]:
            if row["policy"] == policy and row["subscribers"] == size:
                by_beta.setdefault(row["beta"], []).append(row[column])
    betas = sorted(by_beta)
    return betas, [statistics.mean(by_beta[beta]) for beta in betas]


def at_accuracy(runs: list[dict], policy: str, size: int, accuracy: str = "0.80") -> float:
    """Return a policy's J per query at a target accuracy, mean over the seeds."""
    return statistics.mean(
        entry["J_at_accuracy"][accuracy]
        for run in runs
        for entry in run["frontiers"]
        if entry["subscribers"] == size and entry["policy"] == policy
    )


def savings(runs: list[dict], policy: str, reference: str, accuracy: str = "0.80") -> list:
    """Return (mean, low, high) of a policy's saving over a reference, per size, across seeds."""
    out = []
    for size in SIZES:
        per_seed = []
        for run in runs:
            energy = {
                entry["policy"]: entry["J_at_accuracy"][accuracy]
                for entry in run["frontiers"]
                if entry["subscribers"] == size
            }
            if energy.get(policy) and energy.get(reference):
                per_seed.append(100 * (1 - energy[policy] / energy[reference]))
        out.append((statistics.mean(per_seed), min(per_seed), max(per_seed)))
    return out


# ==========================================
# Drawing helpers
# ==========================================
def new_figure(columns: int = 1, width: float = 7.5, sharey: bool = False):
    """Return a figure and its panels on the chart surface."""
    figure, panels = plt.subplots(1, columns, figsize=(width, 4.4), sharey=sharey)
    figure.patch.set_facecolor(SURFACE)
    return figure, np.atleast_1d(panels)


def style(axes, title: str, xlabel: str, ylabel: str) -> None:
    """Give a chart recessive axes and grid, and its title and labels in text ink."""
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=INK, fontsize=10.5, pad=10)
    axes.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=8)


def line(axes, xs, ys, name: str, color: str, marker: str, **extra) -> None:
    """Draw one series: a 2 px line with surface-ringed markers."""
    axes.plot(
        xs,
        ys,
        color=color,
        linewidth=2,
        marker=marker,
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=name,
        **extra,
    )


def label_end(axes, xs, ys, text: str, color: str, offset: float = 0.0) -> None:
    """Write a series' name beside its last point, in text ink after a colour swatch."""
    axes.annotate(
        "■",
        (xs[-1], ys[-1]),
        xytext=(8, offset),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=color,
    )
    axes.annotate(
        text,
        (xs[-1], ys[-1]),
        xytext=(18, offset),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK,
    )


def legend(axes, where: str = "upper left") -> None:
    """Add the legend, kept for every chart of two or more series."""
    axes.legend(loc=where, frameon=False, fontsize=8, labelcolor=INK)


def households_axis(axes, right: float = 60000) -> None:
    """Put the household counts on a log axis."""
    axes.set_xscale("log")
    axes.set_xticks(SIZES, [f"{size:,}" for size in SIZES])
    axes.minorticks_off()
    axes.set_xlim(800, right)


def save(figure, name: str) -> Path:
    """Write a figure to FIGURES and close it."""
    figure.tight_layout()
    path = FIGURES / name
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


# ==========================================
# 1. The traffic
# ==========================================
def plot_traffic() -> Path:
    """Messages per hour over one test week, at 10,000 households, without and with drift."""
    questions = list(range(1319))
    figure, (axes,) = new_figure(width=11)
    style(
        axes,
        f"Messages sent per hour, one week, {SIZE:,} households",
        "Day of the week (each day starts at midnight)",
        "Messages per hour",
    )
    for scenario, sigma, hours in (("main", 0.0, 12.3), ("burst_all", 0.457, 12.3)):
        homes = Households(SIZE, 1.0, 3.6, burst_sigma=sigma, burst_hours=hours)
        times = np.array([hour for _, hour, _ in homes.stream(questions, 7, 28, seed=8)]) - 28 * 24
        counts = np.bincount(times.astype(int), minlength=7 * 24)[: 7 * 24]
        name, color, _ = SCENARIOS[scenario]
        axes.plot(np.arange(7 * 24) / 24, counts, color=color, linewidth=1.6, label=name)
    axes.set_xticks(range(8))
    legend(axes)
    return save(figure, "traffic_week.png")


# ==========================================
# 2. The beta knob
# ==========================================
def plot_beta_knob() -> Path:
    """How β moves accuracy and energy, for each policy."""
    runs = load_runs("main")
    figure, panels = new_figure(columns=2, width=11)
    style(
        panels[0],
        "Accuracy delivered rises with β",
        BETA,
        "Share of answers that are right",
    )
    style(panels[1], "So does the energy spent", BETA, "Energy per query (J)")
    for policy, (name, color, marker) in SERIES.items():
        betas, accuracy = per_beta(runs, policy, SIZE, "accuracy")
        _, joules = per_beta(runs, policy, SIZE, "J_per_query")
        line(panels[0], betas, accuracy, name, color, marker)
        line(panels[1], betas, joules, name, color, marker)
    for policy, name in ALONE.items():
        accuracy, joules = tier_alone(policy)
        for axes, value, text in ((panels[0], accuracy, name), (panels[1], joules, name)):
            axes.axhline(value, color=MUTED, linewidth=0.8, linestyle=":")
            axes.annotate(
                text,
                (0.1, value),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7.5,
                color=MUTED,
            )
    legend(panels[1])
    figure.suptitle(
        f"No drift, {SIZE:,} households, mean of 3 seeds",
        x=0.01,
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    return save(figure, "beta_knob.png")


# ==========================================
# 3. Reading every policy at the same accuracy
# ==========================================
READING_LABELS = {
    "recserve": (-46, 0),
    "static_hour_self": (-46, 6),
    "broadcast": (12, -9),
}
"""Where each reading's value is written, so the two closest ones do not collide."""

ALONE = {"user_alone": "Phone alone", "onu_alone": "ONU alone", "olt_alone": "OLT alone"}
"""The single-tier runs (results/study/alone_seed*.json), each tier answering every query."""


def tier_alone(policy: str) -> tuple[float, float]:
    """Return (accuracy, J per query) of one tier answering every query, mean over the seeds."""
    rows = [
        row
        for seed in SEEDS
        for row in json.loads((STUDY / f"alone_seed{seed}.json").read_text(encoding="utf-8"))[
            "rows"
        ]
        if row["policy"] == policy and row["subscribers"] == SIZE
    ]
    return (
        statistics.mean(row["accuracy"] for row in rows),
        statistics.mean(row["J_per_query"] for row in rows),
    )


def plot_equal_accuracy() -> Path:
    """Energy against accuracy, where each policy's curve crosses 0.80, and each tier alone."""
    runs = load_runs("main")
    figure, (axes,) = new_figure(width=9)
    style(
        axes,
        f"Energy against accuracy, no drift, {SIZE:,} households",
        "Accuracy delivered (each marker one β, 0.1 to 0.9)",
        "Energy per query (J)",
    )
    for policy, name in ALONE.items():
        accuracy, joules = tier_alone(policy)
        axes.plot(
            [accuracy], [joules], marker="X", markersize=9, color=MUTED, linestyle="none", zorder=4
        )
        axes.annotate(
            f"{name}, {joules:.0f} J",
            (accuracy, joules),
            xytext=(-6, 9),
            textcoords="offset points",
            ha="right" if policy == "olt_alone" else "left",
            fontsize=8,
            color=MUTED,
        )
    axes.axvline(0.80, color=MUTED, linewidth=1, linestyle="--")
    axes.annotate(
        "target 0.80",
        (0.80, 262),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )
    for policy, (name, color, marker) in SERIES.items():
        _, accuracy = per_beta(runs, policy, SIZE, "accuracy")
        _, joules = per_beta(runs, policy, SIZE, "J_per_query")
        line(axes, accuracy, joules, name, color, marker)
        reading = at_accuracy(runs, policy, SIZE)
        axes.plot(
            [0.80],
            [reading],
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=INK,
            markeredgewidth=1.4,
        )
        axes.annotate(
            f"{reading:.0f} J",
            (0.80, reading),
            xytext=READING_LABELS[policy],
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    axes.set_xlim(0.44, 0.95)
    legend(axes)
    return save(figure, "equal_accuracy.png")


# ==========================================
# 4. Where queries are answered
# ==========================================
def plot_tiers() -> Path:
    """Share of queries answered at each tier, by β, under RecServe and the broadcast."""
    runs = load_runs("main")
    figure, panels = new_figure(columns=2, width=11, sharey=True)
    for axes, policy in zip(panels, ("recserve", "broadcast")):
        style(
            axes,
            f"{SERIES[policy][0]}: where queries are answered",
            BETA,
            f"Share of queries, {SIZE:,} households" if policy == "recserve" else "",
        )
        bottom = None
        for tier, (name, color) in TIERS.items():
            betas, share = per_beta(runs, policy, SIZE, f"final_{tier}")
            share = np.array(share) * 100
            axes.bar(
                betas,
                share,
                width=0.075,
                bottom=bottom,
                color=color,
                label=name,
                edgecolor=SURFACE,
                linewidth=1,
            )
            for beta, value, base in zip(betas, share, bottom if bottom is not None else 0 * share):
                if value >= 9:
                    axes.annotate(
                        f"{value:.0f}",
                        (beta, base + value / 2),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if tier != "user" else INK,
                    )
            bottom = share if bottom is None else bottom + share
        axes.set_ylim(0, 100)
        axes.set_xticks([round(0.1 * step, 1) for step in range(1, 10)])
    handles, names = panels[0].get_legend_handles_labels()
    figure.legend(
        handles,
        names,
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
        bbox_to_anchor=(0.99, 1.0),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIGURES / "tiers.png"
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


# ==========================================
# 5. The batching model against the GPU
# ==========================================
def plot_batching_validation() -> Path:
    """The static sweep against continuous batching, and the simulator against the GPU."""
    report = json.loads(sorted(RESULTS.glob("gpu_energy_continuous_*.json"))[-1].read_text())
    static = {row["batch"]: row for row in olt_reference()["batch_curve"]}
    figure, panels = new_figure(columns=2, width=11)
    style(
        panels[0],
        "Energy per generated token on the L4",
        "Requests generating together",
        "J per generated token (GPU card)",
    )
    levels = [row["concurrency"] for row in report["fixed"]]
    measured = [row["J_per_generated_token"] for row in report["fixed"]]
    expected = [
        static[level]["decode_J_per_output_token"]
        + static[level]["prefill_J_per_input_token"]
        * row["prompt_tokens"]
        / row["generated_tokens"]
        for level, row in zip(levels, report["fixed"])
    ]
    line(panels[0], levels, expected, "Static sweep (the simulator's curve)", "#2a78d6", "o")
    line(
        panels[0], levels, measured, "Continuous batching, measured", "#eb6834", "s", linestyle="--"
    )
    panels[0].set_xscale("log")
    panels[0].set_yscale("log")
    panels[0].set_xticks(levels[::2] + [64], [str(level) for level in levels[::2] + [64]])
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    panels[0].set_yticks(ticks, [f"{tick:g}" for tick in ticks])
    panels[0].minorticks_off()
    legend(panels[0], "upper right")

    style(
        panels[1],
        "Simulator replaying the measured arrivals",
        "Energy the GPU spent (kJ)",
        "Energy the simulator predicts (kJ)",
    )
    for key, name, color, marker in (
        ("gross", "Gross, average accounting", "#2a78d6", "o"),
        ("net", "Net of idle, marginal accounting", "#1baf7a", "^"),
    ):
        pairs = [compare_run(run, report["idle_power_W"])[key] for run in report["poisson"]]
        panels[1].scatter(
            [pair[0] / 1000 for pair in pairs],
            [pair[1] / 1000 for pair in pairs],
            s=40,
            color=color,
            marker=marker,
            edgecolors=SURFACE,
            linewidths=1.2,
            label=name,
            zorder=3,
        )
    limits = (4, 25)
    panels[1].plot(limits, limits, color=MUTED, linewidth=1, linestyle=":", label="Exact")
    panels[1].fill_between(
        limits,
        [0.9 * x for x in limits],
        [1.1 * x for x in limits],
        color=GRID,
        alpha=0.6,
        label="±10% criterion",
    )
    panels[1].set_xlim(*limits)
    panels[1].set_ylim(*limits)
    legend(panels[1])
    return save(figure, "batching_validation.png")


# ==========================================
# 6. The results
# ==========================================
def plot_savings_over_recserve() -> Path:
    """Each energy-aware policy's saving over RecServe, by population, no drift."""
    runs = load_runs("main")
    figure, (axes,) = new_figure()
    style(
        axes,
        "Saving over RecServe at 0.80 accuracy, no drift",
        "Households on the OLT",
        "Energy saved (%), mean of 3 seeds",
    )
    offsets = {"static_hour_self": -5, "broadcast": 5}
    for policy, offset in offsets.items():
        name, color, marker = SERIES[policy]
        means = [point[0] for point in savings(runs, policy, "recserve")]
        line(axes, SIZES, means, name, color, marker)
        label_end(axes, SIZES, means, f"{means[-1]:.0f}%", color, offset)
    households_axis(axes, right=30000)
    legend(axes)
    return save(figure, "saving_over_recserve.png")


def plot_broadcast_over_timetable() -> Path:
    """The broadcast's saving over the timetable, by population and drift."""
    figure, (axes,) = new_figure()
    style(
        axes,
        "Broadcast's saving over the timetable, 0.80 accuracy",
        "Households on the OLT",
        "Energy saved (%), mean and range of 3 seeds",
    )
    axes.axhline(0, color=MUTED, linewidth=0.8)
    for scenario, (name, color, marker) in SCENARIOS.items():
        points = savings(load_runs(scenario), "broadcast", "static_hour_self")
        means = [point[0] for point in points]
        axes.errorbar(
            SIZES,
            means,
            yerr=[
                [point[0] - point[1] for point in points],
                [point[2] - point[0] for point in points],
            ],
            color=color,
            linewidth=2,
            marker=marker,
            markersize=6,
            capsize=3,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=name,
        )
    households_axis(axes, right=30000)
    legend(axes)
    return save(figure, "broadcast_over_timetable.png")


def plot_drift() -> Path:
    """Each policy's energy at 0.80 accuracy as the drift grows, at a fixed population."""
    figure, (axes,) = new_figure(width=8.5)
    style(
        axes,
        f"Energy at 0.80 accuracy as the traffic drifts, {SIZE:,} households",
        "Drift in the traffic",
        "Energy per query (J), mean and range of 3 seeds",
    )
    width = 0.26
    for slot, (policy, (name, color, _)) in enumerate(SERIES.items()):
        means, lows, highs = [], [], []
        for scenario in SCENARIOS:
            per_seed = [
                entry["J_at_accuracy"]["0.80"]
                for run in load_runs(scenario)
                for entry in run["frontiers"]
                if entry["subscribers"] == SIZE and entry["policy"] == policy
            ]
            means.append(statistics.mean(per_seed))
            lows.append(means[-1] - min(per_seed))
            highs.append(max(per_seed) - means[-1])
        positions = np.arange(len(SCENARIOS)) + (slot - 1) * width
        axes.bar(
            positions,
            means,
            width=width,
            color=color,
            edgecolor=SURFACE,
            linewidth=1,
            yerr=[lows, highs],
            error_kw={"ecolor": MUTED, "capsize": 3, "linewidth": 1},
            label=name,
        )
        for position, value, high in zip(positions, means, highs):
            axes.annotate(
                f"{value:.0f}",
                (position, value + high),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK,
            )
    axes.set_xticks(range(len(SCENARIOS)), [name for name, _, _ in SCENARIOS.values()])
    axes.grid(axis="x", visible=False)
    legend(axes, "upper right")
    return save(figure, "drift.png")


def plot_bandwidth() -> Path:
    """RecServe's communication burden against our policies', next to the energy each spends."""
    runs = load_runs("main")
    figure, panels = new_figure(columns=2, width=11)
    style(
        panels[0],
        "RecServe's communication burden",
        "Accuracy delivered (each marker one β, 0.1 to 0.9)",
        "MB carried between tiers per 1,000 queries",
    )
    for policy, (name, color, marker) in SERIES.items():
        _, accuracy = per_beta(runs, policy, SIZE, "accuracy")
        _, megabytes = per_beta(runs, policy, SIZE, "comm_MB_per_1k_queries")
        line(panels[0], accuracy, megabytes, name, color, marker)
    panels[0].axvline(0.80, color=MUTED, linewidth=1, linestyle="--")
    legend(panels[0])

    style(
        panels[1],
        "What each policy changes at 0.80 accuracy",
        "",
        "Change against RecServe (%), mean of 3 seeds",
    )
    panels[1].axhline(0, color=MUTED, linewidth=0.8)
    measures = (
        ("Energy per query", "J_per_query"),
        ("Communication burden", "comm_MB_per_1k_queries"),
    )
    width = 0.36
    for slot, policy in enumerate(("static_hour_self", "broadcast")):
        name, color, _ = SERIES[policy]
        changes = []
        for _, column in measures:
            per_seed = []
            for run in runs:
                values = at_column(run, SIZE, "0.80", column)
                per_seed.append(100 * (values[policy] / values["recserve"] - 1))
            changes.append(statistics.mean(per_seed))
        positions = np.arange(len(measures)) + (slot - 0.5) * width
        panels[1].bar(
            positions, changes, width=width, color=color, edgecolor=SURFACE, linewidth=1, label=name
        )
        for position, value in zip(positions, changes):
            panels[1].annotate(
                f"{value:+.0f}%",
                (position, value),
                xytext=(0, 4 if value >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                color=INK,
            )
    panels[1].set_xticks(range(len(measures)), [name for name, _ in measures])
    panels[1].grid(axis="x", visible=False)
    legend(panels[1], "upper left")
    figure.suptitle(f"No drift, {SIZE:,} households", x=0.01, ha="left", fontsize=9, color=MUTED)
    return save(figure, "bandwidth.png")


def main() -> int:
    """Draw every figure and print where each was written.

    Returns:
        The process exit code.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    for old in FIGURES.glob("frontiers_*.png"):
        old.unlink()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
