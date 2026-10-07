#!/usr/bin/env python3
"""Plot the case study's main results: results/study/figures/*.png.

Every point is a mean over the seeds 7, 8 and 9; every saving is read at equal accuracy
(simulate/frontier.py), as SUMMARY.md reports it.
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from energy.three_tier import STUDY  # noqa: E402

FIGURES = STUDY / "figures"
"""Where the figures are written."""

SEEDS = (7, 8, 9)
SIZES = (1000, 2000, 5000, 10000, 20000)

# Categorical slots 1-4 of the validated default palette, in fixed order. Two of them sit
# below 3:1 against the surface, so every line also carries a direct label and its own marker.
SERIES = {
    "recserve": ("RecServe", "#2a78d6", "o"),
    "static_hour": ("Timetable, learned under RecServe", "#eb6834", "s"),
    "static_hour_self": ("Timetable, relearned under itself", "#1baf7a", "^"),
    "broadcast": ("Broadcast", "#eda100", "D"),
}
SCENARIOS = {
    "main": ("Average days", "#2a78d6", "o"),
    "burst_week": ("Drift within a week", "#eb6834", "s"),
    "burst_all": ("All of BurstGPT's drift", "#1baf7a", "^"),
}
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


def curve(runs: list[dict], policy: str, size: int) -> tuple[list[float], list[float]]:
    """Return a policy's (accuracy, J per query) at each beta, as means over the seeds."""
    by_beta: dict[float, list[dict]] = {}
    for run in runs:
        for row in run["rows"]:
            if row["policy"] == policy and row["subscribers"] == size:
                by_beta.setdefault(row["beta"], []).append(row)
    betas = sorted(by_beta)
    return (
        [statistics.mean(row["accuracy"] for row in by_beta[beta]) for beta in betas],
        [statistics.mean(row["J_per_query"] for row in by_beta[beta]) for beta in betas],
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
        out.append((statistics.mean(per_seed), min(per_seed), max(per_seed)) if per_seed else None)
    return out


# ==========================================
# Drawing
# ==========================================
def style(axes, title: str, xlabel: str, ylabel: str) -> None:
    """Give a chart recessive axes and grid, and its title and labels in text ink."""
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    axes.set_xlabel(xlabel, color=MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=8)


def label_end(axes, xs, ys, text: str, color: str, offset: float = 0.0) -> None:
    """Write a series' name beside its last point, in text ink, with a colour swatch."""
    axes.annotate(
        f"■ {text}",
        (xs[-1], ys[-1]),
        xytext=(8, offset),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK,
    )
    axes.annotate(
        "■",
        (xs[-1], ys[-1]),
        xytext=(8, offset),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=color,
    )


def plot_frontiers() -> Path:
    """Energy per query against accuracy, at 10,000 households, average days and full drift."""
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    figure.patch.set_facecolor(SURFACE)
    for axes, scenario in zip(panels, ("main", "burst_all")):
        runs = load_runs(scenario)
        style(
            axes,
            f"{SCENARIOS[scenario][0]}, 10,000 households",
            "Accuracy delivered (each point one RecServe beta, 0.1 to 0.9)",
            "Energy per query (J)" if scenario == "main" else "",
        )
        offsets = {"recserve": 0, "static_hour": 6, "static_hour_self": -6, "broadcast": -16}
        for policy, (name, color, marker) in SERIES.items():
            accuracy, joules = curve(runs, policy, 10000)
            axes.plot(
                accuracy,
                joules,
                color=color,
                linewidth=2,
                marker=marker,
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=name,
            )
            label_end(axes, accuracy, joules, name, color, offsets[policy])
        axes.set_xlim(0.48, 1.08)
    panels[0].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    figure.tight_layout()
    path = FIGURES / "frontiers_10000.png"
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_savings_over_recserve() -> Path:
    """Each energy-aware policy's saving over RecServe, by population, average days."""
    runs = load_runs("main")
    figure, axes = plt.subplots(figsize=(7.5, 4.4))
    figure.patch.set_facecolor(SURFACE)
    style(
        axes,
        "Saving over RecServe at 0.80 accuracy, average days",
        "Households on the OLT",
        "Energy saved (%)",
    )
    offsets = {"static_hour": -8, "static_hour_self": 0, "broadcast": 8}
    for policy in ("static_hour", "static_hour_self", "broadcast"):
        name, color, marker = SERIES[policy]
        points = savings(runs, policy, "recserve")
        means = [point[0] for point in points]
        axes.plot(
            SIZES,
            means,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=name,
        )
        label_end(axes, SIZES, means, f"{name} ({means[-1]:.0f}%)", color, offsets[policy])
    axes.set_xscale("log")
    axes.set_xticks(SIZES, [f"{size:,}" for size in SIZES])
    axes.set_xlim(800, 60000)
    axes.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    figure.tight_layout()
    path = FIGURES / "saving_over_recserve.png"
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_broadcast_over_timetable() -> Path:
    """The broadcast's saving over the self-calibrated timetable, by population and drift."""
    figure, axes = plt.subplots(figsize=(7.5, 4.4))
    figure.patch.set_facecolor(SURFACE)
    style(
        axes,
        "Broadcast's saving over the timetable relearned under itself, 0.80 accuracy",
        "Households on the OLT",
        "Energy saved (%), mean and range of 3 seeds",
    )
    axes.axhline(0, color=MUTED, linewidth=0.8)
    offsets = {"main": -6, "burst_week": 2, "burst_all": 8}
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
        label_end(
            axes, SIZES, means, f"{name} ({means[-2]:+.1f}% at 10,000)", color, offsets[scenario]
        )
    axes.set_xscale("log")
    axes.set_xticks(SIZES, [f"{size:,}" for size in SIZES])
    axes.set_xlim(800, 80000)
    axes.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    figure.tight_layout()
    path = FIGURES / "broadcast_over_timetable.png"
    figure.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return path


def main() -> int:
    """Draw every figure and print where each was written.

    Returns:
        The process exit code.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (plot_frontiers(), plot_savings_over_recserve(), plot_broadcast_over_timetable()):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
