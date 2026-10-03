#!/usr/bin/env python3
"""Check that skipping tiers does not disturb RecServe's beta-quantile windows.

The energy-aware policies forward some queries past the phone or the ONU. If that changed
which queries a tier sees, its confidence window would shift and RecServe's threshold would
lose its meaning. This runs the simulator with instrumented windows and reports, per policy,
population, beta and tier: the share of queries the tier ran, how often it escalated them
(RecServe's promise: about beta) and their mean confidence.

Usage: python src/analyze/check_beta_windows.py [simulate.py arguments]
"""

# pylint: disable=wrong-import-position

import contextlib
import io
import itertools
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulate"))
import cascade
import simulate
from energy.three_tier import TIERS
from routing import Window

STATS: dict = {}
"""Per tier, for the run being recorded: ran, judged, escalated, confidences. Empty: off."""

NAMES = itertools.cycle(TIERS)
"""cascade.run builds one window per tier, in TIERS order, every run."""

DEFAULT_ARGS = [
    "--config",
    str(simulate.CONFIG.parent / "study.yaml"),
    "--subscribers",
    "5000,20000",
    "--betas",
    "0.2,0.5,0.8",
    "--policies",
    "recserve,static_hour,broadcast",
]
"""What the check runs when given no arguments."""


class Probe(Window):
    """RecServe's confidence window that also records what its tier ran and escalated."""

    def __init__(self, size):
        """Create the window and name it after its tier.

        Args:
            size: How many recent confidences to keep.
        """
        super().__init__(size)
        self.tier, self.threshold = next(NAMES), None

    def quantile(self, p):
        """Return Window.quantile, remembering the threshold the next answer is judged against."""
        self.threshold = super().quantile(p)
        return self.threshold

    def add(self, confidence):
        """Count whether this answer was judged and escalated, then record it as Window.add does.

        Args:
            confidence: The confidence of the answer the tier just produced.
        """
        stats = STATS.get(self.tier)
        if stats is not None:
            stats["ran"] += 1
            stats["confidences"].append(confidence)
            if self.threshold is not None:
                stats["judged"] += 1
                stats["escalated"] += confidence < self.threshold
        self.threshold = None
        super().add(confidence)


def main() -> int:
    """Run the simulator with Probe windows and print the per-tier table.

    Returns:
        The process exit code.
    """
    rows, population = [], {}
    real_run, real_run_policies = simulate.run, simulate.run_policies

    def run_policies(setup, subscribers, betas, policies):
        """Remember which population the runs below belong to."""
        population["subscribers"] = subscribers
        return real_run_policies(setup, subscribers, betas, policies)

    def run(setup, beta, policy):
        """Call cascade.run with fresh per-tier counters, keeping them for the table."""
        STATS.update(
            {tier: {"ran": 0, "judged": 0, "escalated": 0, "confidences": []} for tier in TIERS}
        )
        result = real_run(setup, beta, policy)
        rows.append(
            (
                policy,
                population["subscribers"],
                beta,
                result[0]["accuracy"],
                len(setup.stream),
                dict(STATS),
            )
        )
        STATS.clear()
        return result

    # --- Run the simulator, instrumented ---
    cascade.Window, simulate.run, simulate.run_policies = Probe, run, run_policies
    with tempfile.TemporaryDirectory() as scratch:
        sys.argv = [
            "simulate.py",
            *(sys.argv[1:] or DEFAULT_ARGS),
            "--out",
            str(Path(scratch) / "check.csv"),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            simulate.main()

    # --- Report ---
    below_top = TIERS[:-1]  # the top tier never escalates
    print(
        f"{'policy':>11} {'homes':>7} {'beta':>4} {'acc':>6} | "
        + " | ".join(f"{tier}: ran  escalated  mean conf" for tier in below_top)
    )
    for policy, subscribers, beta, accuracy, queries, stats in rows:
        cells = []
        for tier in below_top:
            tier_stats = stats[tier]
            escalated = (
                tier_stats["escalated"] / tier_stats["judged"]
                if tier_stats["judged"]
                else float("nan")
            )
            confidence = (
                statistics.mean(tier_stats["confidences"])
                if tier_stats["confidences"]
                else float("nan")
            )
            cells.append(f"{tier_stats['ran'] / queries:6.1%} {escalated:9.1%} {confidence:9.3f}")
        print(f"{policy:>11} {subscribers:7d} {beta:4.1f} {accuracy:6.3f} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
