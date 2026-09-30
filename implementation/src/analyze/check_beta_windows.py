#!/usr/bin/env python3
"""Check that skipping tiers does not disturb RecServe's beta-quantile windows.

The energy-aware policies forward some queries past the phone or the ONU. If that changed
which queries a tier sees, its confidence window would drift and RecServe's threshold would
lose its meaning. This runs the simulator with instrumented windows and reports, per policy,
beta and tier: the share of queries the tier ran, how often it escalated them (RecServe's
promise: about beta) and their mean confidence.

Usage: python src/analyze/check_beta_windows.py [simulate.py arguments]
"""

# pylint: disable=wrong-import-position

import contextlib
import io
import statistics as st
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
"""Per tier, for the run in progress: ran, judged, escalated, confidences."""

PENDING_NAMES: list[str] = []
"""Tier names still to hand out: cascade.run builds one window per tier, in TIERS order."""


class Probe(Window):
    """RecServe's confidence window that also records what its tier ran and escalated."""

    def __init__(self, size):
        """Create the window, naming it after the next tier in PENDING_NAMES.

        Args:
            size: How many recent confidences to keep.
        """
        super().__init__(size)
        self.tier, self.thr = PENDING_NAMES.pop(0), None

    def quantile(self, p):
        """Return Window.quantile, remembering the threshold the next answer is judged against."""
        self.thr = super().quantile(p)
        return self.thr

    def add(self, confidence):
        """Count whether this answer was judged and escalated, then record it as Window.add does.

        Args:
            confidence: The confidence of the answer the tier just produced.
        """
        s = STATS[self.tier]
        s["ran"] += 1
        s["conf"].append(confidence)
        if self.thr is not None:  # the tier judged this answer against its threshold
            s["judged"] += 1
            s["esc"] += confidence < self.thr
        self.thr = None
        super().add(confidence)


def main() -> int:
    """Run the simulator with Probe windows and print the per-tier table.

    Returns:
        The process exit code.
    """
    rows, real_run = [], cascade.run

    def run(setup, beta, policy):
        """Call cascade.run with fresh per-tier counters, keeping them for the table."""
        STATS.clear()
        STATS.update({t: {"ran": 0, "judged": 0, "esc": 0, "conf": []} for t in TIERS})
        PENDING_NAMES[:] = TIERS
        m, hr = real_run(setup, beta, policy)
        rows.append(
            (
                policy,
                max(setup.loads),
                beta,
                m["accuracy"],
                len(setup.stream),
                {t: dict(STATS[t]) for t in TIERS},
            )
        )
        return m, hr

    # --- Run the simulator, instrumented ---
    cascade.Window, simulate.run = Probe, run
    extra = sys.argv[1:] or [
        "--config",
        str(simulate.CONFIG.parent / "study.yaml"),
        "--surge-factor",
        "3",
        "--peak-loads",
        "8,32",
        "--betas",
        "0.2,0.5,0.8",
        "--policies",
        "recserve,static_hour,broadcast",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ["simulate.py", *extra, "--out", str(Path(tmp) / "check.csv")]
        with contextlib.redirect_stdout(io.StringIO()):
            simulate.main()

    # --- Report ---
    below_top = TIERS[:-1]  # the top tier never escalates
    print(
        f"{'policy':>11} {'max load':>8} {'beta':>4} {'acc':>6} | "
        + " | ".join(f"{t}: ran  escalated  mean conf" for t in below_top)
    )
    for policy, peak, beta, acc, n, s in rows:
        cells = []
        for t in below_top:
            x = s[t]
            esc = x["esc"] / x["judged"] if x["judged"] else float("nan")
            conf = st.mean(x["conf"]) if x["conf"] else float("nan")
            cells.append(f"{x['ran'] / n:6.1%} {esc:9.1%} {conf:9.3f}")
        print(f"{policy:>11} {peak:8.1f} {beta:4.1f} {acc:6.3f} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
