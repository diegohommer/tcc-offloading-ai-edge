#!/usr/bin/env python3
"""Does skipping tiers disturb RecServe's beta-quantile windows? (energy_tests.md §8.6)

ROLE IN THE PIPELINE
    A check on the simulator (simulate/), not a result generator.

WHY
    The energy-aware policies forward some queries past the phone or the ONU. If
    that changed WHICH queries a tier sees, its confidence window (routing.Window)
    would drift and RecServe's threshold would lose its meaning. This runs the
    simulator with the windows instrumented and reports, per policy, beta and
    tier: the share of queries the tier ran, how often it escalated what it ran
    (RecServe's promise: about beta) and the mean confidence of what it saw (the
    same as under recserve if skipping changes how many queries a tier sees but not
    which kind).

Usage:
    python src/analyze/check_beta_windows.py      # config/study.yaml, surprises x3, loads 8 and 32 (~5 min)
    python src/analyze/check_beta_windows.py --config config/study.yaml ...   # any simulate.py arguments
"""
import contextlib
import io
import statistics as st
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                # implementation/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulate"))
import cascade  # noqa: E402
import simulate  # noqa: E402
from energy.three_tier import TIERS  # noqa: E402
from routing import Window  # noqa: E402

stats: dict = {}          # per tier, for the run in progress: ran, judged, escalated, confidences
names = iter(())          # hands each new window its tier name, in TIERS order


class Probe(Window):
    """RecServe's confidence window that also records what its tier ran and escalated."""

    def __init__(self, n):
        """Same as Window; the tier's name comes from `names` (cascade.run builds one window per tier, in order)."""
        super().__init__(n)
        self.tier, self.thr = next(names), None

    def quantile(self, p):
        """Window.quantile, remembering the threshold the next answer is judged against."""
        self.thr = super().quantile(p)
        return self.thr

    def add(self, x):
        """Window.add, first counting whether this answer was judged and escalated."""
        s = stats[self.tier]
        s["ran"] += 1
        s["conf"].append(x)
        if self.thr is not None:                 # the tier judged this answer against its threshold
            s["judged"] += 1
            s["esc"] += x < self.thr
        self.thr = None
        super().add(x)


def main() -> int:
    """Run the simulator with Probe windows and print the per-tier table."""
    global names
    rows, real_run = [], cascade.run

    def run(rec, stream, batches, loads, beta, policy, *a, **k):
        """cascade.run with fresh per-tier counters, keeping them for the table."""
        global names
        stats.clear()
        stats.update({t: {"ran": 0, "judged": 0, "esc": 0, "conf": []} for t in TIERS})
        names = iter(TIERS)
        m, hr = real_run(rec, stream, batches, loads, beta, policy, *a, **k)
        rows.append((policy, max(loads), beta, m["accuracy"], len(stream), {t: dict(stats[t]) for t in TIERS}))
        return m, hr

    cascade.Window, simulate.run = Probe, run
    extra = sys.argv[1:] or ["--config", str(simulate.CONFIG.parent / "study.yaml"), "--surge-factor", "3",
                             "--peak-loads", "8,32", "--betas", "0.2,0.5,0.8",
                             "--policies", "recserve,static_hour,broadcast"]
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ["simulate.py", *extra, "--out", str(Path(tmp) / "check.csv")]
        with contextlib.redirect_stdout(io.StringIO()):
            simulate.main()

    below_top = TIERS[:-1]                       # the top tier never escalates
    print(f"{'policy':>11} {'max load':>8} {'beta':>4} {'acc':>6} | "
          + " | ".join(f"{t}: ran  escalated  mean conf" for t in below_top))
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
