#!/usr/bin/env python3
"""Does skipping tiers disturb RecServe's beta-quantile windows? (energy_tests.md §8.6)

Runs the simulator with its confidence windows instrumented and reports, per policy,
beta and tier: the share of queries the tier ran, how often it escalated what it ran
(RecServe's promise: about beta) and the mean confidence of what it saw (the same as
under RecServe if skipping changes how many queries a tier sees but not which kind).
Extra arguments go to sim_piggyback.py.

    python src/scripts/check_beta_windows.py      # config/study.yaml, surprises x3, loads 8 and 32
"""
import contextlib
import io
import statistics as st
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # implementation/src
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_piggyback as sim  # noqa: E402

TIERS = sim.TIERS
stats: dict = {}
names = iter(())


class Probe(sim.Window):
    """A confidence window that also records what its tier ran and escalated."""

    def __init__(self, n):
        super().__init__(n)
        self.tier, self.thr = next(names), None

    def quantile(self, p):
        self.thr = super().quantile(p)
        return self.thr

    def add(self, x):
        s = stats[self.tier]
        s["ran"] += 1
        s["conf"].append(x)
        if self.thr is not None:                 # the tier judged this answer against its threshold
            s["judged"] += 1
            s["esc"] += x < self.thr
        self.thr = None
        super().add(x)


def main() -> int:
    global names
    rows, real_run = [], sim.run

    def run(rec, stream, batches, loads, beta, policy, *a, **k):
        global names
        stats.clear()
        stats.update({t: {"ran": 0, "judged": 0, "esc": 0, "conf": []} for t in TIERS})
        names = iter(TIERS)                      # run() builds one window per tier, in TIERS order
        m, hr = real_run(rec, stream, batches, loads, beta, policy, *a, **k)
        rows.append((policy, max(loads), beta, m["accuracy"], len(stream), {t: dict(stats[t]) for t in TIERS}))
        return m, hr

    sim.Window, sim.run = Probe, run
    extra = sys.argv[1:] or ["--config", str(sim.CONFIG.parent / "study.yaml"), "--surge-factor", "3",
                             "--peak-loads", "8,32", "--betas", "0.2,0.5,0.8",
                             "--policies", "stepwise,schedule,broadcast"]
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ["sim_piggyback.py", *extra, "--out", str(Path(tmp) / "check.csv")]
        with contextlib.redirect_stdout(io.StringIO()):
            sim.main()

    below_top = TIERS[:-1]                       # the top tier never escalates
    print(f"{'policy':>9} {'max load':>8} {'beta':>4} {'acc':>6} | "
          + " | ".join(f"{t}: ran  escalated  mean conf" for t in below_top))
    for policy, peak, beta, acc, n, s in rows:
        cells = []
        for t in below_top:
            x = s[t]
            esc = x["esc"] / x["judged"] if x["judged"] else float("nan")
            conf = st.mean(x["conf"]) if x["conf"] else float("nan")
            cells.append(f"{x['ran'] / n:6.1%} {esc:9.1%} {conf:9.3f}")
        print(f"{policy:>9} {peak:8.1f} {beta:4.1f} {acc:6.3f} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
