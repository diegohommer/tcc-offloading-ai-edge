"""Comparing policies at equal accuracy.

Part of the simulator (simulate.py). A policy that skips to a more accurate tier
changes accuracy as well as energy, so policies are compared by the energy they
need to reach the same accuracy, read off each policy's sweep over RecServe's beta.
"""
from __future__ import annotations

import numpy as np

def iso_accuracy(rows: list[dict]) -> None:
    """Add, to each row, recserve's J/query at the same accuracy and the saving against it."""
    step = sorted((r["accuracy"], r["J_per_query"]) for r in rows if r["policy"] == "recserve")
    xs, ys = [a for a, _ in step], [j for _, j in step]
    for r in rows:
        a = r["accuracy"]
        if xs[0] <= a <= xs[-1]:
            ref = float(np.interp(a, xs, ys))
            r["J_recserve_same_accuracy"] = ref
            r["saving_same_accuracy"] = 1 - r["J_per_query"] / ref
        else:
            r["J_recserve_same_accuracy"] = float("nan")
            r["saving_same_accuracy"] = float("nan")


def frontier(rows: list[dict], targets: list[float]) -> dict:
    """One policy's J/query at each target accuracy, along its beta sweep.

    Only Pareto points are kept (no other beta is both more accurate and
    cheaper), then J is interpolated between them. It is the cost of reaching
    AT LEAST that accuracy: a target below the policy's range gets its least
    accurate point (it cannot be made less accurate, but does not need to be);
    a target above the range gets None.
    """
    pareto, best = [], float("inf")
    for a, j in sorted(((r["accuracy"], r["J_per_query"]) for r in rows), reverse=True):
        if j < best:
            pareto.append((a, j))
            best = j
    pareto.sort()
    xs, ys = [a for a, _ in pareto], [j for _, j in pareto]
    return {f"{t:.2f}": (None if t > xs[-1] else ys[0] if t < xs[0] else float(np.interp(t, xs, ys)))
            for t in targets}
