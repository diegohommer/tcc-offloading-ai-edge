#!/usr/bin/env python3
"""Check that a collected answers file's confidence scores are sound before anything routes on them.

Per tier: every generated token has one logprob, all <= 0; the stored confidence equals
exp(mean token logprob), RecServe's; how many answers hit the token limit; and how well
confidence separates correct from wrong answers (Welch t, for exp(mean) and exp(min)).

Usage: python src/analyze/check_confidence.py results/measurements/<answers>.raw.jsonl
"""

from __future__ import annotations

import collections
import json
import math
import statistics as st
import sys
from pathlib import Path


def welch_t(a: list[float], b: list[float]) -> float | None:
    """Return Welch's t for mean(a) - mean(b).

    Args:
        a: The first group's values.
        b: The second group's values.

    Returns:
        Welch's t, or None when either group is too small.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = st.variance(a), st.variance(b)
    denom = math.sqrt(va / len(a) + vb / len(b))
    return (st.mean(a) - st.mean(b)) / denom if denom > 0 else None


def format_t(t: float | None) -> str:
    """Format a Welch t for the report."""
    return "n/a (too few in a group)" if t is None else f"{t:+.2f}"


def main() -> int:
    """Check every tier in the answers file given on the command line.

    Returns:
        The process exit code: 0 if every check passes, 1 if any fails, 2 on bad usage.
    """
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    by_tier: dict[str, list[dict]] = collections.defaultdict(list)
    with open(Path(sys.argv[1]), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_tier[rec["tier"]].append(rec)

    failed = False
    for tier in [t for t in ("user", "onu", "olt", "fog", "cloud") if t in by_tier] + [
        t for t in by_tier if t not in ("user", "onu", "olt", "fog", "cloud")
    ]:
        # --- Integrity and formula, per answer ---
        recs = by_tier[tier]
        problems = collections.Counter()
        conf_mean = {True: [], False: []}
        conf_min = {True: [], False: []}
        for r in recs:
            lps = r.get("logprobs") or []
            if len(lps) != r["tokens_gen"]:
                problems["logprob count != tokens_gen"] += 1
            if any(x is None for x in lps):
                problems["missing logprob"] += 1
            valid = [x for x in lps if x is not None]
            if any(x > 1e-6 for x in valid):
                problems["logprob > 0"] += 1
            if not valid:
                problems["no logprobs at all"] += 1
                continue
            recomputed = math.exp(sum(valid) / len(valid))
            if abs(recomputed - r["confidence"]) > 1e-9:
                problems["stored confidence != exp(mean logprob)"] += 1
            if not 0.0 < r["confidence"] <= 1.0:
                problems["confidence outside (0, 1]"] += 1
            conf_mean[bool(r["correct"])].append(recomputed)
            conf_min[bool(r["correct"])].append(math.exp(min(valid)))

        # --- Report ---
        n = len(recs)
        truncated = sum(r.get("finish_reason") == "length" for r in recs)
        acc = sum(bool(r["correct"]) for r in recs) / n
        t_mean = welch_t(conf_mean[True], conf_mean[False])
        t_min = welch_t(conf_min[True], conf_min[False])

        print(
            f"\n[{tier}] {recs[0].get('model', '?')}  n={n}  accuracy={acc:.3f}  "
            f"truncated at token limit: {truncated}"
        )
        # Report every check on its own line, so one failing check never hides
        # the status of the others.
        checks = [
            ("count", "logprob count != tokens_gen", "one logprob per generated token"),
            ("missing", "missing logprob", "no logprob missing"),
            ("sign", "logprob > 0", "all logprobs <= 0"),
            ("empty", "no logprobs at all", "every record has logprobs"),
            (
                "formula",
                "stored confidence != exp(mean logprob)",
                "stored confidence = exp(mean token logprob)",
            ),
            ("range", "confidence outside (0, 1]", "confidence within (0, 1]"),
        ]
        for name, problem, ok_text in checks:
            if problems[problem]:
                failed = True
                print(f"  FAIL  {name:8} {problem}: {problems[problem]} of {n}")
            else:
                print(f"  ok    {name:8} {ok_text}")
        gaps = collections.Counter(len(r.get("logprobs") or []) - r["tokens_gen"] for r in recs)
        if set(gaps) - {0}:
            print(f"        len(logprobs) - tokens_gen, by count: {dict(sorted(gaps.items()))}")
        for label, groups in (("exp(mean logprob)", conf_mean), ("exp(min logprob) ", conf_min)):
            ok, bad = groups[True], groups[False]
            mean_ok = f"{st.mean(ok):.4f}" if ok else "  -   "
            mean_bad = f"{st.mean(bad):.4f}" if bad else "  -   "
            t = t_mean if label.startswith("exp(mean") else t_min
            print(
                f"  signal     {label}: correct {mean_ok} vs wrong {mean_bad}  Welch t = {format_t(t)}"
            )

    print(
        "\nALL CHECKS PASSED"
        if not failed
        else "\nSOME CHECKS FAILED -- fix before routing on this trace"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
