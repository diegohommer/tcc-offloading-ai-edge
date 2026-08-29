#!/usr/bin/env python3
"""Run every tier on every query and record the full answer matrix (JSONL).

This is phase 1 of the energy-policy lambda sweep (design doc section 11).
The sweep itself (src/scripts/sweep_energy_policy.py) is phase 2 and is
pure offline replay -- it never loads a model.

Why a separate matrix instead of reusing a cascade trace: the escalation
decision is path-dependent. Changing lambda changes whether a query
escalates, which changes which tiers it visits, which changes what enters
each tier's confidence history -- and that history is exactly what sets
T(beta) for subsequent queries. A normal cascade trace only records the
tiers a query actually reached, so it cannot answer "what would onu have
said?" for a query that stopped at user. Running every tier once removes
that gap.

This is exact, not an approximation: each tier's pipeline is stateless and
deterministic given the input text, so a tier's answer does not depend on
which path reached it.

Example:
    python src/scripts/run_policy_matrix.py --dataset sst2 --limit 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "recserve" / "vendor"))

from recserve.run_classification_cascade import (  # noqa: E402
    CLOUD_MODEL,
    FOG_MODEL,
    LABEL_MAP,
    ONU_MODEL,
    USER_MODEL,
)
from recserve.traced_recursive_serve import TracedRecursiveServe  # noqa: E402
from utils import load_sentiment_dataset  # noqa: E402  (vendored RecServe module)

TIERS = ("user", "onu", "fog", "cloud")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="sst2",
                        choices=["sst2", "imdb", "rotten_tomatoes", "yelp_polarity", "amazon_polarity"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40, help="cap on number of examples (0 = full split)")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0+ for CUDA device index")
    parser.add_argument("--out", default=None,
                        help="output JSONL path (default: results/traces/<dataset>_<split>.matrix.jsonl)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "results" / "traces" / f"{args.dataset}_{args.split}.matrix.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset}/{args.split} ...")
    dataset = load_sentiment_dataset(args.dataset, args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print("Loading pipelines (user/onu/fog/cloud) ...")
    # beta is irrelevant here -- no escalation decision is made in this phase.
    service = TracedRecursiveServe(USER_MODEL, ONU_MODEL, FOG_MODEL, CLOUD_MODEL, device=args.device)

    per_tier_correct = {tier: 0 for tier in TIERS}
    started = time.perf_counter()

    with open(out_path, "w") as f:
        for i, example in enumerate(dataset):
            true_label = LABEL_MAP[example["label"]]
            tiers = service.classify_all_tiers(example["text"])
            for tier, cell in tiers.items():
                cell["correct"] = cell["predicted_label"] == true_label
                per_tier_correct[tier] += int(cell["correct"])

            f.write(json.dumps({
                "dataset": args.dataset,
                "split": args.split,
                "index": i,
                "true_label": true_label,
                "tiers": tiers,
            }) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(dataset)} processed...")

    n = len(dataset)
    print(f"\nWrote policy matrix: {out_path}")
    print(f"Elapsed: {time.perf_counter() - started:.1f}s for {n} queries x {len(TIERS)} tiers")
    print("\nStandalone accuracy per tier (every tier answering every query):")
    for tier in TIERS:
        print(f"  {tier:>5}: {per_tier_correct[tier] / n:.4f} ({per_tier_correct[tier]}/{n})")
    print("\nNext: python src/scripts/sweep_energy_policy.py " + str(out_path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
