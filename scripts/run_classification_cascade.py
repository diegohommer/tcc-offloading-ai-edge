#!/usr/bin/env python3
"""Run the vendored RecServe classification cascade, extended to the
4-tier user/onu/fog/cloud architecture (Pakpahan and Hwang, IEEE Access
vol. 14, 2026), over a dataset and record a per-query trace (JSONL): which
layers were visited, how many prompt tokens each layer saw, confidence,
latency, and correctness.

This is the "Part 1, functional" half of the section 12 experimental
design: produce a real trace from a real run. The energy pricing of that
trace (Part 2) is a separate step -- see compute_energy_report.py.

Example:
    python scripts/run_classification_cascade.py --dataset sst2 --limit 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "recserve"))

from recserve_trace.traced_recursive_serve import TracedRecursiveServe  # noqa: E402
from utils import load_sentiment_dataset  # noqa: E402  (vendored RecServe module)

# Increasing-capability ladder mirroring the 4-tier reference architecture.
# The cloud-tier model is the exact DeBERTa-large checkpoint RecServe's own
# paper (Section VII-C, Fig. 5) uses as a cloud-side swap-in.
USER_MODEL = "azizbarank/distilroberta-base-sst2-distilled"
ONU_MODEL = "textattack/roberta-base-SST-2"
FOG_MODEL = "howey/roberta-large-sst2"
CLOUD_MODEL = "Tomor0720/deberta-large-finetuned-sst2"

LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sst2",
                         choices=["sst2", "imdb", "rotten_tomatoes", "yelp_polarity", "amazon_polarity"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40, help="cap on number of examples (0 = full split)")
    parser.add_argument("--beta", type=float, default=0.3, help="beta-quantile escalation threshold")
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0+ for CUDA device index")
    parser.add_argument("--out", default=None, help="output JSONL path (default: results/traces/<dataset>_<split>.jsonl)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "results" / "traces" / f"{args.dataset}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset}/{args.split} ...")
    dataset = load_sentiment_dataset(args.dataset, args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print("Loading pipelines (user/onu/fog/cloud) ...")
    service = TracedRecursiveServe(
        USER_MODEL, ONU_MODEL, FOG_MODEL, CLOUD_MODEL, beta=args.beta, device=args.device
    )

    correct = 0
    layer_visits = {"user": 0, "onu": 0, "fog": 0, "cloud": 0}
    layer_final = {"user": 0, "onu": 0, "fog": 0, "cloud": 0}
    total_latency_s = 0.0

    with open(out_path, "w") as f:
        for i, example in enumerate(dataset):
            true_label = LABEL_MAP[example["label"]]
            start = time.perf_counter()
            trace = service.predict(example["text"])
            wall_s = time.perf_counter() - start

            is_correct = trace.final_label == true_label
            correct += int(is_correct)
            for hop in trace.hops:
                layer_visits[hop.layer] += 1
            layer_final[trace.final_layer] += 1
            total_latency_s += wall_s

            record = {
                "dataset": args.dataset,
                "split": args.split,
                "index": i,
                "true_label": true_label,
                "final_label": trace.final_label,
                "final_confidence": trace.final_confidence,
                "correct": is_correct,
                "wall_latency_s": wall_s,
                "hops": [asdict(h) for h in trace.hops],
            }
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(dataset)} processed...")

    n = len(dataset)
    print(f"\nWrote trace: {out_path}")
    print(f"Accuracy: {correct / n:.4f} ({correct}/{n})")
    print(f"Layer hop counts (every hop visited, incl. escalations): {layer_visits}")
    print(f"Final-answer layer distribution: {layer_final}")
    print(f"Avg wall latency/query: {total_latency_s / n:.4f}s")


if __name__ == "__main__":
    main()
