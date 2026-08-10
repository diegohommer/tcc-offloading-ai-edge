#!/usr/bin/env python3
"""Convert a trace produced by run_classification_cascade.py into an energy
report using config/layer_energy.yaml (section 5's derivation formulas).

IMPORTANT CAVEAT -- read before trusting any number this prints:
RecServe's shipped cascade classifies (single forward pass, no generation),
while every entry in config/layer_energy.yaml was measured on decoder LLMs
doing multi-token decode (0.5B-70B+ class models) against classifiers that
are 66M-355M params. There is no published energy measurement for
distilroberta/roberta-base/roberta-large at all. So this script does NOT
invent one. What it reports:

  1. Real, measured metrics from the actual run: latency, tokens processed,
     escalation path, accuracy. These are trustworthy.
  2. An OPTIONAL, clearly-labeled energy proxy (--smoke-test-energy) that
     maps RecServe's end/edge/cloud tiers onto three of the four energy
     layers (user/onu/fog) and prices each hop's prompt tokens as if they
     were decode tokens on that layer's representative model. This exists
     only to exercise the section-5 arithmetic end to end before the real
     generative cascade exists -- it is explicitly NOT a thesis result, and
     the report says so every time it's used.

Example:
    python scripts/compute_energy_report.py results/traces/sst2_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy.cost import LayerVisit, aggregate_J_per_token, cascade_cost_J  # noqa: E402
from energy.layer_energy import LayerEnergyTable  # noqa: E402

# RecServe tier -> (energy layer, representative model, precision) used ONLY
# for the smoke-test energy proxy. Picked as the smallest / most-local model
# tabulated for each layer, since RecServe's classifiers are themselves tiny
# relative to what these tables measure.
TIER_TO_LAYER = {
    "end": ("user", "llama3.2_1b", "cpu"),
    "edge": ("onu", "qwen2.5_1.5b", "w4"),
    "cloud": ("fog", None, None),  # fog uses the primary A30 point directly, no per-model table
}


def smoke_test_energy_for_hop(energy_table: LayerEnergyTable, tier: str, tokens_prompt: int) -> float:
    layer, model_key, precision = TIER_TO_LAYER[tier]
    if layer == "fog":
        e_dec = energy_table.fog_primary_J_per_token()
    else:
        e_dec = energy_table.decode_point(layer, model_key, precision).decode_J_per_token
    # Proxy: treat the classifier's single forward pass over the prompt as if
    # every prompt token were a decode step on the mapped layer's model. This
    # is an admitted approximation solely to exercise the cost formula.
    visit = LayerVisit(layer=layer, tokens_prompt=tokens_prompt, tokens_gen=tokens_prompt, E_dec_J_per_token=e_dec)
    return visit.E_dec_J_per_token * visit.tokens_gen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_path", type=Path)
    parser.add_argument("--smoke-test-energy", action="store_true",
                         help="also compute the labeled energy proxy described above")
    parser.add_argument("--out", default=None, help="output CSV path (default: alongside the trace, .energy.csv)")
    args = parser.parse_args()

    energy_table = LayerEnergyTable()
    records = [json.loads(line) for line in args.trace_path.read_text().splitlines() if line.strip()]

    out_path = Path(args.out) if args.out else args.trace_path.with_suffix(".energy.csv")
    rows = []
    for rec in records:
        hops = rec["hops"]
        tokens_prompt_total = sum(h["tokens_prompt"] for h in hops)
        latency_s_total = sum(h["latency_s"] for h in hops)
        n_hops = len(hops)
        n_link_hops = max(0, n_hops - 1)  # escalations cross a link; the first hop doesn't
        row = {
            "index": rec["index"],
            "final_layer": hops[-1]["layer"],
            "n_hops": n_hops,
            "tokens_prompt_total": tokens_prompt_total,
            "latency_s_total": latency_s_total,
            "correct": rec["correct"],
        }
        if args.smoke_test_energy:
            proxy_energies = [smoke_test_energy_for_hop(energy_table, h["layer"], h["tokens_prompt"]) for h in hops]
            compute_j = sum(proxy_energies)
            link_j = n_link_hops * energy_table.link_energy_per_hop_J()
            row["smoke_test_compute_J"] = compute_j
            row["smoke_test_link_J"] = link_j
            row["smoke_test_total_J"] = compute_j + link_j
            row["smoke_test_J_per_token"] = aggregate_J_per_token(compute_j, tokens_prompt_total, 0)
        rows.append(row)

    import csv

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    accuracy = sum(r["correct"] for r in rows) / n
    avg_hops = sum(r["n_hops"] for r in rows) / n
    avg_latency = sum(r["latency_s_total"] for r in rows) / n
    print(f"Wrote energy report: {out_path}")
    print(f"n={n}  accuracy={accuracy:.4f}  avg_hops={avg_hops:.2f}  avg_latency_s={avg_latency:.4f}")
    if args.smoke_test_energy:
        avg_total_j = sum(r["smoke_test_total_J"] for r in rows) / n
        print(f"avg SMOKE-TEST total_J={avg_total_j:.4f}  <-- NOT a thesis result, see script docstring caveat")
    else:
        print("Run with --smoke-test-energy to also emit the labeled energy-formula smoke test.")


if __name__ == "__main__":
    main()
