#!/usr/bin/env python3
"""Sweep the energy-aware escalation policy: weighting form x lambda.

Phase 2 of the experiment in design doc section 11 (Prof. Nazar, 2026-08-24:
keep both signals, weight the threshold multiplicatively or exponentially,
sweep configurations and measure the impact). Pure offline replay over the
matrix written by run_policy_matrix.py -- no model is loaded here.

THE MECHANISM
-------------
RecServe escalates when a tier's confidence falls below T(beta), the
beta-quantile of that tier's own recent confidence history. This sweep
weights that threshold by the known static energy cost of the hop:

    additive        T_eff = T(beta) - cost/lambda      baseline at lambda -> inf
    multiplicative  T_eff = T(beta) * (1 - lambda*cost)  baseline at lambda = 0
    exponential     T_eff = T(beta) * exp(-lambda*cost)  baseline at lambda = 0

A lower T_eff is easier to clear, so an expensive hop makes a tier keep more
queries locally. Exponential is the proposed default: exp(-lambda*cost) is in
(0,1] for non-negative lambda,cost, so T_eff stays in (0, T(beta)] with no
clamping. Additive needs a clamp and inverts the lambda convention; both are
kept in the sweep because the impact of the form itself is part of what is
being measured.

DECISION COST VS MEASURED ENERGY (deliberately different)
---------------------------------------------------------
The *decision* uses a static per-tier-pair constant, as section 5 requires --
a deployed tier knows its neighbour's typical cost from configuration, not
from live telemetry or from the query it has not run yet:

    decision cost(t -> t') = J_per_token(t') * nominal_tokens + link_J

The *reported* energy uses actual per-query tokens on the tiers actually
visited. That asymmetry is the point: the policy acts on a static estimate,
the evaluation charges it the real bill.

SMOKE-TEST CAVEAT
-----------------
Same caveat as compute_energy_report.py, and for the same reason: these
classifiers do a single forward pass and generate nothing, while every
J/token value in layer_energy.yaml was measured on decoder LLMs doing
multi-token decode. Prompt tokens are priced as if they were decode tokens.
The *relative* comparison between policy configurations is the result here;
the absolute joules are not a thesis number.

Example:
    python src/scripts/sweep_energy_policy.py results/traces/sst2_test.matrix.jsonl
    python src/scripts/sweep_energy_policy.py <matrix> --cost-config batched-cloud
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from energy.layer_energy import LayerEnergyTable  # noqa: E402

TIERS = ("user", "onu", "fog", "cloud")
NEXT_TIER = {"user": "onu", "onu": "fog", "fog": "cloud"}
FORMS = ("additive", "multiplicative", "exponential")

# Lambda grids per form. Additive's lambda is J per confidence-point and its
# baseline is lambda -> inf, so it sweeps upward to a large value; the other
# two are 1/J with an exact baseline at 0.
DEFAULT_LAMBDAS = {
    "additive": [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1e6],
    "multiplicative": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "exponential": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
}


def cost_configs(table: LayerEnergyTable) -> dict[str, dict[str, float]]:
    """Per-tier J/token, one dict per rung of the section 11.3 complexity ladder.

    Every value is a real measured row from layer_energy.yaml -- the ladder
    swaps which row is selected, it never fabricates a number.
    """
    onu_15b = table.decode_point("onu", "qwen2.5_1.5b", "w4").decode_J_per_token
    onu_14b = table.decode_point("onu", "qwen2.5_14b", "w4").decode_J_per_token
    user_1b = table.decode_point("user", "llama3.2_1b", "cpu").decode_J_per_token
    fog_prod = table.fog_primary_J_per_token()
    cloud_prod = table.cloud_production_J_per_token()
    cloud_mlperf = float(table.layers["cloud"]["mlperf_power_regime"]["offline"]["decode_J_per_token"])
    onu_8b = table.decode_point("onu", "llama3.1_8b", "w4").decode_J_per_token

    # QwQ-32B on 1xH100 at both batching extremes (Caravaca et al., Table IV).
    # Flagged SECONDARY in layer_energy.yaml as below the 65B+ cloud target -- a
    # documented deviation, used because it is the model a fully-local generative
    # ladder can actually run, and because it is measured at BOTH batch=1 and the
    # paper's best-found batching config.
    def _qwq(batch_key) -> float:
        for entry in table.layers["cloud"]["batch_curve"]:
            if entry["model"].startswith("QwQ-32B") and str(entry["batch"]).startswith(str(batch_key)):
                return float(entry["decode_J_per_token"])
        raise KeyError(f"QwQ-32B batch={batch_key} not found in cloud batch_curve")

    cloud_32b_batch1 = _qwq(1)
    cloud_32b_opt = _qwq("optimized")

    return {
        # Rung 1: monotonically increasing, as directed. Achievable with real
        # numbers by selecting the cheapest ONU config and the BF16 (not FP4)
        # cloud point.
        "monotonic": {"user": user_1b, "onu": onu_15b, "fog": fog_prod, "cloud": cloud_prod},
        # Rung 2: the batched-cloud point undercuts fog -- escalating to cloud
        # becomes cheaper per token than staying at fog.
        "batched-cloud": {"user": user_1b, "onu": onu_15b, "fog": fog_prod, "cloud": cloud_mlperf},
        # Rung 3: an oversized ONU model costs more than the tier above it.
        "oversized-onu": {"user": user_1b, "onu": onu_14b, "fog": fog_prod, "cloud": cloud_prod},

        # --- Configs for the GENERATIVE ladder (run_generative_matrix.py) ---
        # The same four models the cascade actually runs, each priced with the
        # published energy for THAT model on hardware of its tier's class. This
        # is the pairing the classification track could never have: confidence
        # and energy finally refer to the same models.
        #
        # The cloud tier is deliberately split into two states rather than
        # averaged. Caravaca et al. measured QwQ-32B on 1xH100 at both extremes
        # and the gap is 30x -- unbatched cloud costs MORE per token than fog,
        # batched cloud far less. One averaged value would erase exactly the
        # inversion that section 11.3 rung 2 exists to test.
        "gen-cloud-idle": {
            "user": user_1b, "onu": onu_8b, "fog": fog_prod, "cloud": cloud_32b_batch1,
        },
        "gen-cloud-batched": {
            "user": user_1b, "onu": onu_8b, "fog": fog_prod, "cloud": cloud_32b_opt,
        },
    }


def billable_tokens(cell: dict) -> int:
    """Tokens the energy model actually prices for one tier visit.

    Every J/token value in layer_energy.yaml is DECODE energy -- joules per
    OUTPUT token. A generative matrix records tokens_gen, so that is what gets
    priced. The classification matrix has no generation at all (tokens_gen
    absent), and falls back to prompt tokens -- which is exactly the labelled
    smoke-test proxy documented in compute_energy_report.py, not a real
    decode measurement.
    """
    if cell.get("tokens_gen") is not None:
        return int(cell["tokens_gen"])
    return int(cell["tokens_prompt"])


def apply_weight(threshold: float, cost: float, lam: float, form: str) -> float:
    """Weight T(beta) by the static hop cost. Returns the effective threshold."""
    if form == "additive":
        if lam <= 0:
            return float("-inf")  # energy infinitely expensive -> never escalate
        return threshold - cost / lam
    if form == "multiplicative":
        return max(0.0, threshold * (1.0 - lam * cost))
    if form == "exponential":
        return threshold * math.exp(-lam * cost)
    raise ValueError(f"unknown weighting form: {form}")


def replay(records, beta, form, lam, decision_costs, j_per_token, link_j, max_history=10000):
    """Re-run RecServe's escalation loop offline under one (form, lambda) config.

    Mirrors TracedRecursiveServe.predict exactly -- same threshold rule, same
    `len(history) > 1` guard, same append-after-decide ordering -- so that the
    baseline configuration reproduces the real cascade run bit for bit.
    """
    history = {tier: [] for tier in TIERS}
    n_correct = 0
    total_energy_j = 0.0
    total_hops = 0
    visits = {tier: 0 for tier in TIERS}
    finals = {tier: 0 for tier in TIERS}
    escalations = {tier: 0 for tier in TIERS}

    for rec in records:
        tier = "user"
        visited = []
        while True:
            cell = rec["tiers"][tier]
            confidence = cell["confidence"]
            visited.append(tier)
            visits[tier] += 1

            next_tier = NEXT_TIER.get(tier)
            escalate = False
            hist = history[tier]
            if next_tier is not None and len(hist) > 1:
                threshold = float(np.percentile(hist, beta * 100))
                t_eff = apply_weight(threshold, decision_costs[(tier, next_tier)], lam, form)
                if confidence < t_eff:
                    escalate = True
                    escalations[tier] += 1

            hist.append(confidence)
            if len(hist) > max_history:
                hist.pop(0)

            if escalate:
                tier = next_tier
                continue
            break

        final_tier = visited[-1]
        finals[final_tier] += 1
        n_correct += int(rec["tiers"][final_tier]["correct"])
        total_hops += len(visited)
        total_energy_j += sum(j_per_token[t] * billable_tokens(rec["tiers"][t]) for t in visited)
        total_energy_j += link_j * (len(visited) - 1)

    n = len(records)
    return {
        "n": n,
        "accuracy": n_correct / n,
        "total_energy_J": total_energy_j,
        "energy_J_per_query": total_energy_j / n,
        "avg_hops": total_hops / n,
        **{f"visits_{t}": visits[t] for t in TIERS},
        **{f"final_{t}": finals[t] for t in TIERS},
        **{f"esc_rate_{t}": (escalations[t] / visits[t] if visits[t] else 0.0) for t in TIERS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrix_path", type=Path, help="JSONL from run_policy_matrix.py")
    parser.add_argument("--beta", type=float, default=0.3, help="beta-quantile (must match the cascade run)")
    parser.add_argument("--cost-config", default="monotonic",
                        help="which rung of the section 11.3 ladder to price with (default: monotonic)")
    parser.add_argument("--measured-energy", type=Path, default=None,
                        help="JSON from measure_tier_energy.py. Replaces the borrowed layer_energy.yaml "
                             "values with energy actually measured on this machine for the models that "
                             "actually run. Overrides --cost-config.")
    parser.add_argument("--nominal-tokens", type=float, default=None,
                        help="tokens assumed when computing the STATIC decision cost "
                             "(default: mean prompt tokens across the matrix)")
    parser.add_argument("--forms", nargs="+", default=list(FORMS), choices=FORMS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=None,
                        help="override the per-form lambda grid with one shared grid")
    parser.add_argument("--out", default=None, help="output CSV (default: alongside the matrix, .sweep.csv)")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.matrix_path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"no records in {args.matrix_path}")

    table = LayerEnergyTable()
    link_j = table.link_energy_per_hop_J()

    if args.measured_energy is not None:
        measured = json.loads(args.measured_energy.read_text())
        j_per_token = {t: float(measured["tiers"][t]["net_J_per_prompt_token"]) for t in TIERS}
        cost_label = f"MEASURED on this machine ({args.measured_energy.name})"
        # The link constant comes from PON literature (J per hop) and is unrelated
        # to the locally measured CPU figures; at these magnitudes it would
        # dominate every hop decision and drown out the measured signal.
        link_j = 0.0
    else:
        configs = cost_configs(table)
        if args.cost_config not in configs:
            raise SystemExit(f"unknown --cost-config {args.cost_config!r}; choose from {sorted(configs)}")
        j_per_token = configs[args.cost_config]
        cost_label = args.cost_config

    nominal_tokens = args.nominal_tokens
    if nominal_tokens is None:
        nominal_tokens = float(np.mean([billable_tokens(r["tiers"]["user"]) for r in records]))

    # Static per-tier-pair decision cost: what a tier is told, at deployment
    # time, that escalating will typically cost.
    decision_costs = {
        (t, nxt): j_per_token[nxt] * nominal_tokens + link_j
        for t, nxt in NEXT_TIER.items()
    }

    print(f"Matrix: {args.matrix_path}  (n={len(records)})")
    print(f"Cost config: {cost_label}  ->  " +
          "  ".join(f"{t}={j_per_token[t]:.5f}" for t in TIERS) + " J/token")
    if args.measured_energy is not None:
        print("  (idle-subtracted RAPL package energy per PROMPT token, CPU, this machine;")
        print("   link term set to 0 -- the PON per-hop constant is from unrelated literature)")
    print(f"Nominal tokens for the static decision cost: {nominal_tokens:.1f}")
    print("Static decision cost per hop: " +
          "  ".join(f"{t}->{nxt}={decision_costs[(t, nxt)]:.3f}J" for t, nxt in NEXT_TIER.items()))
    print()

    rows = []
    # Baseline: plain RecServe, energy ignored entirely. Everything else is
    # measured against this, and the exponential/multiplicative lambda=0 rows
    # must reproduce it exactly (checked below).
    baseline = replay(records, args.beta, "exponential", 0.0, decision_costs, j_per_token, link_j)
    rows.append({"form": "baseline", "lambda": 0.0, **baseline})

    for form in args.forms:
        grid = args.lambdas if args.lambdas is not None else DEFAULT_LAMBDAS[form]
        for lam in grid:
            rows.append({"form": form, "lambda": lam,
                         **replay(records, args.beta, form, lam, decision_costs, j_per_token, link_j)})

    out_path = Path(args.out) if args.out else args.matrix_path.with_suffix(".sweep.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'form':<16}{'lambda':>10}{'accuracy':>10}{'J/query':>10}{'hops':>7}   final-tier distribution"
    print(header)
    print("-" * len(header))
    for row in rows:
        dist = "/".join(str(row[f"final_{t}"]) for t in TIERS)
        print(f"{row['form']:<16}{row['lambda']:>10.4g}{row['accuracy']:>10.4f}"
              f"{row['energy_J_per_query']:>10.3f}{row['avg_hops']:>7.2f}   {dist}")
    print(f"\n(final-tier distribution is {'/'.join(TIERS)})")

    # Correctness check: the two bounded forms must collapse onto the baseline
    # at lambda=0. If they do not, the replay does not faithfully reproduce
    # RecServe's rule and nothing downstream is trustworthy.
    problems = []
    for form in ("multiplicative", "exponential"):
        if form not in args.forms:
            continue
        zero_rows = [r for r in rows if r["form"] == form and r["lambda"] == 0.0]
        if not zero_rows:
            continue
        got = zero_rows[0]
        if abs(got["accuracy"] - baseline["accuracy"]) > 1e-12 or \
           abs(got["energy_J_per_query"] - baseline["energy_J_per_query"]) > 1e-9:
            problems.append(f"{form} at lambda=0 does not match the baseline")
    print("\nBaseline check: " + ("OK -- lambda=0 reproduces plain RecServe exactly"
                                  if not problems else "FAILED -- " + "; ".join(problems)))

    print(f"\nWrote sweep: {out_path}")
    if args.measured_energy is not None:
        print("NOTE: joules here were measured on this machine for the models that actually ran")
        print("      (RAPL, CPU, per prompt token) -- not the borrowed decode-phase proxy. They are")
        print("      real for THIS 4-tier classifier cascade, and say nothing about a generative one.")
    else:
        print("NOTE: absolute joules are the labeled smoke-test proxy (see module docstring);")
        print("      the comparison *between* configurations is what this sweep measures.")
        print("      Use --measured-energy to price with locally measured values instead.")


if __name__ == "__main__":
    main()
