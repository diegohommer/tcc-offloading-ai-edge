#!/usr/bin/env python3
"""Three offline tests over the SST-2 policy matrix. No GPU, no network, no billing.

Everything here is pure replay against results/traces/sst2_test.matrix.jsonl,
which run_policy_matrix.py produces on CPU from four locally cached classifiers.
No model is loaded. That matters because the generative track is currently
blocked -- the GSM8K traces were lost to .gitignore and the rented-GPU route ran
out of free credit -- and these three questions do not need it.

TEST 1 -- IS BETA A QUALITY BAR, OR A TRAFFIC QUOTA?
------------------------------------------------------
RecServe escalates when confidence falls below T(beta), the beta-quantile of the
tier's OWN confidence history. A quantile of your own history is self-
normalizing: whatever the confidence distribution looks like, the bottom beta of
it falls below the beta-quantile. So in steady state each tier should escalate
almost exactly beta of its traffic, and the fraction reaching tier k should be
beta^k -- regardless of the workload.

If that holds, beta is not a quality bar at all, it is a traffic split, and the
whole family of "weight the threshold by X" proposals collapses: multiplying a
quantile by any query-independent factor just selects a different quantile, i.e.
another beta. That is the algebraic reason the energy-weighting result in the
design doc's section 14 came out redundant, stated as a property of the rule
rather than as an experimental accident.

TEST 2 -- THE HISTORY WINDOW IS NEVER ABLATED
----------------------------------------------
RecServe ships max_history_size=10000 and reports no sensitivity analysis. A
long window means the quantile is computed over stale confidences. On a
stationary stream that is harmless. On a stream whose difficulty shifts, the
threshold is calibrated to a workload that is no longer arriving -- so the tier
over- or under-escalates for as long as the stale samples dominate the window.
Both errors cost: over-escalation burns energy, under-escalation loses accuracy.

The non-stationary stream here is built by ORDERING the same queries, easiest
first (by the user tier's own confidence). No query is invented or dropped; only
the arrival order changes, which is exactly the variable a deployed cascade does
not control.

TEST 3 -- STEPWISE VS. CHOOSING A DESTINATION
----------------------------------------------
RecServe fuses two decisions into one: "escalate?" and "escalate to where?",
answering the second with "always the next tier up". Test 1's result says beta
already owns the first decision completely. The second is a discrete choice over
targets that beta cannot express at all -- there is no value of beta that means
"skip the fog tier".

That asymmetry is the point. A query ending at cloud under stepwise recursion has
paid for user + onu + fog + cloud, and the tiers it traversed contributed nothing
to the answer -- they were wrong, which is why it escalated. On the RAPL-measured
ladder in config/layer_energy.yaml that traversal is 4.588 J against 2.452 J for
going straight there: 47% of the energy is spent on tiers whose answers were
discarded.

Three destination rules are compared, and the calibrated one is the interesting
case -- it is fitted on half the queries and evaluated on the other half, so it
cannot cheat.

Run:
    python src/scripts/run_policy_matrix.py --dataset sst2 --limit 0 --device -1
    python src/scripts/replay_routing.py results/traces/sst2_test.matrix.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TIERS = ("user", "onu", "fog", "cloud")
NEXT_TIER = {"user": "onu", "onu": "fog", "fog": "cloud"}

# Joules per inference, measured on THIS machine via Intel RAPL and recorded in
# config/layer_energy.yaml's local_measurement block. The 'net' figures (idle
# power subtracted) are used; the file records 'total' as well and flags that
# neither is definitive. The ladder is monotonic, ~10x user to cloud.
ENERGY_J = {"user": 0.2195, "onu": 0.3899, "fog": 1.7465, "cloud": 2.2323}


def load_matrix(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile, matching numpy.percentile's default method.

    Reimplemented rather than imported so this script has no dependency beyond
    the standard library -- it must stay runnable when nothing else is.
    """
    if not sorted_vals:
        return float("-inf")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def replay(rows: list[dict], beta: float, window: int, destination: str | None = None) -> dict:
    """Replay the cascade once. destination=None is RecServe's stepwise recursion.

    With a destination, the escalation DECISION is identical -- same rule, same
    beta, same history -- and only the target changes: an escalating query jumps
    straight to `destination` instead of walking there one tier at a time. That
    isolates the routing choice from the escalation choice, so any difference is
    attributable to the destination alone.
    """
    history: dict[str, list[float]] = {t: [] for t in TIERS}
    reached = {t: 0 for t in TIERS}
    escalated_from = {t: 0 for t in TIERS}
    decided_at = {t: 0 for t in TIERS}
    n_correct = 0
    energy_total = 0.0
    final_tiers = {t: 0 for t in TIERS}
    # Entry-tier decisions in arrival order, so escalation rate can be measured
    # over segments of one continuous run rather than by restarting the history.
    entry_decisions: list[int] = []

    for row in rows:
        tier = "user"
        while True:
            reached[tier] += 1
            energy_total += ENERGY_J[tier]
            cell = row["tiers"][tier]
            confidence = cell["confidence"]

            nxt = NEXT_TIER.get(tier)
            escalate = False
            if nxt is not None and len(history[tier]) > 1:
                # RecServe's rule verbatim: beta-quantile of this tier's own
                # recent confidences, escalate if below it.
                threshold = quantile(sorted(history[tier]), beta)
                escalate = confidence < threshold

            history[tier].append(confidence)
            if len(history[tier]) > window:
                history[tier].pop(0)

            if nxt is not None:
                decided_at[tier] += 1
                escalated_from[tier] += int(escalate)
                if tier == "user":
                    entry_decisions.append(int(escalate))

            if escalate:
                if destination is None:
                    tier = nxt
                else:
                    # Jump straight to the destination, provided it is above the
                    # current tier; otherwise fall back to the ordinary next hop
                    # so the query still makes progress.
                    di, ci = TIERS.index(destination), TIERS.index(tier)
                    tier = destination if di > ci else nxt
                continue

            n_correct += int(cell["correct"])
            final_tiers[tier] += 1
            break

    n = len(rows)
    return {
        "beta": beta,
        "window": window,
        "destination": destination or "stepwise",
        "accuracy": n_correct / n,
        "J_per_query": energy_total / n,
        "escalation_rate": {t: escalated_from[t] / decided_at[t] if decided_at[t] else 0.0
                            for t in TIERS if t in NEXT_TIER},
        "reach_fraction": {t: reached[t] / n for t in TIERS},
        "final_tier_share": {t: final_tiers[t] / n for t in TIERS},
        "entry_decisions": entry_decisions,
    }


def test1_beta_is_a_quota(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("TEST 1 -- is beta a quality bar, or a traffic quota?")
    print("=" * 78)
    print("If beta is a quota, escalation rate ~ beta at every tier, and the")
    print("fraction reaching tier k ~ beta^k, independent of the data.\n")
    print(f"{'beta':>6} {'esc user':>9} {'esc onu':>9} {'esc fog':>9} "
          f"{'reach onu':>10} {'pred b^1':>9} {'reach fog':>10} {'pred b^2':>9} "
          f"{'reach cld':>10} {'pred b^3':>9}")
    for beta in (0.1, 0.2, 0.3, 0.5):
        r = replay(rows, beta, window=10000)
        e, f = r["escalation_rate"], r["reach_fraction"]
        print(f"{beta:>6.2f} {e['user']:>9.3f} {e['onu']:>9.3f} {e['fog']:>9.3f} "
              f"{f['onu']:>10.4f} {beta:>9.4f} {f['fog']:>10.4f} {beta**2:>9.4f} "
              f"{f['cloud']:>10.4f} {beta**3:>9.4f}")


def test2_window_ablation(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("TEST 2 -- history window, on a stream whose difficulty shifts")
    print("=" * 78)
    print("Two regimes: the confidence-wise easier half arrives first, then the")
    print("harder half. Queries are SHUFFLED WITHIN each regime, so the stream is")
    print("stationary inside a regime and shifts once, in the middle -- rather")
    print("than being a monotonic ramp, which would trivially escalate everything")
    print("regardless of window and prove nothing.")
    print("Only arrival order changes. Target escalation rate is beta = 0.30.\n")

    ranked = sorted(rows, key=lambda r: -r["tiers"]["user"]["confidence"])
    half = len(ranked) // 2
    rng = random.Random(0)
    easy, hard = ranked[:half], ranked[half:]
    rng.shuffle(easy)
    rng.shuffle(hard)
    stream = easy + hard

    def rate(decisions: list[int]) -> float:
        return sum(decisions) / len(decisions) if decisions else 0.0

    print(f"{'window':>8} {'before shift':>13} {'just after':>11} {'settled':>9} "
          f"{'overall':>8} {'accuracy':>9} {'J/query':>9}")
    for window in (10000, 500, 200, 50, 20):
        r = replay(stream, 0.30, window)
        d = r["entry_decisions"]
        # Segments of ONE run: the history carried across the shift is the real one.
        seg = max(1, len(d) // 10)
        print(f"{window:>8} {rate(d[half - seg:half]):>13.3f} "
              f"{rate(d[half:half + seg]):>11.3f} "
              f"{rate(d[half + 2 * seg:]):>9.3f} {rate(d):>8.3f} "
              f"{r['accuracy']:>9.4f} {r['J_per_query']:>9.4f}")
    print("\n(RecServe ships window=10000 -- the top row -- and never ablates it.)")


def calibrate_destination(train: list[dict]) -> tuple[str, dict[str, float]]:
    """Cheapest tier that is statistically as accurate as the best one.

    Fitted on the calibration half only. 'As accurate as' uses a one-standard-
    error slack so a tier is not preferred over a cheaper one on noise alone.
    """
    acc, n = {}, len(train)
    for t in TIERS:
        acc[t] = sum(r["tiers"][t]["correct"] for r in train) / n
    best = max(acc.values())
    se = (best * (1 - best) / n) ** 0.5
    viable = [t for t in TIERS if acc[t] >= best - se]
    return min(viable, key=lambda t: ENERGY_J[t]), acc


def test3_stepwise_vs_destination(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("TEST 3 -- stepwise recursion vs. choosing a destination")
    print("=" * 78)

    half = len(rows) // 2
    train, evaluate = rows[:half], rows[half:]
    chosen, acc = calibrate_destination(train)
    print("Per-tier standalone accuracy on the calibration half:")
    for t in TIERS:
        print(f"    {t:>6} {acc[t]:.4f}   ({ENERGY_J[t]:.4f} J/inference)")
    print(f"  -> calibrated destination: {chosen}  "
          f"(cheapest tier within one standard error of the best)\n")

    print("Evaluated on the held-out half. Same beta, same rule, same history --")
    print("only the destination of an escalating query differs.\n")
    print(f"{'beta':>6}  {'routing':>22} {'accuracy':>9} {'J/query':>9} {'vs stepwise':>12}")
    for beta in (0.1, 0.2, 0.3, 0.5):
        base = replay(evaluate, beta, 10000, destination=None)
        print(f"{beta:>6.2f}  {'stepwise (RecServe)':>22} "
              f"{base['accuracy']:>9.4f} {base['J_per_query']:>9.4f} {'--':>12}")
        for dest, label in ((chosen, f"jump to {chosen} (calib.)"), ("cloud", "jump to cloud (top)")):
            r = replay(evaluate, beta, 10000, destination=dest)
            ratio = base["J_per_query"] / r["J_per_query"]
            print(f"{'':>6}  {label:>22} {r['accuracy']:>9.4f} {r['J_per_query']:>9.4f} "
                  f"{ratio:>11.2f}x")
        print()


def test4_when_does_skipping_pay(rows: list[dict]) -> None:
    """The criterion, derived from Test 1's quota law, applied to two real ladders.

    Test 1 established that tier k receives beta^k of the traffic. So per query:

        stepwise  = E_user + b*E_onu + b^2*E_fog + b^3*E_cloud
        skip to d = E_user + b*E_d

    and skipping is cheaper exactly when

        E_d  <  E_onu + b*E_fog + b^2*E_cloud

    i.e. the destination must undercut the NEXT tier plus the discounted tail of
    the chain -- not the full traversal. Comparing against the full traversal is
    the intuitive move and it is wrong, because beta^k means almost nothing
    completes the full traversal: at beta=0.3 only 2.7% of queries reach cloud.
    """
    print("\n" + "=" * 78)
    print("TEST 4 -- when does skipping pay? (the criterion, on two real ladders)")
    print("=" * 78)
    print("Skipping to d is cheaper than stepwise iff  E_d < E_onu + b*E_fog + b^2*E_cloud\n")

    ladders = {
        "SST-2 classifiers, this CPU (RAPL, J/inference)": ENERGY_J,
        "generative/PON tables (J/query @200 tok, layer_energy.yaml)":
            {"user": 14.8, "onu": 196.0, "fog": 75.7, "cloud": 18.76},
    }
    for name, E in ladders.items():
        monotonic = E["user"] <= E["onu"] <= E["fog"] <= E["cloud"]
        print(f"{name}")
        print(f"    ladder {[round(E[t], 3) for t in TIERS]}  monotonic={monotonic}")
        for b in (0.1, 0.3, 0.5):
            budget = E["onu"] + b * E["fog"] + b * b * E["cloud"]
            winners = [t for t in ("fog", "cloud") if E[t] < budget]
            print(f"    beta={b:.1f}: budget {budget:8.3f} J  ->  "
                  f"skipping pays for {winners or 'no destination'}")
        print()
    print("Reading: a monotonic ladder leaves nothing worth skipping toward -- the")
    print("cheapest tier above you IS the next one. Skipping pays only where the")
    print("ladder inverts, which is what the fog/cloud batching numbers do.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrix", type=Path, nargs="?",
                        default=Path("results/traces/sst2_test.matrix.jsonl"))
    args = parser.parse_args()

    rows = load_matrix(args.matrix)
    print(f"{len(rows)} queries, {len(TIERS)} tiers, replayed offline "
          f"(no model loaded, no network)")
    print(f"energy: RAPL local_measurement, net J/inference "
          f"{ {t: ENERGY_J[t] for t in TIERS} }")

    test1_beta_is_a_quota(rows)
    test2_window_ablation(rows)
    test3_stepwise_vs_destination(rows)
    test4_when_does_skipping_pay(rows)


if __name__ == "__main__":
    main()
