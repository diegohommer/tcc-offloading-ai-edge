#!/usr/bin/env python3
"""Seven offline tests over the SST-2 policy matrix. No GPU, no network, no billing.

HEADLINE, stated up front because four of the five tests land on it:

    The geometric decay beta^k is ALREADY an effective energy-saving device. It
    makes expensive events rare by construction, so per-escalation optimizations
    have very little left to save. Every mechanism tried here dies on that same
    fact -- not on four separate accidents.

      * energy-weighted threshold  -> beta is already that knob      (Test 1)
      * jump to a destination      -> beta^k already makes the
                                      expensive tiers rare           (Tests 3, 4)
      * per-tier beta vector       -> reach is a PRODUCT, so tiers
                                      cannot be targeted separately  (Test 5)

      * closed-loop rate control   -> a shorter window is simpler
                                      AND better                    (Test 6)

    Test 7 is the exception, and it is the one positive mechanism here: choosing
    WHICH TIERS EXIST changes the E_k vector rather than the traffic profile
    beta^k imposes on it, so the degeneracy argument does not reach it. On the
    measured monotonic ladder the full chain is justified; on an inverted ladder
    the middle tiers are Pareto-dominated and two tiers deliver 3.3x less energy.


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

The intuition was that traversal is waste: a query ending at cloud paid for
user + onu + fog + cloud, and the tiers it crossed contributed nothing -- they
were wrong, which is why it escalated. On the RAPL ladder that path is 4.588 J
against 2.452 J for going straight there.

RESULT: THAT INTUITION IS WRONG, and this test refutes it. Skipping costs MORE
at every beta. The error is that 4.588 J prices a full traversal, but Test 1's
own beta^k law says almost nothing completes one -- at beta=0.3 only 2.7% of
queries reach cloud. Most escalations stop at the cheap ONU, so routing 30% of
traffic straight to fog costs more than letting 2.7% climb there.

TEST 4 -- WHEN WOULD SKIPPING PAY?
-----------------------------------
Salvaging the general form from Test 3's refutation. Skipping to d beats stepwise
iff  E_d < E_onu + beta*E_fog + beta^2*E_cloud: the destination must undercut the
NEXT tier plus the discounted tail, not the full traversal. Applied to two real
ladders, this says skipping pays only where the ladder INVERTS -- which the
measured SST-2 ladder does not, and the fog/cloud batching numbers do.

TEST 5 -- PER-TIER BETA VECTOR
-------------------------------
The last place an energy-aware choice could act without being redundant with beta.
Reach(k) is the product of the betas below k, so one global beta forces a
geometric profile while a vector could give any decreasing one.

RESULT: it does not beat the scalar, on either ladder, with a train/test split.
And the reason is structural, not statistical: reach is a PRODUCT, so to deliver
traffic to a cheap top tier you must first pay the expensive tier beneath it. No
beta vector decouples them. That is precisely the limit destination selection
would break -- which is why skipping is about decoupling tiers, not about energy.

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


def replay(rows: list[dict], beta, window: int, destination: str | None = None,
           energy: dict[str, float] | None = None) -> dict:
    """Replay the cascade once. destination=None is RecServe's stepwise recursion.

    With a destination, the escalation DECISION is identical -- same rule, same
    beta, same history -- and only the target changes: an escalating query jumps
    straight to `destination` instead of walking there one tier at a time. That
    isolates the routing choice from the escalation choice, so any difference is
    attributable to the destination alone.
    """
    # beta may be a scalar (RecServe: one global beta) or a per-tier mapping.
    # Per-tier is strictly more expressive: with one beta the traffic profile is
    # forced to be geometric (beta, beta^2, beta^3), while a vector can be any
    # decreasing sequence -- escalate freely at a cheap tier, sparingly at an
    # expensive one.
    betas = beta if isinstance(beta, dict) else {t: beta for t in TIERS}
    E = energy if energy is not None else ENERGY_J

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
            energy_total += E[tier]
            cell = row["tiers"][tier]
            confidence = cell["confidence"]

            nxt = NEXT_TIER.get(tier)
            escalate = False
            if nxt is not None and len(history[tier]) > 1:
                # RecServe's rule verbatim: beta-quantile of this tier's own
                # recent confidences, escalate if below it.
                threshold = quantile(sorted(history[tier]), betas[tier])
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
        "betas": dict(betas),
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


def test5_per_tier_beta(rows: list[dict]) -> None:
    """Does a per-tier beta VECTOR beat the single global beta RecServe uses?

    Motivation: Test 1 showed reach(k) = prod of the betas below k. With one
    global beta that product is forced to be geometric -- beta, beta^2, beta^3.
    A vector can produce any decreasing sequence, so in principle it can escalate
    freely at a cheap tier and sparingly at an expensive one. That is a real extra
    degree of freedom, and the only place left where an energy-aware choice could
    act without being redundant with beta itself.

    The vector has 3 parameters against the scalar's 1, so it can overfit a
    frontier. Both are therefore FITTED on one half and REPORTED on the other, and
    the comparison is at matched energy: for each budget, take the configuration
    with the best training accuracy that fits the budget, then read its held-out
    accuracy.
    """
    print("\n" + "=" * 78)
    print("TEST 5 -- per-tier beta vector vs. RecServe's single global beta")
    print("=" * 78)

    half = len(rows) // 2
    train, test = rows[:half], rows[half:]
    grid = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

    scalar_cfgs = [{t: b for t in TIERS} for b in grid]
    vector_cfgs = [{"user": a, "onu": b, "fog": c, "cloud": 0.0}
                   for a in grid for b in grid for c in grid]

    for ladder_name, E in (
        ("SST-2 RAPL ladder (monotonic, measured on this CPU)", ENERGY_J),
        ("generative/PON ladder (inverted) -- sensitivity, NOT a measurement",
         {"user": 14.8, "onu": 196.0, "fog": 75.7, "cloud": 18.76}),
    ):
        print(f"\n{ladder_name}")

        def fit(cfgs):
            return [(replay(train, c, 10000, energy=E), c) for c in cfgs]

        s_fit, v_fit = fit(scalar_cfgs), fit(vector_cfgs)
        budgets = [r["J_per_query"] for r, _ in s_fit]

        print(f"  {'budget J':>9} {'scalar acc':>11} {'vector acc':>11} "
              f"{'delta':>8} {'best vector (user,onu,fog)':>28}")
        for budget in sorted(budgets):
            def best(fitted):
                ok = [(r, c) for r, c in fitted if r["J_per_query"] <= budget * 1.001]
                return max(ok, key=lambda rc: rc[0]["accuracy"]) if ok else None

            bs, bv = best(s_fit), best(v_fit)
            if bs is None or bv is None:
                continue
            # Held-out evaluation of the configurations chosen on train.
            s_test = replay(test, bs[1], 10000, energy=E)
            v_test = replay(test, bv[1], 10000, energy=E)
            trio = tuple(round(bv[1][t], 2) for t in ("user", "onu", "fog"))
            print(f"  {budget:>9.3f} {s_test['accuracy']:>11.4f} "
                  f"{v_test['accuracy']:>11.4f} "
                  f"{v_test['accuracy'] - s_test['accuracy']:>+8.4f} {str(trio):>28}")


def replay_controlled(rows: list[dict], target: float, window: int, gain: float,
                      rate_window: int = 50, energy: dict[str, float] | None = None) -> dict:
    """Closed-loop variant: regulate the ESCALATION RATE instead of the quantile.

    RecServe sets threshold = quantile(history, beta) and hopes the resulting
    escalation rate equals beta. Test 2 shows it does not when the workload
    shifts: a stale history makes everything look below average.

    Here the tier measures its own recent escalation rate and corrects the
    quantile level it asks for:

        q <- q + gain * (target - measured_rate)

    so if the history is stale and the tier is escalating everything, q falls
    until the rate comes back to target. It regulates the OUTPUT rather than
    trusting the input, which is the whole point.
    """
    E = energy if energy is not None else ENERGY_J
    history: dict[str, list[float]] = {t: [] for t in TIERS}
    recent: dict[str, list[int]] = {t: [] for t in TIERS}
    q = {t: target for t in TIERS}

    reached = {t: 0 for t in TIERS}
    escalated_from = {t: 0 for t in TIERS}
    decided_at = {t: 0 for t in TIERS}
    n_correct = 0
    energy_total = 0.0
    entry_decisions: list[int] = []

    for row in rows:
        tier = "user"
        while True:
            reached[tier] += 1
            energy_total += E[tier]
            cell = row["tiers"][tier]

            nxt = NEXT_TIER.get(tier)
            escalate = False
            if nxt is not None and len(history[tier]) > 1:
                threshold = quantile(sorted(history[tier]), q[tier])
                escalate = cell["confidence"] < threshold

            history[tier].append(cell["confidence"])
            if len(history[tier]) > window:
                history[tier].pop(0)

            if nxt is not None:
                decided_at[tier] += 1
                escalated_from[tier] += int(escalate)
                if tier == "user":
                    entry_decisions.append(int(escalate))
                # Feedback: nudge the requested quantile toward the target rate.
                recent[tier].append(int(escalate))
                if len(recent[tier]) > rate_window:
                    recent[tier].pop(0)
                measured = sum(recent[tier]) / len(recent[tier])
                q[tier] = min(1.0, max(0.0, q[tier] + gain * (target - measured)))

            if escalate:
                tier = nxt
                continue
            n_correct += int(cell["correct"])
            break

    n = len(rows)
    return {
        "accuracy": n_correct / n,
        "J_per_query": energy_total / n,
        "escalation_rate": {t: escalated_from[t] / decided_at[t] if decided_at[t] else 0.0
                            for t in TIERS if t in NEXT_TIER},
        "entry_decisions": entry_decisions,
    }


def test6_closed_loop(rows: list[dict]) -> None:
    """Does closing the loop beat simply using a SHORT window?

    This is the honest question. Test 2 already showed window=20 delivers 0.318
    against a 0.30 target. If a short window is enough, then the "proposal" is a
    config change, not a mechanism, and should be reported as such.
    """
    print("\n" + "=" * 78)
    print("TEST 6 -- closed-loop rate control vs. just shortening the window")
    print("=" * 78)
    print("Same shifted stream as Test 2. Target escalation rate = 0.30.")
    print("A short window tracks the current distribution but estimates the")
    print("quantile from few samples; the controller regulates the rate directly.\n")

    ranked = sorted(rows, key=lambda r: -r["tiers"]["user"]["confidence"])
    half = len(ranked) // 2
    rng = random.Random(0)
    easy, hard = ranked[:half], ranked[half:]
    rng.shuffle(easy)
    rng.shuffle(hard)
    stream = easy + hard

    def rate(d):
        return sum(d) / len(d) if d else 0.0

    seg = max(1, len(stream) // 10)
    print(f"  {'policy':>34} {'just after shift':>17} {'settled':>9} {'overall':>8} "
          f"{'|err|':>7} {'acc':>7} {'J/q':>7}")

    rows_out = []
    for label, r in (
        ("RecServe, window=10000", replay(stream, 0.30, 10000)),
        ("RecServe, window=200", replay(stream, 0.30, 200)),
        ("RecServe, window=50", replay(stream, 0.30, 50)),
        ("RecServe, window=20", replay(stream, 0.30, 20)),
        ("closed loop, window=10000, g=0.5", replay_controlled(stream, 0.30, 10000, 0.5)),
        ("closed loop, window=200,   g=0.5", replay_controlled(stream, 0.30, 200, 0.5)),
        ("closed loop, window=50,    g=0.5", replay_controlled(stream, 0.30, 50, 0.5)),
    ):
        d = r["entry_decisions"]
        overall = rate(d)
        rows_out.append((label, rate(d[half:half + seg]), rate(d[half + 2 * seg:]),
                         overall, abs(overall - 0.30), r["accuracy"], r["J_per_query"]))
    for label, after, settled, overall, err, acc, j in rows_out:
        print(f"  {label:>34} {after:>17.3f} {settled:>9.3f} {overall:>8.3f} "
              f"{err:>7.3f} {acc:>7.4f} {j:>7.4f}")


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
    test5_per_tier_beta(rows)
    test6_closed_loop(rows)
    test7_which_tiers_should_exist(rows)




def replay_chain(rows: list[dict], chain: tuple[str, ...], beta: float, window: int,
                 energy: dict[str, float]) -> dict:
    """Replay over an ARBITRARY tier chain, not the fixed user->onu->fog->cloud.

    This is the one lever left. Five mechanisms failed because they tried to make
    the DECISION RULE energy-aware, and beta already owns that rule. Choosing
    which tiers exist is different in kind: it changes the E_k vector itself
    rather than the traffic profile beta^k imposes on it, so the beta-degeneracy
    argument does not apply to it.

    It is also the only lever that can exploit a non-monotonic energy ladder. A
    beta vector cannot (Test 5: reach is a product) and skipping cannot pay
    unless the ladder already inverts (Test 4). Deleting a tier, by contrast,
    removes its term from the sum outright and promotes every tier above it to a
    shallower -- hence more heavily trafficked -- position.
    """
    nxt = {chain[i]: chain[i + 1] for i in range(len(chain) - 1)}
    history: dict[str, list[float]] = {t: [] for t in chain}
    n_correct = 0
    energy_total = 0.0
    final = {t: 0 for t in chain}

    for row in rows:
        tier = chain[0]
        while True:
            energy_total += energy[tier]
            cell = row["tiers"][tier]
            up = nxt.get(tier)
            escalate = False
            if up is not None and len(history[tier]) > 1:
                escalate = cell["confidence"] < quantile(sorted(history[tier]), beta)
            history[tier].append(cell["confidence"])
            if len(history[tier]) > window:
                history[tier].pop(0)
            if escalate:
                tier = up
                continue
            n_correct += int(cell["correct"])
            final[tier] += 1
            break

    n = len(rows)
    return {"accuracy": n_correct / n, "J_per_query": energy_total / n,
            "final_share": {t: final[t] / n for t in chain}}


def test7_which_tiers_should_exist(rows: list[dict]) -> None:
    """Choose the SUBSET of tiers that minimises energy under an accuracy floor."""
    import itertools

    print("\n" + "=" * 78)
    print("TEST 7 -- which tiers should exist at all?")
    print("=" * 78)
    print("Every subset that keeps the entry tier, replayed at several betas.")
    print("Chosen on one half, reported on the held-out half.\n")

    half = len(rows) // 2
    train, test = rows[:half], rows[half:]

    for ladder, E in (
        ("SST-2 RAPL ladder (monotonic, measured on this CPU)", ENERGY_J),
        ("generative/PON ladder (inverted) -- sensitivity, not a measurement",
         {"user": 14.8, "onu": 196.0, "fog": 75.7, "cloud": 18.76}),
    ):
        print(f"\n{ladder}")
        results = []
        for r in range(1, len(TIERS) + 1):
            for chain in itertools.combinations(TIERS, r):
                if chain[0] != "user":
                    continue
                for beta in (0.1, 0.2, 0.3, 0.5):
                    results.append((chain, beta, replay_chain(train, chain, beta, 10000, E)))

        front = [(c, b, t) for c, b, t in results
                 if not any(o[2]["J_per_query"] < t["J_per_query"]
                            and o[2]["accuracy"] >= t["accuracy"] for o in results)]
        front.sort(key=lambda x: x[2]["J_per_query"])
        full = [(c, b, t) for c, b, t in results if c == TIERS]

        print(f"  {'chain':>26} {'beta':>5} {'test acc':>9} {'test J/q':>9}")
        print("  -- Pareto frontier (chosen on train, reported on held-out) --")
        for chain, beta, _ in front[:6]:
            te = replay_chain(test, chain, beta, 10000, E)
            print(f"  {'->'.join(chain):>26} {beta:>5.2f} "
                  f"{te['accuracy']:>9.4f} {te['J_per_query']:>9.4f}")
        print("  -- full 4-tier chain (what RecServe's architecture fixes) --")
        for chain, beta, _ in sorted(full, key=lambda x: x[1]):
            te = replay_chain(test, chain, beta, 10000, E)
            print(f"  {'->'.join(chain):>26} {beta:>5.2f} "
                  f"{te['accuracy']:>9.4f} {te['J_per_query']:>9.4f}")


if __name__ == "__main__":
    main()
