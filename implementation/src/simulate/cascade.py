"""One simulated run: a policy serving the whole query stream through the cascade.

run() follows one query at a time up the tiers, then lets its answer travel back down so
the households learn from it.
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import collections
import random
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from energy.three_tier import TIERS
from olt_load import BURSTGPT
from routing import Learned, on_arrival, on_escalation, TOP, Window

# ==========================================
# Policies
# ==========================================
# All but the two fixed chains share one routing rule (routing.py) and differ only in
# where they get the OLT's energy cost from:
#   recserve         RecServe as published: always one tier up (the baseline)
#   recserve_no_onu  RecServe on phone -> OLT (control: is a saving just from dropping the ONU?)
#   static_day       one fixed OLT rate for the whole day, calibrated in advance
#   static_hour      one OLT rate per hour of the day, weekday or weekend (a timetable)
#   broadcast        the OLT's 5-minute mean, broadcast on the PON to every ONU (the proposal)
#   oracle           the true expected rate right now (the most any live signal could do)
#   piggyback        the OLT's 5-minute mean, heard only on the household's own answers
#   stale_low/high   static_day calibrated at 1/4 or 4x the real load (a stale config)
POLICIES = (
    "recserve",
    "recserve_no_onu",
    "static_day",
    "stale_low",
    "stale_high",
    "static_hour",
    "piggyback",
    "broadcast",
    "oracle",
)
FIXED = ("recserve", "recserve_no_onu")
"""Fixed chains: no energy information at all."""

STATIC = ("static_day", "stale_low", "stale_high")
"""One fixed OLT rate, shipped as configuration."""


@dataclass
class RunSetup:
    """Everything one run needs; built once per OLT peak load, shared by every beta and policy.

    Attributes:
        answers: answers[question][tier], that tier's recorded answer (correct, confidence,
            prompt and generated tokens, bytes).
        stream: [(question, time in hours, household)], time-ordered (build_stream).
        batches: The OLT batch each query would meet, 1 + Poisson(load).
        loads: The OLT's true offered load at each query's arrival.
        prices: True energy rates (olt_energy.Energy).
        static_rates: The pre-calibrated OLT rates of static_day, stale_low/high, static_hour.
        settings: The run's settings (simulate.py's parsed arguments).
        olt_reports: The OLT's report at each query, for piggyback (--report window).
        schedule_key: Maps a time to static_hour's table key (day type, hour).
        broadcasts: The latest broadcast at each query, for broadcast.
    """

    answers: dict
    stream: list
    batches: list
    loads: list
    prices: object
    static_rates: dict
    settings: object
    olt_reports: list | None = None
    schedule_key: Callable | None = None
    broadcasts: list | None = None


# ==========================================
# One run
# ==========================================
def believed_olt_rate(
    policy: str, setup: RunSetup, n: int, hour: float, load: float, learned: Learned
):
    """Return what a policy believes the OLT costs now: the one place the policies differ.

    Args:
        policy: One of POLICIES (not a fixed chain).
        setup: The run's setup.
        n: Index of the query in the stream.
        hour: The query's arrival time, in hours.
        load: The OLT's true load at that time.
        learned: The household's knowledge (for piggyback).

    Returns:
        (J/prompt token, J/generated token), or None while piggyback has not heard the OLT.
    """
    if policy in STATIC:
        return setup.static_rates[policy]
    if policy == "static_hour":
        return setup.static_rates["static_hour"][setup.schedule_key(hour)]
    if policy == "oracle":
        return setup.prices.expected_olt(load)
    if policy == "broadcast":
        return setup.broadcasts[n]
    return learned.rates.get("olt")  # piggyback


def run(setup: RunSetup, beta: float, policy: str):
    """Serve every query of the stream under one policy and one RecServe beta.

    Energy is charged at the true rates and the batch each query met at the OLT. Latency
    (compute seconds) and the bytes crossing the PON are measured without affecting any
    decision.

    Args:
        setup: Everything the run needs (RunSetup).
        beta: RecServe's escalation quantile.
        policy: One of POLICIES.

    Returns:
        (metrics over the whole stream, per-hour detail).
    """
    s, prices = setup.settings, setup.prices
    # RecServe's thresholds: one per tier, shared by every household (the tier is shared)
    windows = {t: Window(s.window) for t in TIERS}
    households: dict[int, dict[str, Learned]] = {}
    pool = (
        {t: Learned(s.alpha, s.warmup, s.rate_alpha) for t in TIERS[:TOP]}
        if s.shared_stats
        else None
    )
    energy = correct = forwarded = skipped = confidence_delivered = comm_bytes = pon_bytes = 0.0
    rate_errors, latencies = [], []
    answered_at = collections.Counter()
    hourly = [[0.0, 0, 0, 0, 0] for _ in range(24)]  # joules, queries, answered at user/onu/olt

    for n, ((question, hour, household), batch, load) in enumerate(
        zip(setup.stream, setup.batches, setup.loads)
    ):
        learners = households.get(household)
        if learners is None:
            learners = households[household] = {
                t: Learned(s.alpha, s.warmup, s.rate_alpha, pool[t] if pool else None)
                for t in TIERS[:TOP]
            }
        prompt_tokens = setup.answers[question]["user"]["tp"]
        packet: dict[str, tuple[float, float, int]] = {}  # rides back down: rates, answer length
        spent = took = 0.0  # joules and seconds this query costs
        here = 0  # index of the tier the query is at (0 = the phone)
        rates = None
        while True:
            tier = TIERS[here]
            learned = learners.get(tier)

            # --- Can this tier decide where the query goes? ---
            # Not under a fixed chain, not at the OLT, not before it has heard enough.
            decide = policy not in FIXED and learned is not None and learned.ready(TIERS[here:])
            if decide:
                if pool:  # a household knows its own devices' rates; only the OLT's must reach it
                    rates = {u: prices.rates(u, 1) for u in TIERS[here:TOP]}
                    rates["olt"] = learned.rates.get("olt")
                else:
                    rates = {u: learned.rates[u] for u in TIERS[here:]}
                if policy != "piggyback":
                    rates["olt"] = believed_olt_rate(policy, setup, n, hour, load, learned)
                if policy in ("piggyback", "broadcast") and rates["olt"] is not None:
                    true_generated = prices.expected_olt(load)[1]
                    rate_errors.append(abs(rates["olt"][1] - true_generated) / true_generated)
                decide = rates["olt"] is not None  # piggyback: not heard the OLT yet

            # --- On arrival: run here, or forward to a cheaper tier above? ---
            if decide:
                target = on_arrival(here, prompt_tokens, rates, learned, s.delta)
                if target != here:
                    forwarded += 1
                    here = target
                    continue

            # --- The tier answers: replay its recorded answer, charge the true energy ---
            answer = setup.answers[question][tier]
            j_prompt, j_generated = prices.rates(tier, batch)
            spent += j_prompt * answer["tp"] + j_generated * answer["tg"]
            took += prices.seconds(tier, batch, answer["tp"], answer["tg"])
            reported = (
                setup.olt_reports[n]
                if setup.olt_reports and tier == "olt"
                else (j_prompt, j_generated)
            )
            packet[tier] = (*reported, answer["tg"])

            # --- RecServe's test: escalate below the beta-quantile of recent confidences ---
            window = windows[tier]
            escalate = here < TOP and len(window) > 1 and answer["conf"] < window.quantile(beta)
            window.add(answer["conf"])
            if not escalate:
                correct += answer["correct"]
                confidence_delivered += answer["conf"]  # of the answer the user actually gets
                question_bytes = setup.answers[question]["user"]["qb"]
                # RecServe's communication burden: the input goes up and the output comes down
                # once per hop, 2(i-1)(|x|+|y|). Skipping a tier saves inference, not bytes.
                comm_bytes += 2 * here * (question_bytes + answer["ab"])
                # What crosses the shared PON fibre: question up, answer down, once, for
                # every query the OLT answers.
                pon_bytes += (question_bytes + answer["ab"]) if here == TOP else 0
                answered_at[tier] += 1
                break

            # --- On escalation: the next tier up, or a cheaper later one? ---
            target = (
                on_escalation(here, prompt_tokens, rates, learned, s.delta) if decide else here + 1
            )
            if policy == "recserve_no_onu":
                target = TOP
            skipped += target > here + 1
            here = target

        # --- Account for the query ---
        energy += spent
        latencies.append(took)
        row = hourly[int(hour) % 24]
        row[0] += spent
        row[1] += 1
        row[2 + TIERS.index(tier)] += 1

        # --- The answer travels back down ---
        # Every deciding tier at or below where it was produced reads the packet for itself
        # and the tiers above it. The ONU relays everything above the user, so it hears
        # reports even for queries whose inference it skipped.
        for own_tier, learner in learners.items():
            if TIERS.index(own_tier) <= TIERS.index(tier):
                learner.absorb(
                    {u: v for u, v in packet.items() if TIERS.index(u) >= TIERS.index(own_tier)},
                    tier,
                )

    total = len(setup.stream)
    metrics = {
        "accuracy": correct / total,
        "mean_confidence": confidence_delivered / total,
        "comm_MB_per_1k_queries": comm_bytes / total * 1000 / 1e6,  # RecServe's metric
        "pon_MB_per_1k_queries": pon_bytes / total * 1000 / 1e6,  # bytes over the shared fibre
        "latency_s_mean": st.mean(latencies),
        "latency_s_p95": float(np.percentile(latencies, 95)),
        "J_per_query": energy / total,
        "forwarded_on_arrival": forwarded / total,
        "skipped_on_escalation": skipped / total,
        "olt_rate_error": st.mean(rate_errors) if rate_errors else float("nan"),
        **{f"final_{t}": answered_at[t] / total for t in TIERS},
    }
    per_hour = [
        {
            "hour": h,
            "J_per_query": row[0] / row[1] if row[1] else None,
            "queries": row[1],
            **{
                f"final_{t}": (row[2 + k] / row[1] if row[1] else None) for k, t in enumerate(TIERS)
            },
        }
        for h, row in enumerate(hourly)
    ]
    return metrics, per_hour


# ==========================================
# Query stream
# ==========================================
def build_stream(questions, days, per_day, seed, households=1, trace=None, shape="trace"):
    """Build the time-ordered arrivals at the households' phones.

    Without a trace every day is BurstGPT's average day. With one, arrivals cover the
    trace's days after training, a Poisson number per hour. Questions are drawn at random
    with replacement, and each arrival belongs to a household drawn at random.

    Args:
        questions: The question indices that can be asked.
        days: Days simulated (synthetic mode only).
        per_day: Queries per day, per household.
        seed: Random seed.
        households: Households sharing the OLT.
        trace: A TraceLoad to follow, or None for the synthetic average day.
        shape: "trace" (busiest when the OLT is) or "flat" (the same every hour).

    Returns:
        [(question, time in hours, household)], sorted by time.
    """
    rng = random.Random(seed)
    pick_household = (lambda: rng.randrange(households)) if households > 1 else (lambda: 0)
    out = []
    if trace is None:
        total = sum(BURSTGPT)
        for day in range(days):
            times = []
            for h in range(24):
                times += [
                    24 * day + h + rng.random()
                    for _ in range(round(per_day * households * BURSTGPT[h] / total))
                ]
            out += [(rng.choice(questions), t, pick_household()) for t in sorted(times)]
        return out
    nprng = np.random.default_rng([seed, 31])
    test_days = trace.c[trace.train_days :]
    per_request = per_day * households * len(test_days) / test_days.sum()
    for d, day in enumerate(test_days):
        for h, requests in enumerate(day):
            if shape == "flat":
                requests = test_days.mean()
            t0 = 24 * (trace.train_days + d) + h
            for t in sorted(t0 + nprng.random(nprng.poisson(per_request * requests))):
                out.append((rng.choice(questions), float(t), pick_household()))
    return out
