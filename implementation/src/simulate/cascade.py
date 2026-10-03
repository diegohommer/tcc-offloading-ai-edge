"""One simulated run: a policy serving the whole query stream through the cascade.

run() follows one query at a time up the tiers, then lets its answer travel back down so
the households learn from it.
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import collections
import heapq
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from energy.three_tier import TIERS
from olt_server import OltServer
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
        stream: [(question, time in hours, household)], time-ordered.
        curve: The OLT's measured batch curve (energy.three_tier.OltCurve).
        prices: True energy rates (olt_energy.Energy).
        static_rates: The pre-calibrated OLT rates of static_day, stale_low/high, static_hour.
        settings: The run's settings (simulate.py's parsed arguments).
        schedule_key: Maps a time to static_hour's table key (day type, hour).
    """

    answers: dict
    stream: list
    curve: object
    prices: object
    static_rates: dict
    settings: object
    schedule_key: Callable | None = None


# ==========================================
# One run
# ==========================================
def believed_olt_rate(policy: str, setup: RunSetup, server, hour: float, learned: Learned):
    """Return what a policy believes the OLT costs now: the one place the policies differ.

    Args:
        policy: One of POLICIES (not a fixed chain).
        setup: The run's setup.
        server: The OLT, for the policies that can see it.
        hour: The query's arrival time, in hours.
        learned: The household's knowledge (for piggyback).

    Returns:
        (J/prompt token, J/generated token), or None while piggyback has not heard the OLT.
    """
    if policy in STATIC:
        return setup.static_rates[policy]
    if policy == "static_hour":
        return setup.static_rates["static_hour"][setup.schedule_key(hour)]
    if policy == "oracle":
        # the window mean with no staleness: the most a live signal could carry. Reading the
        # batch of this instant instead would be worse than useless, since a query joins the
        # OLT seconds later and generates for nine more, with others arriving throughout.
        return server.reported_rate()
    if policy == "broadcast":
        # the same mean, but only as fresh as the OLT's last send; the ONU relays it to the
        # household's devices over the LAN, taken as free and instant
        return server.broadcast(hour * 3600, setup.settings.broadcast_interval_s)
    return learned.rates.get("olt")  # piggyback


def run(setup: RunSetup, beta: float, policy: str):
    """Serve every query of the stream under one policy and one RecServe beta.

    Queries are walked in arrival order, but one that escalates does not reach the OLT at
    once: it spends seconds being answered by the tiers below first. Those are held in
    `waiting` and handed over at the moment they truly arrive, so the batch a query meets is
    made of whoever is genuinely mid-answer beside it.

    The OLT's cost is therefore not drawn from anything. It is whatever the households' own
    escalated queries make it, which means a policy that sends more that way also makes the
    OLT cheaper per query. That feedback is real, and it is left in.

    Args:
        setup: Everything the run needs (RunSetup).
        beta: RecServe's escalation quantile.
        policy: One of POLICIES.

    Returns:
        (metrics over the whole stream, per-hour detail, the OLT rates seen in each
        (day type, hour) cell, which is what a timetable is built from).
    """
    # The loop below walks one query at a time. Two helpers sit between it and the numbers:
    # settle() records a query whose answer is final, wherever it was produced, and land()
    # settles the ones the OLT has just handed back. Everything else is inline.
    s, prices = setup.settings, setup.prices
    top = TIERS[TOP]
    windows = {t: Window(s.window) for t in TIERS}
    households: dict[int, dict[str, Learned]] = {}
    pool = (
        {t: Learned(s.alpha, s.warmup, s.rate_alpha) for t in TIERS[:TOP]}
        if s.shared_stats
        else None
    )
    server = OltServer(setup.curve, prices, s.report_window * 60)

    energy = correct = forwarded = skipped = confidence_delivered = comm_bytes = pon_bytes = 0.0
    rate_errors, latencies = [], []
    answered_at = collections.Counter()
    hourly = [[0.0, 0, 0, 0, 0] for _ in range(24)]
    seen: dict = {}  # (day type, hour) -> [sum of J/prompt token, sum of J/gen token, answers]
    waiting: list = []  # heap of (second it reaches the OLT, query id)
    flight: dict = {}  # query id -> what a query carries while the OLT holds it
    next_id = 0

    def settle(q, hour, household, tier, spent, took, packet):
        """Record a finished query and let its answer travel back down.

        Args:
            q: The question asked.
            hour: Its arrival time, in hours.
            household: Who asked it.
            tier: The tier whose answer was kept.
            spent: Joules the query cost.
            took: Seconds it took, queueing at the OLT included.
            packet: {tier: (J/prompt token, J/generated token, tokens)} riding back down.
        """
        nonlocal energy, correct, confidence_delivered, comm_bytes, pon_bytes
        answer = setup.answers[q][tier]
        correct += answer["correct"]
        confidence_delivered += answer["conf"]
        qb = setup.answers[q]["user"]["qb"]
        # RecServe's communication burden: the input goes up and the output comes down once
        # per hop, 2(i-1)(|x|+|y|). Skipping a tier saves inference, not bytes.
        comm_bytes += 2 * TIERS.index(tier) * (qb + answer["ab"])
        # What crosses the shared PON fibre: question up, answer down, once, for every query
        # the OLT answers.
        pon_bytes += (qb + answer["ab"]) if tier == top else 0
        energy += spent
        latencies.append(took)
        answered_at[tier] += 1
        row = hourly[int(hour) % 24]
        row[0] += spent
        row[1] += 1
        row[2 + TIERS.index(tier)] += 1
        # Every deciding tier at or below where it was produced reads the packet for itself
        # and the tiers above it. The ONU relays everything above the user, so it hears
        # reports even for queries whose inference it skipped.
        for own, learner in households[household].items():
            if TIERS.index(own) <= TIERS.index(tier):
                learner.absorb(
                    {u: v for u, v in packet.items() if TIERS.index(u) >= TIERS.index(own)}, tier
                )

    def land(completions):
        """Settle every query the OLT has just finished.

        Args:
            completions: [(query id, wall clock, decode joules)] as the server returns them.
        """
        for qid, done_at, decode_j in completions:
            f = flight.pop(qid)
            tg = setup.answers[f["q"]][top]["tg"]
            own = (f["reported"][0], decode_j / tg if tg else 0.0, tg)
            cell = seen.setdefault(setup.schedule_key(f["hour"]), [0.0, 0.0, 0])
            cell[0] += own[0]
            cell[1] += own[1]
            cell[2] += 1
            # The packet carries either what this query's own answer cost, or the OLT's mean
            # over its recent traffic, which is far steadier under marginal accounting.
            windowed = server.reported_rate() if s.report == "window" else None
            f["packet"][top] = (*windowed, tg) if windowed else own
            settle(
                f["q"],
                f["hour"],
                f["household"],
                top,
                f["spent"] + f["prefill_j"] + decode_j,
                f["took"] + (done_at - f["reached"]),
                f["packet"],
            )

    for q, hour, household in setup.stream:
        now = hour * 3600

        # --- Hand over everyone who has genuinely reached the OLT by now ---
        while waiting and waiting[0][0] <= now:
            at, qid = heapq.heappop(waiting)
            land(server.advance(at))
            f = flight[qid]
            f["prefill_j"] = server.admit(qid, f["prompt_tokens"], f["gen_tokens"], at)
            f["reported"] = prices.rates(top, max(server.batch, 1))
        land(server.advance(now))

        # --- Walk the query up until it is answered, or until it leaves for the OLT ---
        if household not in households:
            households[household] = {
                t: Learned(s.alpha, s.warmup, s.rate_alpha, pool[t] if pool else None)
                for t in TIERS[:TOP]
            }
        learners = households[household]
        prompt_tokens = setup.answers[q]["user"]["tp"]
        packet: dict[str, tuple[float, float, float]] = {}
        spent = took = 0.0
        here = 0
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
                    rates["olt"] = believed_olt_rate(policy, setup, server, hour, learned)
                if policy in ("piggyback", "broadcast") and rates["olt"] is not None:
                    true_dec = prices.rates(top, max(server.batch, 1))[1]
                    if true_dec:
                        rate_errors.append(abs(rates["olt"][1] - true_dec) / true_dec)
                decide = rates["olt"] is not None  # piggyback: not heard the OLT yet

            # --- On arrival: run here, or forward to a cheaper tier above? ---
            if decide:
                target = on_arrival(here, prompt_tokens, rates, learned, s.delta)
                if target != here:
                    forwarded += 1
                    here = target
                    continue

            # --- The OLT answers in its own time, so the query leaves the walk here ---
            if here == TOP:
                flight[next_id] = {
                    "q": q,
                    "hour": hour,
                    "household": household,
                    "prompt_tokens": prompt_tokens,
                    "gen_tokens": setup.answers[q][top]["tg"],
                    "spent": spent,
                    "took": took,
                    "packet": packet,
                    "reached": now + took,
                }
                heapq.heappush(waiting, (now + took, next_id))
                next_id += 1
                break

            # --- A tier below answers at once: replay it, charge the true energy ---
            answer = setup.answers[q][tier]
            j_prompt, j_generated = prices.rates(tier, 1)
            spent += j_prompt * answer["tp"] + j_generated * answer["tg"]
            took += prices.seconds(tier, 1, answer["tp"], answer["tg"])
            packet[tier] = (j_prompt, j_generated, answer["tg"])

            # --- RecServe's test: escalate below the beta-quantile of recent confidences ---
            window = windows[tier]
            escalate = len(window) > 1 and answer["conf"] < window.quantile(beta)
            window.add(answer["conf"])
            if not escalate:
                settle(q, hour, household, tier, spent, took, packet)
                break

            # --- On escalation: the next tier up, or a cheaper later one? ---
            target = (
                on_escalation(here, prompt_tokens, rates, learned, s.delta) if decide else here + 1
            )
            if policy == "recserve_no_onu":
                target = TOP
            skipped += target > here + 1
            here = target

    # --- Nobody else is coming: let the OLT take and finish what is left ---
    while waiting:
        at, qid = heapq.heappop(waiting)
        land(server.advance(at))
        f = flight[qid]
        f["prefill_j"] = server.admit(qid, f["prompt_tokens"], f["gen_tokens"], at)
        f["reported"] = prices.rates(top, max(server.batch, 1))
    land(server.drain())

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
        "mean_olt_batch": server.mean_batch,
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
    observed = {k: (v[0] / v[2], v[1] / v[2]) for k, v in seen.items() if v[2]}
    return metrics, per_hour, observed


# ==========================================
# What a timetable is built from
# ==========================================
def calibrate(setup: RunSetup, beta: float) -> dict:
    """Return the OLT rates the static policies ship with, observed over the training days.

    These tables cannot be worked out in advance any more. The OLT's cost is made by the
    traffic the households send it, so it has to be watched. An operator building a timetable
    has no energy-aware policy running yet, so what it watches is a plain RecServe month: the
    cascade climbing one tier at a time, with nothing routing on energy. The tables are then
    what the OLT charged over that month.

    Args:
        setup: A setup whose stream covers the calibration days.
        beta: RecServe's escalation quantile, the same the run will use.

    Returns:
        {"static_day": rates, "stale_low": ..., "stale_high": ..., "static_hour": {cell: rates}}.
    """
    metrics, _, observed = run(setup, beta, "recserve")
    if not observed:
        return {}
    n = len(observed)
    day = (
        sum(v[0] for v in observed.values()) / n,
        sum(v[1] for v in observed.values()) / n,
    )
    # The stale pair ship the day figure of an OLT whose batch is wrong by stale_factor,
    # standing for a table built for a different population than the one it meets.
    batch = max(metrics["mean_olt_batch"], 1.0)
    stale = setup.settings.stale_factor
    return {
        "static_day": day,
        "stale_low": setup.prices.rates("olt", max(int(batch / stale), 1)),
        "stale_high": setup.prices.rates("olt", max(int(batch * stale), 1)),
        "static_hour": observed,
    }
