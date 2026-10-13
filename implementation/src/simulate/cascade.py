"""One simulated run: a policy serving the whole query stream through the cascade.

run() follows one query at a time up the tiers, then lets its answer travel back down so
the households learn from it.
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import collections
import dataclasses
import heapq
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
from energy.three_tier import TIERS
from olt_server import OltServer
from routing import Learned, on_arrival, on_escalation, OwnStats, TOP, Window

# ==========================================
# Policies
# ==========================================
# All but the two fixed chains share one routing rule (routing.py) and differ only in
# where they get the OLT's energy cost from:
#   recserve         RecServe as published: always one tier up (the baseline)
#   recserve_no_onu  RecServe on phone -> OLT (control: is a saving just from dropping the ONU?)
#   static_day       one fixed OLT rate for the whole day, observed in advance
#   static_hour      one OLT rate per hour of the day, weekday or weekend (a timetable)
#   broadcast        the OLT's 5-minute report, broadcast on the PON to every ONU (the proposal)
#   oracle           the same mean with no broadcast delay
#   piggyback        the same report, heard only on the household's own answers
#   stale_low/high   static_day observed on a population 1/4 or 4x the real one
#   user/onu/olt_alone  one tier answers every query (reference points, no cascade)
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
    "user_alone",
    "onu_alone",
    "olt_alone",
)
ALONE = {"user_alone": 0, "onu_alone": 1, "olt_alone": TOP}
"""Single-tier policies, and the tier that answers every query."""

FIXED = ("recserve", "recserve_no_onu", *ALONE)
"""Fixed chains: no energy information at all."""

STATIC = ("static_day", "stale_low", "stale_high")
"""One fixed OLT rate, shipped as configuration."""


@dataclass
class RunSetup:
    """Everything one run needs; built once per population, shared by every beta and policy.

    Attributes:
        answers: answers[question][tier], that tier's recorded answer (correct, confidence,
            prompt and generated tokens, bytes).
        stream: [(question, time in hours, household)], time-ordered.
        curve: The OLT's measured batch curve (energy.three_tier.OltCurve).
        prices: True energy rates (olt_energy.Energy).
        static_rates: The pre-observed OLT rates of static_day, stale_low/high, static_hour.
        settings: The run's settings (simulate.py's parsed arguments).
        schedule_key: Maps a time to static_hour's table key (day type, hour).
        calibration: The setup of the calibration days, for tables relearned per beta.
        follow: follow[i] = (index of the next message of stream[i]'s conversation, pause
            in seconds before it) or None, for --timing after-answer.
    """

    answers: dict
    stream: list
    curve: object
    prices: object
    static_rates: dict
    settings: object
    schedule_key: Callable | None = None
    calibration: object = None
    follow: list | None = None


# ==========================================
# One run
# ==========================================
def believed_olt_rate(policy: str, setup: RunSetup, server, learned: Learned, hour: float):
    """Return what a policy believes the OLT costs now: the one place the policies differ.

    Args:
        policy: One of POLICIES (not a fixed chain).
        setup: The run's setup.
        server: The OLT, for the policies that can see it.
        learned: The household's knowledge (for piggyback).
        hour: The query's arrival time, in hours.

    Returns:
        (J/prompt token, J/generated token), or None while the policy has nothing to go on.
    """
    if policy in STATIC:
        return setup.static_rates.get(policy)
    if policy == "static_hour":
        timetable = setup.static_rates.get("static_hour", {})
        return timetable.get(setup.schedule_key(hour), setup.static_rates.get("static_day"))
    if policy == "oracle":
        return server.reported_rate()
    if policy == "broadcast":
        # the ONU relays it to the household's devices over the LAN, taken as free and instant
        return server.last_broadcast
    return learned.rates.get("olt")  # piggyback


def run(setup: RunSetup, beta: float, policy: str):
    """Serve every query of the stream under one policy and one RecServe beta.

    Queries are walked in arrival order. One that escalates reaches the OLT only after the
    seconds the tiers below spent on it, so it waits in `arrivals` until then, and the batch
    it meets is whoever is genuinely mid-answer at that moment. The OLT's cost is therefore
    whatever the households' own escalated queries make it.

    Args:
        setup: Everything the run needs (RunSetup).
        beta: RecServe's escalation quantile.
        policy: One of POLICIES.

    Returns:
        (metrics over the whole stream, per-hour detail, {(day type, hour): (mean J/prompt
        token, mean J/generated token, OLT answers)}, which is what a timetable is built from).
    """
    settings, prices = setup.settings, setup.prices
    top = TIERS[TOP]
    stats = settings.stats
    escalation = beta if settings.escalation == "beta" else None
    households: dict[int, dict[str, Learned]] = {}
    pool = (
        {
            tier: Learned(
                settings.alpha, settings.warmup, settings.rate_alpha, escalation=escalation
            )
            for tier in TIERS[:TOP]
        }
        if stats == "pooled"
        else None
    )

    # What ships with each tier's model, measured on the benchmark before deployment: its
    # mean answer length, and confidences to start a household's RecServe window from.
    shipped_tokens = {
        tier: statistics.mean(answer[tier]["tg"] for answer in setup.answers.values())
        for tier in TIERS
    }
    shipped_confidences = {
        tier: [answer[tier]["conf"] for answer in setup.answers.values()] for tier in TIERS
    }
    olt_own = OwnStats(settings.alpha, shipped_tokens[top])
    own: dict[int, dict[str, OwnStats]] = {}  # household -> the phone's and ONU's OwnStats

    shared_windows = {tier: Window(settings.window) for tier in TIERS}
    household_windows: dict[tuple[int, str], Window] = {}

    def window_for(household, tier):
        """Return the confidence window RecServe's test reads at this tier.

        Args:
            household: Who asked.
            tier: The tier that just answered.

        Returns:
            One window per tier for every household (--windows shared), or the household's
            own, started from shipped confidences (--windows household).
        """
        if settings.windows == "shared":
            return shared_windows[tier]
        key = (household, tier)
        if key not in household_windows:
            rng = np.random.default_rng([settings.seed, household, TIERS.index(tier)])
            confidences = shipped_confidences[tier]
            window = Window(settings.window)
            window.fill(
                [confidences[i] for i in rng.integers(0, len(confidences), settings.window)]
            )
            household_windows[key] = window
        return household_windows[key]

    server = OltServer(setup.curve, prices, settings.report_window * 60)

    energy = correct = forwarded = skipped = confidence_delivered = comm_bytes = pon_bytes = 0.0
    rate_errors, latencies = [], []
    answered_at = collections.Counter()
    hourly = [[0.0, 0, 0, 0, 0] for _ in range(24)]
    seen: dict = {}  # (day type, hour) -> [sum of J/prompt token, sum of J/gen token, answers]
    arrivals: list = []  # heap of (second it reaches the OLT, query id)
    in_flight: dict = {}  # query id -> what a query carries while the OLT holds it
    next_id = 0
    next_broadcast = -math.inf

    # Messages waiting to be sent, as a heap of (time in hours, stream index). With fixed
    # timing every message is in it from the start; after-answer timing starts with each
    # conversation's first message and adds the next one when an answer arrives.
    after_answer = settings.timing == "after-answer"
    if after_answer:
        followers = {link[0] for link in setup.follow if link is not None}
        pending = [
            (hour, index)
            for index, (_, hour, _) in enumerate(setup.stream)
            if index not in followers
        ]
    else:
        pending = [(hour, index) for index, (_, hour, _) in enumerate(setup.stream)]
    heapq.heapify(pending)

    def settle(index, hour, tier, spent, took, packet):
        """Record a finished query, let its answer travel back down, and queue the next message.

        Args:
            index: The query's place in the stream.
            hour: When it was sent, in hours.
            tier: The tier whose answer was kept.
            spent: Joules the query cost.
            took: Seconds it took, waiting at the OLT included.
            packet: {tier: (J/prompt token, J/generated token, tokens)} riding back down.
        """
        question, _, household = setup.stream[index]
        nonlocal energy, correct, confidence_delivered, comm_bytes, pon_bytes
        answer = setup.answers[question][tier]
        correct += answer["correct"]
        confidence_delivered += answer["conf"]
        question_bytes = setup.answers[question]["user"]["qb"]
        # RecServe's communication burden: the input goes up and the output comes down once
        # per hop, 2(i-1)(|x|+|y|). Skipping a tier saves inference, not bytes.
        comm_bytes += 2 * TIERS.index(tier) * (question_bytes + answer["ab"])
        # What crosses the shared PON fibre: question up, answer down, once, for every query
        # the OLT answers.
        pon_bytes += (question_bytes + answer["ab"]) if tier == top else 0
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
        for own_tier, learner in households[household].items():
            if TIERS.index(own_tier) <= TIERS.index(tier):
                heard = {
                    reported_tier: report
                    for reported_tier, report in packet.items()
                    if TIERS.index(reported_tier) >= TIERS.index(own_tier)
                }
                if stats == "published":
                    learner.adopt(heard)
                else:
                    learner.absorb(heard, tier)
        # The user reads the answer, pauses, and sends the conversation's next message.
        if after_answer and setup.follow[index] is not None:
            successor, pause_s = setup.follow[index]
            heapq.heappush(pending, ((hour * 3600 + took + pause_s) / 3600, successor))

    def land(completions):
        """Settle every query the OLT has just finished.

        Args:
            completions: [(query id, clock, prefill joules, decode joules)] from the server.
        """
        for query_id, done_at, prefill_joules, decode_joules in completions:
            query = in_flight.pop(query_id)
            answer = setup.answers[query["question"]][top]
            # What this query cost the network per token: its share of the batches it ran
            # in under average accounting, what it added under marginal accounting.
            if prices.accounting == "marginal":
                cost = prices.added_rates(query["running"])
            else:
                cost = (
                    prefill_joules / answer["tp"] if answer["tp"] else 0.0,
                    decode_joules / answer["tg"] if answer["tg"] else 0.0,
                )
            if query["belief"] is not None:
                rate_errors.append(abs(query["belief"] - query["fresh"]) / query["fresh"])
            cell = seen.setdefault(setup.schedule_key(query["hour"]), [0.0, 0.0, 0])
            cell[0] += cost[0]
            cell[1] += cost[1]
            cell[2] += 1
            # The packet carries this query's cost, or the OLT's report over its window,
            # which is far steadier under marginal accounting.
            reported = server.reported_rate() if settings.report == "window" else cost
            if stats == "published":
                olt_own.record(answer["tg"], None)
                query["packet"][top] = (*reported, olt_own)
            else:
                query["packet"][top] = (*reported, answer["tg"])
            settle(
                query["index"],
                query["hour"],
                top,
                query["spent"] + prefill_joules + decode_joules,
                query["took"] + (done_at - query["reached"]),
                query["packet"],
            )

    def hand_over(until):
        """Bring the OLT up to `until`, admitting what reaches it and settling what finishes.

        An answer it finishes can queue a follow-up message sent before `until`, and the OLT
        must not run past that message, so it stops at each answer to check.

        Returns:
            Whether it got to `until`, rather than stopping at an earlier message.
        """
        while True:
            stop = min(until, pending[0][0] * 3600) if pending else until
            if arrivals and arrivals[0][0] <= stop:
                reached, query_id = heapq.heappop(arrivals)
                land(server.advance(reached))
                query = in_flight[query_id]
                answer = setup.answers[query["question"]][top]
                query["running"] = server.batch
                query["fresh"] = server.reported_rate()[1]  # what the OLT itself says now
                server.admit(query_id, answer["tp"], answer["tg"])
            elif after_answer and server.next_finish() <= stop:
                land(server.advance(server.next_finish()))
            else:
                land(server.advance(stop))
                return stop >= until

    while pending or (after_answer and (arrivals or server.next_finish() < math.inf)):
        if not pending:  # only answers still on their way, which may queue more messages
            hand_over(min(arrivals[0][0] if arrivals else math.inf, server.next_finish()))
            continue
        hour, index = pending[0]
        question, _, household = setup.stream[index]
        now = hour * 3600

        # --- Bring the OLT up to now, stopping at the last broadcast on the way ---
        if policy == "broadcast":
            interval = settings.broadcast_interval_s
            last_send = math.floor(now / interval) * interval
            if last_send >= next_broadcast:
                if not hand_over(last_send):
                    continue  # an answer on the way queued an earlier message
                server.send_broadcast()
                next_broadcast = last_send + interval
        if not hand_over(now):
            continue
        heapq.heappop(pending)

        # --- Walk the query up until it is answered, or until it leaves for the OLT ---
        if household not in households:
            households[household] = {
                tier: Learned(
                    settings.alpha,
                    settings.warmup,
                    settings.rate_alpha,
                    pool[tier] if pool else None,
                    escalation,
                )
                for tier in TIERS[:TOP]
            }
            if stats == "published":
                for learner in households[household].values():
                    learner.ship(shipped_tokens, beta)
                own[household] = {
                    tier: OwnStats(settings.alpha, shipped_tokens[tier]) for tier in TIERS[:TOP]
                }
        learners = households[household]
        prompt_tokens = setup.answers[question]["user"]["tp"]
        packet: dict[str, tuple[float, float, float]] = {}
        spent = took = 0.0
        here = ALONE.get(policy, 0)
        rates = belief = None
        while True:
            tier = TIERS[here]
            learned = learners.get(tier)

            # --- Can this tier decide where the query goes? ---
            # Not under a fixed chain, not at the OLT, not before it has heard enough.
            decide = policy not in FIXED and learned is not None and learned.ready(TIERS[here:])
            if decide:
                # A household knows its own devices' rates; only the OLT's must reach it.
                if stats != "household":
                    rates = {upper: prices.rates(upper, 1) for upper in TIERS[here:TOP]}
                    rates["olt"] = learned.rates.get("olt")
                else:
                    rates = {upper: learned.rates[upper] for upper in TIERS[here:]}
                if policy != "piggyback":
                    rates["olt"] = believed_olt_rate(policy, setup, server, learned, hour)
                decide = rates["olt"] is not None
                if decide:
                    belief = rates["olt"][1]

            # --- On arrival: run here, or forward to a cheaper tier above? ---
            if decide:
                target = on_arrival(here, prompt_tokens, rates, learned, settings.delta)
                if target != here:
                    forwarded += 1
                    here = target
                    continue

            # --- The OLT answers in its own time, so the query leaves the walk here ---
            if here == TOP:
                in_flight[next_id] = {
                    "index": index,
                    "question": question,
                    "hour": hour,
                    "spent": spent,
                    "took": took,
                    "packet": packet,
                    "reached": now + took,
                    "belief": belief,
                }
                heapq.heappush(arrivals, (now + took, next_id))
                next_id += 1
                break

            # --- A tier below answers at once: replay it, charge the true energy ---
            answer = setup.answers[question][tier]
            prompt_rate, generated_rate = prices.rates(tier, 1)
            spent += prompt_rate * answer["tp"] + generated_rate * answer["tg"]
            took += prices.seconds(tier, 1, answer["tp"], answer["tg"])
            packet[tier] = (prompt_rate, generated_rate, answer["tg"])

            # --- RecServe's test: escalate below the beta-quantile of recent confidences ---
            window = window_for(household, tier)
            escalate = (
                policy not in ALONE and len(window) > 1 and answer["conf"] < window.quantile(beta)
            )
            window.add(answer["conf"])
            if stats == "published":
                own[household][tier].record(answer["tg"], escalate)
                packet[tier] = (prompt_rate, generated_rate, own[household][tier])
            if not escalate:
                settle(index, hour, tier, spent, took, packet)
                break

            # --- On escalation: the next tier up, or a cheaper later one? ---
            target = (
                on_escalation(here, prompt_tokens, rates, learned, settings.delta)
                if decide
                else here + 1
            )
            if policy == "recserve_no_onu":
                target = TOP
            skipped += target > here + 1
            here = target

    # --- Nobody else is coming: let the OLT take and finish what is left ---
    hand_over(max((reached for reached, _ in arrivals), default=server.clock))
    land(server.drain())

    total = len(setup.stream)
    metrics = {
        "accuracy": correct / total,
        "mean_confidence": confidence_delivered / total,
        "comm_MB_per_1k_queries": comm_bytes / total * 1000 / 1e6,  # RecServe's metric
        "pon_MB_per_1k_queries": pon_bytes / total * 1000 / 1e6,  # bytes over the shared fibre
        "latency_s_mean": statistics.mean(latencies),
        "latency_s_p95": float(np.percentile(latencies, 95)),
        "J_per_query": energy / total,
        "forwarded_on_arrival": forwarded / total,
        "skipped_on_escalation": skipped / total,
        # how far the OLT rate a query was routed on was from the OLT's own report when
        # the query reached it: how stale or wrong the policy's information was
        "olt_rate_error": statistics.mean(rate_errors) if rate_errors else float("nan"),
        "mean_olt_batch": server.mean_batch,
        "olt_queued": server.queued / server.admitted if server.admitted else 0.0,
        **{f"final_{tier}": answered_at[tier] / total for tier in TIERS},
    }
    per_hour = [
        {
            "hour": hour_of_day,
            "J_per_query": row[0] / row[1] if row[1] else None,
            "queries": row[1],
            **{
                f"final_{tier}": (row[2 + index] / row[1] if row[1] else None)
                for index, tier in enumerate(TIERS)
            },
        }
        for hour_of_day, row in enumerate(hourly)
    ]
    observed = {
        cell: (prompt_sum / answers, generated_sum / answers, answers)
        for cell, (prompt_sum, generated_sum, answers) in seen.items()
    }
    return metrics, per_hour, observed


# ==========================================
# What a timetable is built from
# ==========================================
def calibrate(setup: RunSetup, beta: float, policy: str = "recserve") -> dict:
    """Return the OLT rates the static policies ship with, observed over the training days.

    The OLT's cost is made by the traffic sent to it, so it has to be watched. An operator
    with no energy-aware policy yet watches plain RecServe, the cascade climbing one tier
    at a time, and ships what the OLT charged over that month. Watching a static policy
    instead, with setup.static_rates as its tables, shows what the OLT charges the traffic
    that policy itself sends.

    Args:
        setup: A setup whose stream covers the calibration days.
        beta: RecServe's escalation quantile, the same the run will use.
        policy: The policy watched (one of POLICIES).

    Returns:
        {"static_day": rates, "static_hour": {cell: rates}}, or {} if the OLT answered
        nothing.
    """
    _, _, observed = run(setup, beta, policy)
    answers = sum(count for _, _, count in observed.values())
    if not answers:
        return {}
    return {
        "static_day": (
            sum(prompt * count for prompt, _, count in observed.values()) / answers,
            sum(generated * count for _, generated, count in observed.values()) / answers,
        ),
        "static_hour": {
            cell: (prompt, generated) for cell, (prompt, generated, _) in observed.items()
        },
    }


SELF_CALIBRATION_ROUNDS = 3
"""Times a static policy's tables are relearned from the traffic it sends itself."""


def self_consistent_tables(setup: RunSetup, beta: float, policy: str) -> dict:
    """Return a static policy's tables, relearned from the traffic the policy itself sends.

    Tables observed under RecServe make the OLT look dearer than it is once a static policy
    sends it more work, which it then makes cheaper. Starting from RecServe's, the tables
    are relearned over the calibration days while running the policy with them, as an
    operator would keep refreshing them, SELF_CALIBRATION_ROUNDS times.

    Args:
        setup: A setup whose stream covers the calibration days.
        beta: RecServe's escalation quantile, the same the run will use.
        policy: static_day or static_hour.

    Returns:
        {"static_day": rates, "static_hour": {cell: rates}}, as calibrate() returns them.
    """
    tables = calibrate(setup, beta)
    for _ in range(SELF_CALIBRATION_ROUNDS):
        if not tables:
            break
        tables = calibrate(dataclasses.replace(setup, static_rates=tables), beta, policy)
    return tables
