"""One simulated run: a policy serving the whole query stream through the cascade.

Part of the simulator (simulate.py).

    POLICIES      every policy the simulator knows (see README.md for what each knows
                  about the OLT's cost).
    build_stream  the time-ordered queries arriving at the households' phones.
    run           one policy at one RecServe beta over that stream: accuracy, energy,
                  where queries ended up, and hour-by-hour detail.
"""
from __future__ import annotations

import collections
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # implementation/src
from energy.three_tier import TIERS  # noqa: E402
from olt_load import BURSTGPT  # noqa: E402
from routing import TOP, Learned, Window, on_arrival, on_escalation  # noqa: E402

# Every policy. All but the two fixed chains share one routing rule (routing.py) and
# differ ONLY in where they get the OLT's energy cost from:
#   recserve         RecServe as published: always one tier up. The baseline.
#   recserve_no_onu  RecServe on a two-tier chain, phone -> OLT. Control: is a saving
#                    just from dropping the ONU?
#   static_day       one fixed OLT rate for the whole day, calibrated in advance
#   static_hour      one OLT rate per hour of the day, weekday or weekend: a timetable
#   broadcast        the OLT's own 5-minute mean, broadcast on the PON to every ONU
#                    every few seconds: the proposal
#   oracle           the true expected rate right now: the most any live signal could do
#   piggyback        the OLT's 5-minute mean, heard only on the household's own answers
#                    (ablation: why broadcast)
#   stale_low/high   static_day calibrated at 1/4 or 4x the real load (a stale config)
POLICIES = ("recserve", "recserve_no_onu", "static_day", "stale_low", "stale_high", "static_hour", "piggyback", "broadcast",
            "oracle")
FIXED = ("recserve", "recserve_no_onu")              # fixed chains: no energy information at all
STATIC = ("static_day", "stale_low", "stale_high")   # one fixed OLT rate, shipped as configuration


def run(rec, stream, batches, loads, beta, policy, args, E, static_rates, reports=None, sched_key=None, bcast=None):
    """Serve every query of the stream under one policy and one RecServe beta.

    For each query: start at the phone; at each tier the policy may forward the
    query past it (on_arrival) before running it; the tier answers, and RecServe's
    beta-quantile test decides whether the answer is kept or the query escalates;
    on escalation the policy picks the next tier (on_escalation). Energy is
    charged at the true rates (E) and the batch this query met at the OLT. The
    answer travels back down carrying each tier's reported rate, from which the
    households learn (Learned.absorb).

    rec           answers by question index and tier (from energy.three_tier.load_answers)
    stream        [(question index, time in hours, household)], time-ordered (build_stream)
    batches       the OLT batch each query would meet, 1 + Poisson(load): the same for every policy
    loads         the OLT's true offered load at each query's arrival
    beta          RecServe's escalation quantile
    policy        one of POLICIES
    args          the run's settings (simulate.py)
    E             true energy rates (olt_energy.Energy)
    static_rates  the pre-calibrated OLT rates of static_day, stale_low/high, static_hour
    reports       the OLT's report at each query, for piggyback (--report window)
    sched_key     time -> static_hour's table key (day type, hour)
    bcast         the latest broadcast at each query, for broadcast

    Also measured, without affecting any decision: latency (compute seconds at every
    tier the query visited) and the bytes that cross the shared PON fibre.

    Returns (metrics over the whole stream, per-hour detail).
    """
    history = {t: Window(args.window) for t in TIERS}   # RecServe's thresholds: one per tier, shared by every household
    homes: dict[int, dict[str, Learned]] = {}           # per household, its tiers that can decide
    pool = ({t: Learned(args.alpha, args.warmup, args.rate_alpha) for t in TIERS[:TOP]}
            if args.shared_stats else None)             # question statistics pooled across households
    energy = correct = forwarded = skipped = conf_delivered = comm_bytes = pon_bytes = 0.0
    rate_err, latency = [], []
    final_at = collections.Counter()
    hourly = [[0.0, 0, 0, 0, 0] for _ in range(24)]      # joules, queries, final at user/onu/olt

    for n_q, ((qi, t_h, hh), b, load) in enumerate(zip(stream, batches, loads)):   # load: the OLT's true offered load now
        learners = homes.get(hh)
        if learners is None:
            learners = homes[hh] = {t: Learned(args.alpha, args.warmup, args.rate_alpha, pool[t] if pool else None)
                                    for t in TIERS[:TOP]}
        prompt = rec[qi]["user"]["tp"]
        packet: dict[str, tuple[float, float, int]] = {}
        spent = took = 0.0                          # joules and seconds this query costs
        i = 0
        rates = None
        while True:
            t = TIERS[i]
            L = learners.get(t)
            decide = policy not in FIXED and L is not None and L.ready(TIERS[i:])
            if decide:
                if pool:    # a household knows its own devices' rates; only the OLT's has to reach it
                    rates = {u: E.rates(u, 1) for u in TIERS[i:TOP]}
                    rates["olt"] = L.rates.get("olt")
                else:
                    rates = {u: L.rates[u] for u in TIERS[i:]}
                if policy in STATIC:
                    rates["olt"] = static_rates[policy]
                elif policy == "static_hour":
                    rates["olt"] = static_rates["static_hour"][sched_key(t_h)]
                elif policy == "oracle":
                    rates["olt"] = E.expected_olt(load)
                elif policy == "broadcast":
                    rates["olt"] = bcast[n_q]
                if policy in ("piggyback", "broadcast") and rates["olt"] is not None:
                    true_dec = E.expected_olt(load)[1]
                    rate_err.append(abs(rates["olt"][1] - true_dec) / true_dec)
                decide = rates["olt"] is not None       # piggyback: this household has not heard the OLT yet
            if decide:
                j = on_arrival(i, prompt, rates, L, args.delta)
                if j != i:
                    forwarded += 1
                    i = j
                    continue

            d = rec[qi][t]
            jpf, jdec = E.rates(t, b)
            spent += jpf * d["tp"] + jdec * d["tg"]
            took += E.seconds(t, b, d["tp"], d["tg"])
            packet[t] = (*(reports[n_q] if reports and t == "olt" else (jpf, jdec)), d["tg"])

            hist = history[t]
            escalate = i < TOP and len(hist) > 1 and d["conf"] < hist.quantile(beta)
            hist.add(d["conf"])
            if not escalate:
                correct += d["correct"]
                conf_delivered += d["conf"]      # confidence of the answer the user actually gets
                # RecServe's communication burden [1]: answered at tier i, the input is
                # uploaded and the output downloaded once per hop, 2(i-1)(|x|+|y|). Skipping a
                # tier does not save bytes -- the PON links are the same -- only inference.
                comm_bytes += 2 * i * (rec[qi]["user"]["qb"] + d["ab"])
                # Of those, what crosses the shared PON fibre: question up and answer down,
                # once, for every query the OLT answers.
                pon_bytes += (rec[qi]["user"]["qb"] + d["ab"]) if i == TOP else 0
                final_at[t] += 1
                break
            nxt = on_escalation(i, prompt, rates, L, args.delta) if decide else i + 1
            if policy == "recserve_no_onu":
                nxt = TOP
            skipped += nxt > i + 1
            i = nxt

        energy += spent
        latency.append(took)
        hr = hourly[int(t_h) % 24]
        hr[0] += spent
        hr[1] += 1
        hr[2 + TIERS.index(t)] += 1
        # The answer travels back down. Every deciding tier at or below where it
        # was produced reads the packet for itself and the tiers above it. The ONU
        # relays everything above the user, so it hears reports even for queries
        # whose inference it skipped.
        final = t
        for tier, Lt in learners.items():
            if TIERS.index(tier) <= TIERS.index(final):
                Lt.absorb({u: v for u, v in packet.items() if TIERS.index(u) >= TIERS.index(tier)}, final)

    n = len(stream)
    return {
        "accuracy": correct / n,
        "mean_confidence": conf_delivered / n,      # of the answer returned, whichever tier produced it
        "comm_MB_per_1k_queries": comm_bytes / n * 1000 / 1e6,   # RecServe's metric, on its terms
        "pon_MB_per_1k_queries": pon_bytes / n * 1000 / 1e6,     # bytes over the shared fibre
        "latency_s_mean": st.mean(latency),                      # compute time to the answer
        "latency_s_p95": float(np.percentile(latency, 95)),
        "J_per_query": energy / n,
        "forwarded_on_arrival": forwarded / n,
        "skipped_on_escalation": skipped / n,
        "olt_rate_error": st.mean(rate_err) if rate_err else float("nan"),
        **{f"final_{t}": final_at[t] / n for t in TIERS},
    }, [{"hour": h, "J_per_query": hr[0] / hr[1] if hr[1] else None, "queries": hr[1],
         **{f"final_{t}": (hr[2 + k] / hr[1] if hr[1] else None) for k, t in enumerate(TIERS)}}
        for h, hr in enumerate(hourly)]


def build_stream(indices, days, per_day, seed, households=1, trace=None, shape="trace"):
    """Time-ordered arrivals at the user tier: (question, hour, household).

    Without a trace every day is BurstGPT's average day: hour h gets a share of
    per_day x households proportional to BURSTGPT. With one, arrivals come over the
    trace's days after training, a Poisson number per hour, per_day x households a
    day on average. shape='trace': proportional to that hour's requests, so the
    households are busiest when the OLT is. shape='flat': the same number every
    hour, so the households' demand is independent of the OLT's load (a
    sensitivity check on that correlation). Each arrival belongs to a household
    drawn at random. Questions are drawn at random, with replacement.
    """
    rng = random.Random(seed)
    home = (lambda: rng.randrange(households)) if households > 1 else (lambda: 0)
    out = []
    if trace is None:
        total = sum(BURSTGPT)
        for day in range(days):
            times = []
            for h in range(24):
                times += [24 * day + h + rng.random()
                          for _ in range(round(per_day * households * BURSTGPT[h] / total))]
            out += [(rng.choice(indices), t, home()) for t in sorted(times)]
        return out
    nprng = np.random.default_rng([seed, 31])
    test = trace.c[trace.train_days:]
    per_count = per_day * households * len(test) / test.sum()
    for d, day in enumerate(test):
        for h, n in enumerate(day):
            if shape == "flat":
                n = test.mean()
            t0 = 24 * (trace.train_days + d) + h
            for t in sorted(t0 + nprng.random(nprng.poisson(per_count * n))):
                out.append((rng.choice(indices), float(t), home()))
    return out
