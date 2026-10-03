#!/usr/bin/env python3
"""Count hourly LLM requests in public traces: the OLT's load for simulate.py --load-trace.

Writes data/load_traces/ (BurstGPT's hourly counts per log type, optionally the Azure 2024
conversation trace, and the drift fitted around an hour-of-day schedule). What each trace
and column holds: data/load_traces/README.md.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # implementation/
CACHE = ROOT / ".cache" / "load_traces"
"""Where the raw downloads are kept."""

OUT = ROOT / "data" / "load_traces"
"""Where the prepared counts are written."""

BURSTGPT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv"
BURSTGPT3_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv"
"""BurstGPT_3 is the only release carrying a Session ID, which is what a household is built from."""
SESSION_GAP_S = 600
"""A pause longer than this opens a new burst, so one burst is one sitting at the keyboard.

Chosen as the value that reproduces BurstGPT's own hourly message curve: resampled bursts
give a correlation of 0.998 and a peak/trough of 51x against the measured 52x. The 30-minute
timeout conventional in web analytics gives nearly the same (0.994, 45x); leaving conversation
ids uncut gives 12x, filling a night the trace shows as empty.
"""
ICDF_POINTS = 2000
"""Points of each stored inverse CDF: enough to resample the measured shape without the raw rows.

Taken at the midpoint of each bin rather than at 0 and 1, so the heaviest conversation in the
trace does not get 1/ICDF_POINTS of the weight it would need 1/sessions of.
"""
AZURE_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
    "AzureLLMInferenceTrace_conv_1week.csv"
)
BURSTGPT_DAYS = 61


def burstgpt_hourly() -> pd.DataFrame:
    """Count BurstGPT's requests per hour over its 61 days, for conversation, api and all traffic.

    Downloads BurstGPT_1.csv once into implementation/.cache/load_traces/.

    Returns:
        One row per hour: day, hour, conversation, api, all.
    """
    raw = CACHE / "BurstGPT_1.csv"
    if not raw.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BURSTGPT_URL}", file=sys.stderr)
        urllib.request.urlretrieve(BURSTGPT_URL, raw)
    df = pd.read_csv(raw, usecols=["Timestamp", "Log Type"])
    n = BURSTGPT_DAYS * 24
    cols = {}
    for name, sub in (
        ("conversation", df[df["Log Type"] == "Conversation log"]),
        ("api", df[df["Log Type"] == "API log"]),
        ("all", df),
    ):
        h = (sub["Timestamp"].to_numpy() // 3600).astype(int)
        cols[name] = np.bincount(h[h < n], minlength=n)
    idx = np.arange(n)
    return pd.DataFrame({"day": idx // 24, "hour": idx % 24, **cols})


def burstgpt_sessions() -> dict:
    """Measure the shape of one burst of conversation in BurstGPT, for the household generator.

    A session is one conversation, the closest the trace comes to a user: it has no user id.
    Three shapes are kept, each as an inverse CDF so the simulator resamples the measured
    distribution instead of a fitted one: how many requests a conversation holds, how long
    its author pauses between two of them, and when conversations start over the day.

    Downloads BurstGPT_3.csv (232 MB) once into implementation/.cache/load_traces/.

    Returns:
        The session statistics, written to data/load_traces/burstgpt_sessions.json.
    """
    raw = CACHE / "BurstGPT_3.csv"
    if not raw.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BURSTGPT3_URL}", file=sys.stderr)
        urllib.request.urlretrieve(BURSTGPT3_URL, raw)
    df = pd.read_csv(raw, usecols=["Timestamp", "Session ID", "Log Type"])

    # --- Conversations only: the API log has no human pausing between requests ---
    c = df[df["Log Type"] == "Conversation log"].copy()
    c["Timestamp"] = c["Timestamp"].astype(float)
    c = c.sort_values(["Session ID", "Timestamp"])
    start = c["Timestamp"].min()
    span_days = (c["Timestamp"].max() - start) / 86400

    # --- Split each conversation into bursts at a pause longer than SESSION_GAP_S ---
    # A conversation id can span a whole day: someone asks, leaves, and comes back after
    # dinner. Resampling that pause as it stands would deliver messages at 4 am, which the
    # trace never shows. Cutting at SESSION_GAP_S keeps a burst to one sitting.
    gap = c.groupby("Session ID")["Timestamp"].diff()
    opens = gap.isna() | (gap > SESSION_GAP_S)
    burst = opens.cumsum()

    # --- Requests per burst, and the pause between two of them ---
    per_session = c.groupby(burst).size()
    gap = gap[(~opens) & (gap > 0)]

    # --- When bursts open, by hour of day, relative to the busiest hour ---
    first = c["Timestamp"][opens.to_numpy()]
    by_hour = np.bincount(((first - start) // 3600 % 24).astype(int), minlength=24).astype(float)

    q = (np.arange(ICDF_POINTS) + 0.5) / ICDF_POINTS
    return {
        "source": BURSTGPT3_URL,
        "log_type": "Conversation log",
        "session_gap_s": SESSION_GAP_S,
        "span_days": round(span_days, 1),
        "sessions": int(len(per_session)),
        "requests": int(len(c)),
        "sessions_per_day": round(len(per_session) / span_days, 1),
        "requests_per_day": round(len(c) / span_days, 1),
        "requests_per_session_mean": round(float(per_session.mean()), 3),
        "requests_per_session_icdf": [int(v) for v in per_session.quantile(q)],
        "think_time_s_icdf": [round(float(v), 1) for v in gap.quantile(q)],
        "starts_by_hour": [round(float(v), 4) for v in by_hour / by_hour.max()],
    }


def azure_hourly() -> pd.DataFrame:
    """Count the Azure 2024 conversation trace's requests per UTC hour, streaming its 1.1 GB once.

    Returns:
        One row per hour of the whole days covered: day, hour, date, conversation.
    """
    counts: collections.Counter = collections.Counter()
    print(f"streaming {AZURE_URL}", file=sys.stderr)
    with urllib.request.urlopen(AZURE_URL) as f:
        next(f)  # header
        for line in f:
            counts[line[:13]] += 1  # b"YYYY-MM-DD HH"
    hours = {dt.datetime.strptime(k.decode(), "%Y-%m-%d %H"): v for k, v in counts.items()}
    first = min(hours).replace(hour=0) + dt.timedelta(days=1) if min(hours).hour else min(hours)
    last = max(hours).replace(hour=0)  # exclusive: the last partial day is dropped
    rows, t = [], first
    while t < last:
        rows.append(
            {
                "day": (t - first).days,
                "hour": t.hour,
                "date": t.date().isoformat(),
                "conversation": hours.get(t, 0),
            }
        )
        t += dt.timedelta(hours=1)
    return pd.DataFrame(rows)


def schedule(c: np.ndarray, days: np.ndarray, weekend: set | None) -> np.ndarray:
    """Return the mean count per hour of day (per day type if weekend is given).

    Args:
        c: Counts, one row per day, one column per hour.
        days: The rows (days) to average over.
        weekend: The weekend days (index mod 7), or None for one schedule for every day.

    Returns:
        The schedule's prediction for every day and hour, shaped like c.
    """
    if weekend is None:
        return np.broadcast_to(c[days].mean(axis=0), c.shape)
    we = np.array([d % 7 in weekend for d in range(len(c))])
    sel_we, sel_wd = [d for d in days if we[d]], [d for d in days if not we[d]]
    return np.where(we[:, None], c[sel_we].mean(axis=0), c[sel_wd].mean(axis=0))


def drift(c: np.ndarray, pred: np.ndarray, rows: np.ndarray) -> dict:
    """Measure the drift of the counts around a schedule, log(count / schedule).

    Args:
        c: Counts, one row per day, one column per hour.
        pred: The schedule's prediction, shaped like c.
        rows: The days to measure on.

    Returns:
        sigma (log-sd net of Poisson counting noise), tau (lag where the autocorrelation
        falls below 1/e), a few autocorrelations and the load/schedule percentiles.
    """
    cc, pp = np.maximum(c[rows], 0.5), np.maximum(pred[rows], 0.5)
    r = np.log(cc / pp).ravel()
    poisson = np.mean(1.0 / np.maximum(c[rows].ravel(), 1))
    x = r - r.mean()
    acf = [float(np.sum(x[:-k] * x[k:]) / np.sum(x * x)) for k in range(1, 97)]
    tau = next((k + 1 for k, a in enumerate(acf) if a < np.exp(-1)), None)
    q = np.exp(r)
    return {
        "sigma": float(np.sqrt(max(r.var() - poisson, 0.0))),
        "sigma_raw": float(r.std()),
        "tau_h": tau,
        "acf": {str(k): round(acf[k - 1], 3) for k in (1, 3, 6, 12, 24, 48)},
        "load_over_schedule_p10_p50_p90": [
            round(float(v), 3) for v in np.percentile(q, [10, 50, 90])
        ],
    }


def fit(c: np.ndarray, train_days: int, weekly: bool) -> dict:
    """Describe one trace: its size, daily swing, and drift around a schedule.

    Args:
        c: Counts, one row per day, one column per hour.
        train_days: Days a schedule is fitted on for the out-of-sample drift (0 = none).
        weekly: Whether to also fit a weekday/weekend schedule.

    Returns:
        The description, one entry of drift_fit.json.
    """
    days = len(c)
    daily = c.sum(axis=1)
    weekend = None
    if weekly:
        dow = np.array([daily[d::7].mean() for d in range(7)])
        weekend = set(int(d) for d in np.argsort(dow)[:2])
    shape = c.mean(axis=0)
    out = {
        "days": days,
        "mean_per_hour": float(c.mean()),
        "daily_cv": float(daily.std() / daily.mean()),
        "peak_over_trough": float(shape.max() / max(shape.min(), 1e-9)),
        "weekend_days_mod7": sorted(weekend) if weekend else None,
    }
    allrows = np.arange(days)
    for name, we in (("hour_of_day", None), ("hour_x_daytype", weekend)):
        if name == "hour_x_daytype" and not weekly:
            continue
        out[name] = {"in_sample": drift(c, schedule(c, allrows, we), allrows)}
        if train_days and train_days < days:
            tr, te = allrows[:train_days], allrows[train_days:]
            out[name]["out_of_sample"] = drift(c, schedule(c, tr, we), te)
    return out


def main() -> int:
    """Write data/load_traces/ (hourly counts and drift_fit.json) and print a summary.

    Returns:
        The process exit code.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--azure",
        action="store_true",
        help="also stream and count the Azure 2024 conversation trace",
    )
    ap.add_argument(
        "--train-days", type=int, default=30, help="BurstGPT days a schedule is fitted on"
    )
    ap.add_argument(
        "--sessions",
        action="store_true",
        help="also measure the shape of a conversation from BurstGPT_3 (232 MB download)",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- BurstGPT ---
    b = burstgpt_hourly()
    b.to_csv(OUT / "burstgpt_hourly.csv", index=False)
    result = {"burstgpt": {"source": BURSTGPT_URL, "train_days": args.train_days}}
    for col in ("conversation", "api", "all"):
        c = b[col].to_numpy().reshape(-1, 24).astype(float)
        result["burstgpt"][col] = fit(c, args.train_days, weekly=True)

    # --- Conversation shape, for the household generator (optional) ---
    if args.sessions:
        s = burstgpt_sessions()
        with open(OUT / "burstgpt_sessions.json", "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
        print(
            f"burstgpt sessions: {s['sessions']:,} conversations over {s['span_days']} days, "
            f"{s['sessions_per_day']}/day, {s['requests_per_session_mean']} requests each"
        )

    # --- Azure 2024 (optional) ---
    if args.azure:
        az = OUT / "azure2024_conv_hourly.csv"
        if az.exists():  # the 1.1 GB stream is only needed once
            a = pd.read_csv(az)
        else:
            a = azure_hourly()
            a.to_csv(az, index=False)
        c = a["conversation"].to_numpy().reshape(-1, 24).astype(float)
        result["azure2024_conv"] = {
            "source": AZURE_URL,
            "first_day_utc": a["date"].iloc[0],
            **fit(c, 0, weekly=False),
        }

    # --- Write and summarize ---
    with open(OUT / "drift_fit.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for src, r in result.items():
        for col, v in r.items():
            if not isinstance(v, dict) or "days" not in v:
                continue
            print(
                f"{src} {col}: {v['days']} days, {v['mean_per_hour']:.0f}/h, "
                f"peak/trough {v['peak_over_trough']:.1f}, "
                f"daily cv {v['daily_cv']:.2f}"
            )
            for sched in ("hour_of_day", "hour_x_daytype"):
                for k, d in v.get(sched, {}).items():
                    print(
                        f"   {sched:15s} {k:13s} sigma {d['sigma']:.2f}  tau {d['tau_h']} h  "
                        f"ACF 1/6/24 h {d['acf']['1']:.2f} {d['acf']['6']:.2f} {d['acf']['24']:.2f}  "
                        f"load/schedule p10/p50/p90 {d['load_over_schedule_p10_p50_p90']}"
                    )
        if isinstance(r.get("days"), int):
            v = r
            print(
                f"{src}: {v['days']} days, {v['mean_per_hour']:.0f}/h, "
                f"peak/trough {v['peak_over_trough']:.1f}, "
                f"daily cv {v['daily_cv']:.2f}"
            )
            d = v["hour_of_day"]["in_sample"]
            print(
                f"   hour_of_day     in_sample     sigma {d['sigma']:.2f}  tau {d['tau_h']} h  "
                f"ACF 1/6/24 h {d['acf']['1']:.2f} {d['acf']['6']:.2f} {d['acf']['24']:.2f}"
            )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
