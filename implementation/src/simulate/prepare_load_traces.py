#!/usr/bin/env python3
"""Measure the shape of a conversation in BurstGPT, for the household generator.

Writes data/load_traces/burstgpt_sessions.json. Downloads BurstGPT_3.csv (232 MB) once
into implementation/.cache/load_traces/.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # implementation/
CACHE = ROOT / ".cache" / "load_traces"
"""Where the raw download is kept."""

OUT = ROOT / "data" / "load_traces" / "burstgpt_sessions.json"
"""Where the conversation shapes are written."""

BURSTGPT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv"
"""BurstGPT_3 is the only release carrying a session id, which a household is built from."""

SESSION_GAP_S = 600
"""A pause longer than this opens a new burst, so one burst is one sitting at the keyboard.

It reproduces BurstGPT's own hourly message curve: resampled bursts give a correlation of
0.998 and a peak/trough of 51x against the measured 52x.
"""

ICDF_POINTS = 2000
"""Points of each stored inverse CDF, taken at bin midpoints."""


def burstgpt_sessions(trace: pd.DataFrame) -> dict:
    """Measure how bursts of conversation are shaped.

    A session is one conversation, the closest the trace comes to a user. Four shapes are
    kept: how many requests a burst holds, the pause between two of them (both as inverse
    CDFs), and how many bursts open in each hour of a weekday and of a weekend day.

    Args:
        trace: BurstGPT rows with Timestamp, Session ID and Log Type.

    Returns:
        The session statistics.
    """
    # --- Conversations only: the API log has no human pausing between requests ---
    chats = trace[trace["Log Type"] == "Conversation log"].copy()
    chats["Timestamp"] = chats["Timestamp"].astype(float)
    chats = chats.sort_values(["Session ID", "Timestamp"])
    start = chats["Timestamp"].min()
    span_days = (chats["Timestamp"].max() - start) / 86400

    # --- Split each conversation into bursts at a pause longer than SESSION_GAP_S ---
    pause = chats.groupby("Session ID")["Timestamp"].diff()
    opens = pause.isna() | (pause > SESSION_GAP_S)
    requests_per_burst = chats.groupby(opens.cumsum()).size()
    think_time = pause[(~opens) & (pause > 0)]

    # --- When bursts open, by hour of day, weekday or weekend ---
    opened_at = (chats["Timestamp"][opens.to_numpy()] - start).to_numpy()
    day = (opened_at // 86400).astype(int)
    hour = (opened_at // 3600 % 24).astype(int)
    opened_per_weekday = [int((day % 7 == weekday).sum()) for weekday in range(7)]
    weekend = sorted(int(weekday) for weekday in np.argsort(opened_per_weekday)[:2])
    on_weekend = np.isin(day % 7, weekend)
    starts_by_hour = {}
    for day_type, rows in (("weekday", ~on_weekend), ("weekend", on_weekend)):
        days_of_type = max(len(np.unique(day[rows])), 1)
        counts = np.bincount(hour[rows], minlength=24).astype(float) / days_of_type
        starts_by_hour[day_type] = [round(count, 3) for count in counts]

    quantiles = (np.arange(ICDF_POINTS) + 0.5) / ICDF_POINTS
    return {
        "source": BURSTGPT_URL,
        "log_type": "Conversation log",
        "session_gap_s": SESSION_GAP_S,
        "span_days": round(span_days, 1),
        "sessions": int(len(requests_per_burst)),
        "requests": int(len(chats)),
        "sessions_per_day": round(len(requests_per_burst) / span_days, 1),
        "requests_per_day": round(len(chats) / span_days, 1),
        "requests_per_session_mean": round(float(requests_per_burst.mean()), 3),
        "requests_per_session_icdf": [
            int(value) for value in requests_per_burst.quantile(quantiles)
        ],
        "think_time_s_icdf": [round(float(value), 1) for value in think_time.quantile(quantiles)],
        "weekend_days_mod7": weekend,
        "starts_by_hour": starts_by_hour,
    }


def main() -> int:
    """Download BurstGPT_3 if needed, measure its conversations and write the shapes.

    Returns:
        The process exit code.
    """
    raw = CACHE / "BurstGPT_3.csv"
    if not raw.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BURSTGPT_URL}", file=sys.stderr)
        urllib.request.urlretrieve(BURSTGPT_URL, raw)
    stats = burstgpt_sessions(pd.read_csv(raw, usecols=["Timestamp", "Session ID", "Log Type"]))
    with open(OUT, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=1)
    print(
        f"{stats['sessions']:,} conversations over {stats['span_days']} days, "
        f"{stats['sessions_per_day']}/day, {stats['requests_per_session_mean']} requests each"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
