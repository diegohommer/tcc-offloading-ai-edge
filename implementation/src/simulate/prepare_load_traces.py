#!/usr/bin/env python3
"""Measure the shape of a conversation, in BurstGPT and in WildChat.

BurstGPT (one Azure OpenAI provider serving enterprises and campuses) gives how the load
drifts. WildChat (people chatting with a free ChatGPT, from home, worldwide, 2023-2025)
gives when people chat in their own local time, how their conversations go and how
unevenly they use it, for everyone and for Brazil alone.

Writes data/load_traces/burstgpt_sessions.json, wildchat_sessions.json and
wildchat_brazil_sessions.json. Downloads BurstGPT_3.csv (232 MB) and WildChat-4.8M (15 GB,
of which only timing and user fields are read) once into implementation/.cache/load_traces/.
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))  # src/simulate
from local_time import LocalTime  # pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[2]  # implementation/
CACHE = ROOT / ".cache" / "load_traces"
"""Where the raw download is kept."""

OUT = ROOT / "data" / "load_traces" / "burstgpt_sessions.json"
"""Where the conversation shapes are written."""

WILDCHAT_OUT = {
    None: ROOT / "data" / "load_traces" / "wildchat_sessions.json",
    "Brazil": ROOT / "data" / "load_traces" / "wildchat_brazil_sessions.json",
}
"""Where the conversation shapes of WildChat's users are written: everyone's, and Brazil's."""

WILDCHAT_URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/"
WILDCHAT_FILES = [f"train-{index:05d}-of-00086.parquet" for index in range(86)]
"""WildChat-4.8M's parquet files (ODC-BY): 3.2M non-toxic conversations from April 2023 to
July 2025, each turn with its time, the user's country and state, and a hashed IP."""

WEEKLY_ACTIVE_WEEKS = 2
"""Weeks a WildChat user must show up in to count among the weekly-active users whose
spread of activity is measured, rather than a one-off visitor."""

BURSTGPT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv"
"""BurstGPT_3 is the only release carrying a session id, which a household is built from."""

SESSION_GAP_S = 600
"""A pause longer than this opens a new burst, so one burst is one sitting at the keyboard.

It reproduces BurstGPT's own hourly message curve: resampled bursts give a correlation of
0.998 and a peak/trough of 51x against the measured 52x.
"""

ICDF_POINTS = 2000
"""Points of each stored inverse CDF, taken at bin midpoints."""

BUSY_HOUR = 20.0
"""Expected bursts an hour from which a cell is used to measure drift: quieter hours are
mostly counting noise."""

WEEK_DAYS = 7
"""Days each week's overall level is averaged over, to set slow growth apart from surges."""


def drift_around_timetable(counts: np.ndarray, weekend: np.ndarray, level_days: int) -> dict:
    """Measure how far and for how long load departs from an hour-of-day timetable.

    The load is read as a timetable times a log-normal factor with mean 1, and the factor's
    log-sd (sigma) and correlation time are measured net of Poisson counting noise:
    E[(count / expected - 1)^2] = (exp(sigma^2) - 1) + 1 / expected.

    Args:
        counts: Bursts opened, one row per day, one column per hour.
        weekend: Whether each day is a weekend day.
        level_days: 0 to keep one timetable for the whole trace, or the days each week's
            level is averaged over, so that slow growth is not counted as drift.

    Returns:
        {"sigma": log-sd of the factor, "hours": its correlation time}, both None when the
        trace has too few busy hours, or too little drift, to measure.
    """
    expected = np.zeros_like(counts)
    for weekend_day in (False, True):
        rows = weekend == weekend_day
        if rows.any():
            expected[rows] = counts[rows].mean(axis=0)
    if level_days:
        daily = counts.sum(axis=1) / np.maximum(expected.sum(axis=1), 1e-9)
        level = pd.Series(daily).rolling(level_days, center=True, min_periods=1).mean()
        expected = expected * level.to_numpy()[:, None]

    # --- Size: the factor's variance, net of counting noise ---
    busy = (expected >= BUSY_HOUR).ravel()
    pairs = busy[:-1] & busy[1:]
    excess = (counts / np.maximum(expected, 1e-9) - 1).ravel()
    noise = 1 / np.maximum(expected, 1e-9).ravel()
    factor_variance = np.mean(excess[busy] ** 2 - noise[busy]) if pairs.any() else 0.0
    if factor_variance <= 0:
        return {"sigma": None, "hours": None}
    sigma_squared = np.log1p(factor_variance)

    # --- Duration: correlation of consecutive busy hours, carried to the log scale ---
    factor_correlation = np.mean(excess[:-1][pairs] * excess[1:][pairs]) / factor_variance
    log_correlation = np.log1p(factor_correlation * np.expm1(sigma_squared)) / sigma_squared
    return {
        "sigma": round(float(np.sqrt(sigma_squared)), 3),
        "hours": round(float(-1 / np.log(log_correlation)), 2),
    }


def burstgpt_sessions(trace: pd.DataFrame) -> dict:
    """Measure how bursts of conversation are shaped, and how their load drifts.

    A session is one conversation, the closest the trace comes to a user. Four shapes are
    kept: how many requests a burst holds, the pause between two of them (both as inverse
    CDFs), and how many bursts open in each hour of a weekday and of a weekend day. The
    drift of those openings around a timetable is measured twice: against one timetable for
    the whole trace, and with each week's level taken out.

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

    # --- How far, and for how long, the load departs from the timetable ---
    whole_days = int(day.max())
    per_hour = np.bincount(day * 24 + hour, minlength=(whole_days + 1) * 24)
    per_hour = per_hour[: whole_days * 24].reshape(whole_days, 24).astype(float)
    weekend_day = np.isin(np.arange(whole_days) % 7, weekend)
    drift = {
        "all": drift_around_timetable(per_hour, weekend_day, 0),
        "within_week": drift_around_timetable(per_hour, weekend_day, WEEK_DAYS),
    }

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
        "drift": drift,
    }


def wildchat_answers(folder: Path) -> pd.DataFrame:
    """Return when each WildChat user got each answer, in UTC, with where the user was.

    Only assistant turns carry a time: when the server received the full answer. A user is
    a hashed IP with its request header, as WildChat's own card suggests for linking
    conversations.

    Args:
        folder: Where WildChat-4.8M's parquet files are.

    Returns:
        One row per answer: user (an integer id), country, state, utc (naive).
    """
    ids: dict[str, int] = {}
    parts = []
    for path in sorted(glob.glob(str(folder / "*.parquet"))):
        table = pq.ParquetFile(path).read(
            columns=[
                "country",
                "state",
                "hashed_ip",
                "header",
                "conversation.list.element.role",
                "conversation.list.element.timestamp",
            ]
        )
        users = [
            ids.setdefault(ip + json.dumps(header, sort_keys=True), len(ids))
            for ip, header in zip(
                table["hashed_ip"].to_pylist(), table["header"].to_pylist(), strict=True
            )
        ]
        conversations = table["conversation"].combine_chunks()
        turns = conversations.flatten()
        parent = pc.list_parent_indices(conversations).to_numpy()
        answer = pc.and_(
            pc.equal(turns.field("role"), "assistant"), pc.is_valid(turns.field("timestamp"))
        )
        rows = parent[answer.to_numpy(zero_copy_only=False)]
        parts.append(
            pd.DataFrame(
                {
                    "user": np.asarray(users)[rows],
                    "country": table["country"].to_numpy(zero_copy_only=False)[rows],
                    "state": table["state"].to_numpy(zero_copy_only=False)[rows],
                    "utc": pc.filter(turns.field("timestamp"), answer).to_pandas().to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def clean_and_localize(answers: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Drop the automated traffic, place every answer at its user's local time.

    WildChat-4.8M's later months hold bursts of API-like traffic: in its spike months most
    answers come from "users" seen exactly once, against about 4% in an ordinary month, so
    users with a single answer are dropped. Answers whose time zone cannot be told
    (local_time.LocalTime) are dropped too.

    Args:
        answers: wildchat_answers()'s rows.
        country: Keep only this country's users, or None for everyone.

    Returns:
        The kept rows with "local", the naive local time, sorted by user and time.
    """
    if country is not None:
        answers = answers[answers["country"] == country]
    per_user = answers.groupby("user").size()
    answers = answers[answers["user"].map(per_user) >= 2].copy()
    mapper = LocalTime()
    pairs = answers[["country", "state"]].drop_duplicates()
    zone_of = {
        (place, state): mapper.zone(place, state)
        for place, state in zip(pairs["country"], pairs["state"], strict=True)
    }
    answers["zone"] = [
        zone_of[pair] for pair in zip(answers["country"], answers["state"], strict=True)
    ]
    answers = answers[answers["zone"].notna()]
    local = pd.Series(pd.NaT, index=answers.index, dtype="datetime64[us]")
    for zone, rows in answers.groupby("zone").groups.items():
        utc = pd.to_datetime(answers.loc[rows, "utc"]).dt.tz_localize("UTC")
        local.loc[rows] = utc.dt.tz_convert(zone).dt.tz_localize(None).to_numpy()
    answers["local"] = local
    return answers.sort_values(["user", "local"])


def activity_spread(per_user: pd.Series) -> dict:
    """Return how unevenly users chat, and the log-normal spread that matches it.

    A log-normal activity with log-sd sigma gives the busiest tenth of users a share
    1 - Phi(z_0.9 - sigma) of the messages, so sigma is read off the measured share.

    Args:
        per_user: Messages each user sent.

    Returns:
        {"users", "top_tenth_share", "sigma"}, the last two None without any user.
    """
    if per_user.sum() <= 0:
        return {"users": int(len(per_user)), "top_tenth_share": None, "sigma": None}
    ordered = per_user.sort_values(ascending=False)
    share = float(ordered.head(max(len(ordered) // 10, 1)).sum() / ordered.sum())
    normal = statistics.NormalDist()
    sigma = normal.inv_cdf(0.9) - normal.inv_cdf(1 - share)
    return {
        "users": int(len(ordered)),
        "top_tenth_share": round(share, 3),
        "sigma": round(sigma, 2),
    }


def wildchat_sessions(answers: pd.DataFrame) -> dict:
    """Measure WildChat conversations the way burstgpt_sessions() does BurstGPT's.

    Bursts split at SESSION_GAP_S as there, in each user's local time. The weekend is the
    calendar's (Saturday and Sunday, days 5 and 6 of a simulation whose day 0 is a Monday),
    though some countries rest on Friday. Drift stays BurstGPT's, one provider's users
    being closer to one OLT's than WildChat's, spread over the world. It adds how unevenly
    the users chat.

    Args:
        answers: clean_and_localize()'s rows.

    Returns:
        The session statistics, in burstgpt_sessions()'s format plus "activity".
    """
    pause = answers.groupby("user")["local"].diff().dt.total_seconds()
    opens = pause.isna() | (pause > SESSION_GAP_S)
    requests_per_burst = answers.groupby(opens.cumsum().to_numpy()).size()
    think_time = pause[(~opens) & (pause > 0)]
    span_days = (answers["local"].max() - answers["local"].min()).total_seconds() / 86400

    starts = answers["local"][opens.to_numpy()]
    on_weekend = (starts.dt.dayofweek >= 5).to_numpy()
    starts_by_hour = {}
    for day_type, rows in (("weekday", ~on_weekend), ("weekend", on_weekend)):
        days_of_type = max(starts[rows].dt.normalize().nunique(), 1)
        counts = np.bincount(starts[rows].dt.hour, minlength=24).astype(float) / days_of_type
        starts_by_hour[day_type] = [round(count, 3) for count in counts]

    weeks = answers.groupby("user")["local"].apply(lambda times: times.dt.to_period("W").nunique())
    per_user = answers.groupby("user").size()
    quantiles = (np.arange(ICDF_POINTS) + 0.5) / ICDF_POINTS
    return {
        "source": WILDCHAT_URL,
        "session_gap_s": SESSION_GAP_S,
        "span_days": round(span_days, 1),
        "users": int(len(per_user)),
        "sessions": int(len(requests_per_burst)),
        "requests": int(len(answers)),
        "sessions_per_day": round(len(requests_per_burst) / span_days, 1),
        "requests_per_day": round(len(answers) / span_days, 1),
        "requests_per_session_mean": round(float(requests_per_burst.mean()), 3),
        "requests_per_session_icdf": [
            int(value) for value in requests_per_burst.quantile(quantiles)
        ],
        "think_time_s_icdf": [round(float(value), 1) for value in think_time.quantile(quantiles)],
        "weekend_days_mod7": [5, 6],
        "starts_by_hour": starts_by_hour,
        "activity": {
            "all": activity_spread(per_user),
            "weekly_active": activity_spread(per_user[weeks >= WEEKLY_ACTIVE_WEEKS]),
        },
    }


def main() -> int:
    """Download the traces if needed, measure their conversations and write the shapes.

    Returns:
        The process exit code.
    """
    folder = CACHE / "wildchat48"
    folder.mkdir(parents=True, exist_ok=True)
    for name in WILDCHAT_FILES:
        if not (folder / name).exists():
            print(f"downloading {WILDCHAT_URL}{name}", file=sys.stderr)
            urllib.request.urlretrieve(WILDCHAT_URL + name, folder / name)
    answers = wildchat_answers(folder)
    for country, out in WILDCHAT_OUT.items():
        stats = {
            "country": country or "all",
            **wildchat_sessions(clean_and_localize(answers, country)),
        }
        with open(out, "w", encoding="utf-8") as file:
            json.dump(stats, file, indent=1)
        print(
            f"WildChat {stats['country']}: {stats['users']:,} users, {stats['sessions']:,} "
            f"conversations over {stats['span_days']} days, {stats['requests_per_session_mean']} "
            f"requests each; activity sigma {stats['activity']['weekly_active']['sigma']} "
            "(weekly active)"
        )

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
    for name, fit in stats["drift"].items():
        print(f"drift around the timetable ({name}): sigma {fit['sigma']}, {fit['hours']} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
