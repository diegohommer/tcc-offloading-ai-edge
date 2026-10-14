"""Tests for measuring conversation shapes in BurstGPT and WildChat."""

import numpy as np
import pandas as pd
import pytest

from prepare_load_traces import (
    activity_spread,
    burstgpt_sessions,
    clean_and_localize,
    drift_around_timetable,
    SESSION_GAP_S,
    wildchat_sessions,
)


def _trace(rows):
    """Return a BurstGPT-like frame from (seconds, session id, log type) rows."""
    return pd.DataFrame(rows, columns=["Timestamp", "Session ID", "Log Type"])


def test_burstgpt_sessions_splits_a_conversation_at_a_long_pause():
    """Two requests a minute apart form one burst; one after a long pause opens another."""
    stats = burstgpt_sessions(
        _trace(
            [
                (0, 1, "Conversation log"),
                (60, 1, "Conversation log"),
                (60 + SESSION_GAP_S + 1, 1, "Conversation log"),
                (86400 * 7, 2, "Conversation log"),
            ]
        )
    )
    assert stats["sessions"] == 3
    assert stats["requests"] == 4
    assert stats["requests_per_session_mean"] == pytest.approx(4 / 3, abs=1e-3)
    assert set(stats["think_time_s_icdf"]) == {60.0}


def test_burstgpt_sessions_ignores_the_api_log():
    """API requests have no person pausing between them, so they are left out."""
    stats = burstgpt_sessions(
        _trace(
            [
                (0, 1, "Conversation log"),
                (30, 1, "Conversation log"),
                (10, 2, "API log"),
                (86400, 3, "Conversation log"),
            ]
        )
    )
    assert stats["requests"] == 3


def test_burstgpt_sessions_counts_bursts_by_hour_since_the_trace_starts():
    """Hours are counted from the trace's first request, which BurstGPT starts at midnight."""
    rows = [(0, 0, "Conversation log")]
    rows += [(86400 * day + 13 * 3600, day, "Conversation log") for day in range(1, 15)]
    stats = burstgpt_sessions(_trace(rows))
    weekday_starts = stats["starts_by_hour"]["weekday"]
    assert weekday_starts[13] > 0
    assert sum(weekday_starts) == pytest.approx(weekday_starts[0] + weekday_starts[13])
    assert len(stats["weekend_days_mod7"]) == 2


def _timetable_with_drift(sigma, hours, days=60, level=200.0, seed=4):
    """Return Poisson counts around a flat timetable, times a drifting log-normal factor."""
    rng = np.random.default_rng(seed)
    persistence = np.exp(-1.0 / hours)
    drift = np.empty(days * 24)
    drift[0] = rng.normal(0.0, sigma)
    for hour in range(1, days * 24):
        drift[hour] = persistence * drift[hour - 1] + rng.normal(
            0.0, sigma * np.sqrt(1 - persistence**2)
        )
    rate = level * np.exp(drift - sigma**2 / 2)
    return rng.poisson(rate).reshape(days, 24).astype(float), np.zeros(days, dtype=bool)


def test_drift_around_timetable_recovers_a_known_drift():
    """Counts drawn with sigma 0.4 over 6 hours measure back close to both, despite the noise."""
    counts, weekend = _timetable_with_drift(0.4, 6.0)
    fit = drift_around_timetable(counts, weekend, level_days=0)
    assert fit["sigma"] == pytest.approx(0.4, rel=0.2)
    assert fit["hours"] == pytest.approx(6.0, rel=0.35)


def test_drift_around_timetable_of_pure_counting_noise_is_none():
    """Poisson counts around the timetable, with no drift, measure no drift."""
    rng = np.random.default_rng(1)
    counts = rng.poisson(200.0, size=(60, 24)).astype(float)
    fit = drift_around_timetable(counts, np.zeros(60, dtype=bool), level_days=0)
    assert fit["sigma"] is None or fit["sigma"] < 0.05


def _answers(rows):
    """Return wildchat_answers()-like rows from (user, local time) pairs."""
    frame = pd.DataFrame(rows, columns=["user", "local"])
    frame["local"] = pd.to_datetime(frame["local"])
    frame["state"] = "São Paulo"
    return frame.sort_values(["user", "local"])


def test_wildchat_sessions_splits_bursts_and_reads_the_calendar_weekend():
    """A user's answers a minute apart form one burst; a Saturday burst counts as weekend."""
    stats = wildchat_sessions(
        _answers(
            [
                ("a", "2023-05-01 21:00:00"),  # Monday
                ("a", "2023-05-01 21:01:00"),
                ("a", "2023-05-01 23:00:00"),
                ("b", "2023-05-06 10:30:00"),  # Saturday
            ]
        )
    )
    assert stats["sessions"] == 3
    assert stats["weekend_days_mod7"] == [5, 6]
    assert stats["starts_by_hour"]["weekday"][21] == 1.0
    assert stats["starts_by_hour"]["weekday"][23] == 1.0
    assert stats["starts_by_hour"]["weekend"][10] == 1.0
    assert set(stats["think_time_s_icdf"]) == {60.0}


def test_activity_spread_reads_sigma_off_the_busiest_tenth():
    """Users drawn log-normal with a known spread give that spread back."""
    rng = np.random.default_rng(1)
    messages = pd.Series(np.exp(rng.normal(0.0, 1.2, 200_000)))
    assert activity_spread(messages)["sigma"] == pytest.approx(1.2, abs=0.03)
    assert activity_spread(pd.Series([1.0] * 100))["sigma"] == pytest.approx(0.0, abs=1e-6)


def test_clean_and_localize_drops_one_shot_users_and_keeps_summer_time():
    """A user seen once is dropped; a Californian in July is seven hours behind UTC."""
    answers = pd.DataFrame(
        {
            "user": [1, 1, 2, 3, 3],
            "country": ["United States"] * 3 + ["Brazil"] * 2,
            "state": ["California", "California", "Texas", "Acre", "Acre"],
            "utc": pd.to_datetime(
                [
                    "2024-07-01 12:00",
                    "2024-07-01 12:01",
                    "2024-07-01 12:00",
                    "2024-07-01 12:00",
                    "2024-07-01 12:02",
                ]
            ),
        }
    )
    everyone = clean_and_localize(answers)
    assert set(everyone["user"]) == {1, 3}
    assert everyone.loc[everyone["user"] == 1, "local"].iloc[0] == pd.Timestamp("2024-07-01 05:00")
    assert everyone.loc[everyone["user"] == 3, "local"].iloc[0] == pd.Timestamp("2024-07-01 07:00")
    assert set(clean_and_localize(answers, "Brazil")["user"]) == {3}
