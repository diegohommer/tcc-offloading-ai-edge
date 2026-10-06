"""Tests for measuring conversation shapes in BurstGPT."""

import numpy as np
import pandas as pd
import pytest

from prepare_load_traces import burstgpt_sessions, drift_around_timetable, SESSION_GAP_S


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
