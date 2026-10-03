"""Tests for measuring conversation shapes in BurstGPT."""

import pandas as pd
import pytest

from prepare_load_traces import burstgpt_sessions, SESSION_GAP_S


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
