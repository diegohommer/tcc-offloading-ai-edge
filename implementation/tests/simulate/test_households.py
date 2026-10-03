"""Tests for the query stream the households send."""

# pylint: disable=redefined-outer-name

import pytest

from households import Households, load_sessions

QUESTIONS = [3, 5, 8, 13]


@pytest.fixture
def pair_stats():
    """Conversations of exactly two messages a minute apart, weekdays busier than weekends."""
    return {
        "requests_per_session_mean": 2.0,
        "requests_per_session_icdf": [2, 2, 2],
        "think_time_s_icdf": [60.0, 60.0, 60.0],
        "weekend_days_mod7": [5, 6],
        "sessions_per_day": 24.0,
        "starts_by_hour": {"weekday": [1.0] * 24, "weekend": [0.25] * 24},
    }


def test_stream_is_time_ordered_and_inside_the_simulated_days(pair_stats):
    """Every message falls in [first day, first day + days), sorted by time."""
    stream = Households(500, 1.0, 4.0, pair_stats).stream(QUESTIONS, 3, 7, seed=1)
    times = [hour for _, hour, _ in stream]
    assert times == sorted(times)
    assert 7 * 24 <= times[0]
    assert times[-1] < 10 * 24


def test_stream_draws_households_and_questions_from_the_given_ranges(pair_stats):
    """Households are 0..count-1 and questions come from the list given."""
    stream = Households(50, 1.0, 4.0, pair_stats).stream(QUESTIONS, 2, 0, seed=1)
    assert {household for _, _, household in stream} <= set(range(50))
    assert {question for question, _, _ in stream} <= set(QUESTIONS)


def test_stream_is_reproducible_from_its_seed(pair_stats):
    """The same seed gives the same stream; another seed gives another."""
    homes = Households(200, 1.0, 4.0, pair_stats)
    assert homes.stream(QUESTIONS, 2, 0, seed=4) == homes.stream(QUESTIONS, 2, 0, seed=4)
    assert homes.stream(QUESTIONS, 2, 0, seed=4) != homes.stream(QUESTIONS, 2, 0, seed=5)


def test_stream_sends_the_population_volume(pair_stats):
    """Over weekdays the messages match households x users x messages per user a day."""
    homes = Households(1000, 2.0, 3.0, pair_stats)
    stream = homes.stream(QUESTIONS, 5, 0, seed=2)  # days 0-4 are weekdays
    assert len(stream) == pytest.approx(5 * homes.messages_per_day, rel=0.03)


def test_stream_keeps_a_conversation_in_one_household(pair_stats):
    """Each two-message conversation comes from one household, one think time apart."""
    stream = Households(10**6, 1.0, 0.01, pair_stats).stream(QUESTIONS, 1, 0, seed=3)
    by_household = {}
    for _, hour, household in stream:
        by_household.setdefault(household, []).append(hour)
    pairs = [hours for hours in by_household.values() if len(hours) == 2]
    assert pairs
    for first, second in pairs:
        assert (second - first) * 3600 == pytest.approx(60.0)


def test_stream_follows_the_weekend_shape(pair_stats):
    """Weekend days, at a quarter of the weekday rate, carry about a quarter of the traffic."""
    homes = Households(2000, 1.0, 4.0, pair_stats)
    weekday = len(homes.stream(QUESTIONS, 1, 0, seed=6))
    weekend = len(homes.stream(QUESTIONS, 1, 5, seed=6))
    assert weekend / weekday == pytest.approx(0.25, rel=0.15)


def test_weekday_reads_the_trace_weekend(pair_stats):
    """Days are weekdays unless their index mod 7 is a weekend day."""
    homes = Households(1, 1.0, 1.0, pair_stats)
    assert homes.weekday(4)
    assert not homes.weekday(5)
    assert not homes.weekday(13)


def test_households_default_to_the_measured_conversations():
    """The measured BurstGPT shapes load and drive a stream."""
    stats = load_sessions()
    assert stats["requests_per_session_mean"] >= 1
    assert Households(100, 1.0, 3.6).stream(QUESTIONS, 1, 0, seed=1)
