"""Tests for turning a continuous-batching run's timings into energy figures."""

import pytest

from measure_gpu_energy_continuous import (
    mean_in_flight,
    poisson_arrivals,
    SERVICE_S,
    summarize_window,
    window_tokens,
)


def test_poisson_arrivals_produce_the_target_load():
    """Arrivals at load L come at L / service time per second, inside the duration."""
    arrivals = poisson_arrivals(load=4.0, duration_s=20_000.0, pool_size=1319, seed=1)
    assert len(arrivals) / 20_000.0 == pytest.approx(4.0 / SERVICE_S, rel=0.03)
    assert all(0 <= offset < 20_000.0 for offset, _ in arrivals)
    assert all(0 <= index < 1319 for _, index in arrivals)


def test_poisson_arrivals_are_reproducible_and_ordered():
    """The same seed gives the same arrivals, in time order."""
    arrivals = poisson_arrivals(2.0, 600.0, 100, seed=5)
    assert arrivals == poisson_arrivals(2.0, 600.0, 100, seed=5)
    assert [offset for offset, _ in arrivals] == sorted(offset for offset, _ in arrivals)


def test_mean_in_flight_weights_requests_by_their_time_in_the_window():
    """One request spanning the window and one covering half of it average 1.5."""
    assert mean_in_flight([(0.0, 20.0), (5.0, 10.0)], start=5.0, end=15.0) == pytest.approx(1.5)


def test_window_tokens_counts_only_events_inside_the_window():
    """Tokens streamed before the window opens or after it closes are left out."""
    events = [(1.0, 100, 0), (2.0, 0, 5), (3.0, 0, 7), (9.0, 0, 4)]
    assert window_tokens(events, start=2.0, end=9.0) == (0, 12)


def test_summarize_window_divides_energy_by_the_tokens_generated():
    """Gross and net energy per generated token over the measured window."""
    row = summarize_window(
        level=2,
        events=[(1.0, 50, 0), (1.5, 0, 100), (2.5, 0, 100)],
        spans=[(0.0, 3.0), (0.0, 3.0)],
        window=(1.0, 3.0),
        joules=100.0,
        idle_watts=10.0,
    )
    assert row["generated_tokens"] == 200
    assert row["prompt_tokens"] == 50
    assert row["mean_in_flight"] == pytest.approx(2.0)
    assert row["J_per_generated_token"] == pytest.approx(0.5)
    assert row["J_per_generated_token_net"] == pytest.approx((100.0 - 20.0) / 200)
    assert row["mean_power_W"] == pytest.approx(50.0)
