"""Tests for checking the simulator's batching against the measured GPU."""

import pytest

from check_continuous_batching import compare_run, curve_at_idle, off_by, replay
from energy.three_tier import olt_reference


def _run(requests, seconds, joules, idle_watts):
    """Return a Poisson run as the measurement writes it."""
    return {
        "target_load": 1.0,
        "seed": 7,
        "seconds": seconds,
        "joules": joules,
        "joules_net": joules - idle_watts * seconds,
        "mean_in_flight": 1.0,
        "requests": requests,
    }


def test_curve_at_idle_keeps_the_sweep_net_rates_at_its_own_idle():
    """Netting against the sweep's own idle power changes nothing."""
    sweep = olt_reference()
    rebuilt = curve_at_idle(sweep["idle_power_W"])
    assert rebuilt.dec1_net == pytest.approx(curve_at_idle(None).dec1_net)
    assert rebuilt.pf1_net == pytest.approx(curve_at_idle(None).pf1_net)


def test_curve_at_idle_lowers_net_rates_on_a_card_that_idles_higher():
    """A card idling 5 W higher loses 5 W over each token's time from its net energy."""
    sweep = olt_reference()
    first = sweep["batch_curve"][0]
    seconds_per_token = (
        first["decode_J_per_output_token"] - first["decode_J_per_output_token_net"]
    ) / sweep["idle_power_W"]
    higher = curve_at_idle(sweep["idle_power_W"] + 5.0)
    assert higher.dec1_net == pytest.approx(curve_at_idle(None).dec1_net - 5.0 * seconds_per_token)


def test_replay_of_a_lone_request_takes_its_batch_one_time():
    """One request alone generates at the batch-1 speed of the curve."""
    first = olt_reference()["batch_curve"][0]
    _, latencies, busy_s = replay(
        [{"arrival_s": 2.0, "prompt_tokens": 100, "generated_tokens": 290}], "average"
    )
    assert latencies[0] == pytest.approx(290 / first["tokens_per_s"])
    assert busy_s == pytest.approx(latencies[0])


def test_compare_run_matches_a_run_that_behaves_like_the_curve():
    """A lone request costed exactly as the curve says compares at about 0%."""
    request = {"arrival_s": 0.0, "finish_s": 0.0, "prompt_tokens": 0, "generated_tokens": 290}
    joules, latencies, _ = replay([request], "average")
    request["finish_s"] = latencies[0]
    idle = olt_reference()["idle_power_W"]
    result = compare_run(_run([request], latencies[0], joules, idle), idle)
    assert off_by(result["gross"]) == pytest.approx(0.0, abs=1e-9)
    assert off_by(result["latency"]) == pytest.approx(0.0, abs=1e-9)


def test_off_by_is_the_relative_difference():
    """110 simulated against 100 measured is 10% above."""
    assert off_by((100.0, 110.0)) == pytest.approx(0.10)
