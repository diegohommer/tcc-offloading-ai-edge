"""Tests for turning the GPU's energy counter readings into per-token figures."""

import math

import pytest

from collect_answers import confidence
from measure_gpu_energy import add_per_query, median_row, summarize_trial


def test_summarize_trial_splits_prefill_from_decode():
    """Decode is the full pass minus the prefill pass, over the tokens after the first."""
    prefill = {"joules": 20.0, "seconds": 1.0, "in_tokens": 100}
    full = {"joules": 220.0, "seconds": 5.0, "in_tokens": 100, "out_tokens": 102}
    trial = summarize_trial(prefill, full, idle_watts=10.0, batch=2)
    assert trial["prefill_J_per_input_token"] == pytest.approx(0.2)
    assert trial["prefill_J_per_input_token_net"] == pytest.approx(0.1)
    assert trial["decode_J_per_output_token"] == pytest.approx(200.0 / 100)
    assert trial["decode_J_per_output_token_net"] == pytest.approx((170.0 - 10.0) / 100)
    assert trial["tokens_per_s"] == pytest.approx(102 / 5.0)


def test_median_row_takes_the_median_of_every_field():
    """Each field is the median across repeats, and the repeats' decode figures are kept."""
    trials = [{"decode_J_per_output_token": value} for value in (3.0, 1.0, 2.0)]
    row = median_row(4, trials)
    assert row["batch"] == 4
    assert row["decode_J_per_output_token"] == 2.0
    assert row["trials_decode_J_per_output_token"] == [3.0, 1.0, 2.0]


def test_add_per_query_marginal_is_the_slope_between_batches():
    """Average is the batch's energy over its queries; marginal the slope from the batch before."""
    rows = add_per_query(
        [
            {"batch": 4, "batch_J": 160.0, "batch_J_net": 80.0},
            {"batch": 1, "batch_J": 100.0, "batch_J_net": 50.0},
        ]
    )
    assert [row["batch"] for row in rows] == [1, 4]
    assert rows[0]["marginal_J_per_query"] == rows[0]["average_J_per_query"] == 100.0
    assert rows[1]["average_J_per_query"] == pytest.approx(40.0)
    assert rows[1]["marginal_J_per_query"] == pytest.approx(20.0)
    assert rows[1]["marginal_J_per_query_net"] == pytest.approx(10.0)


def test_confidence_is_exp_of_the_mean_logprob():
    """RecServe's confidence is 1 / perplexity, skipping missing logprobs."""
    assert confidence([math.log(0.5), None, math.log(0.5)]) == pytest.approx(0.5)
    assert confidence([]) == 0.0
