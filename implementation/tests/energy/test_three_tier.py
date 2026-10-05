"""Tests for the tiers' energy figures and the OLT's measured curve."""

# pylint: disable=protected-access

import math

import pytest

from energy.three_tier import (
    _slope,
    load_answers,
    olt_factor,
    olt_reference,
    OltCurve,
    published_rates,
    TIERS,
)


def test_slope_of_a_straight_line_is_exact():
    """The least-squares slope of points on a line is the line's slope."""
    assert _slope([1, 2, 4, 8], [3 + 2 * x for x in (1, 2, 4, 8)]) == pytest.approx(2.0)


def test_olt_curve_at_returns_the_measured_points(curve):
    """At a measured batch size the curve gives the measurement itself."""
    for batch, decode in zip(curve.b, curve.dec):
        assert curve._at(curve.dec, batch) == pytest.approx(decode)


def test_olt_curve_at_interpolates_log_log():
    """Halfway between two batches on a log scale lands halfway between their logs."""
    rows = [
        {
            "batch": batch,
            "prefill_J_per_input_token": 1.0,
            "decode_J_per_output_token": decode,
            "tokens_per_s": 10.0,
            "prefill_J_per_input_token_net": 1.0,
            "decode_J_per_output_token_net": decode,
        }
        for batch, decode in ((1, 4.0), (4, 1.0))
    ]
    toy = OltCurve(rows)
    assert toy._at(toy.dec, 2) == pytest.approx(math.sqrt(4.0 * 1.0))


def test_olt_curve_at_clamps_outside_the_measured_range(curve):
    """Below the first and above the last batch the curve holds its end values."""
    assert curve._at(curve.dec, 0.5) == pytest.approx(curve.dec[0])
    assert curve._at(curve.dec, 1000) == pytest.approx(curve.dec[-1])


def test_olt_curve_rates_scale_with_the_boundary_factor():
    """The boundary factor multiplies every energy figure."""
    rows = olt_reference()["batch_curve"]
    card, system = OltCurve(rows), OltCurve(rows, 2.5)
    for batch in (1, 3, 64):
        assert system.rates(batch)[1] == pytest.approx(2.5 * card.rates(batch)[1])


def test_olt_curve_energy_per_token_falls_with_batch(curve):
    """Decode gets cheaper per token as the batch grows, as measured."""
    decode = [curve.rates(batch)[1] for batch in (1, 2, 4, 8, 16, 32, 64)]
    assert decode == sorted(decode, reverse=True)


def test_olt_curve_service_time_is_tokens_over_per_sequence_speed(curve):
    """A sequence in a batch of 8 gets an eighth of the batch's throughput."""
    per_sequence = curve._at(curve.tps, 8) / 8
    assert curve.service_s(8, 300) == pytest.approx(300 / per_sequence)


def test_marginal_rates_are_the_measured_net_rates_at_each_batch(marginal_curve):
    """At a measured batch size each sequence pays the net-of-idle rate measured there."""
    for index, batch in enumerate(marginal_curve.b):
        assert marginal_curve.marginal_rates(batch) == pytest.approx(
            (marginal_curve.pf_net[index], marginal_curve.dec_net[index])
        )


def test_marginal_rates_between_batches_interpolate_the_net_curve(marginal_curve):
    """Between measured batches the net rate lies between its two neighbours."""
    low, high = marginal_curve.marginal_rates(4)[1], marginal_curve.marginal_rates(6)[1]
    assert min(low, high) <= marginal_curve.marginal_rates(5)[1] <= max(low, high)


def test_marginal_rates_never_exceed_average_rates(curve, marginal_curve):
    """What a batch adds above idle costs less than its share of the whole draw."""
    for batch in (1, 2, 8, 64):
        assert marginal_curve.marginal_rates(batch)[1] < curve.rates(batch)[1]


def test_olt_factor_orders_the_boundaries():
    """The card alone is x1, a low-PUE site is below the industry PUE, marginal below average."""
    assert olt_factor("gpu") == 1.0
    assert 1.0 < olt_factor("system-low") < olt_factor("system")
    assert olt_factor("system", "marginal") < olt_factor("system", "average")


def test_published_rates_price_the_onu_on_generated_tokens_only():
    """The ONU's figure is all-in per generated token, so its prompt rate is zero."""
    assert published_rates()["onu"]["pf"] == 0


def test_published_rates_marginal_drops_the_onu_idle_draw():
    """Under marginal accounting the ONU costs less; the phone is priced the same."""
    average, marginal = published_rates("average"), published_rates("marginal")
    assert marginal["onu"]["dec"] < average["onu"]["dec"]
    assert marginal["user"] == average["user"]


def test_load_answers_covers_every_question_for_every_tier():
    """Every tier answered the same 1,319 questions, in order, with usable confidences."""
    records, models, sources = load_answers()
    assert set(records) == set(TIERS) == set(models)
    assert len(sources) == 2
    indices = [[record["index"] for record in records[tier]] for tier in TIERS]
    assert all(tier_indices == sorted(tier_indices) for tier_indices in indices)
    assert all(len(tier_indices) == 1319 for tier_indices in indices)
    for tier in TIERS:
        assert all(0 < record["cmean"] <= 1 for record in records[tier])
        assert all(record["cmin"] <= record["cmean"] for record in records[tier])
