"""Tests for the true energy rates and answer times of each tier."""

import pytest

from energy.three_tier import published_rates, published_speeds


def test_rates_below_the_olt_are_the_published_figures(prices):
    """The phone and the ONU cost the same whatever the OLT's batch."""
    published = published_rates("average")
    for tier in ("user", "onu"):
        expected = (published[tier]["pf"], published[tier]["dec"])
        assert prices.rates(tier, 1) == expected
        assert prices.rates(tier, 32) == expected


def test_rates_at_the_olt_follow_the_accounting(curve, marginal_curve, prices, marginal_prices):
    """Average accounting reads the batch curve, marginal its share above idle."""
    assert prices.rates("olt", 4) == curve.rates(4)
    assert marginal_prices.rates("olt", 4) == marginal_curve.marginal_rates(4)


def test_seconds_on_the_phone_is_prefill_plus_decode(prices):
    """The phone has separate prefill and decode speeds."""
    speeds = published_speeds()["user"]
    assert prices.seconds("user", 1, 100, 200) == pytest.approx(
        100 / speeds["pf"] + 200 / speeds["dec"]
    )


def test_seconds_on_the_onu_counts_generated_tokens_only(prices):
    """The ONU's one speed already includes its prefill."""
    speeds = published_speeds()["onu"]
    assert prices.seconds("onu", 1, 100, 200) == pytest.approx(200 / speeds["dec"])


def test_seconds_at_the_olt_depends_on_the_batch(curve, prices):
    """At the OLT a sequence's speed is its share of the batch's throughput."""
    assert prices.seconds("olt", 8, 100, 300) == pytest.approx(curve.service_s(8, 300))
