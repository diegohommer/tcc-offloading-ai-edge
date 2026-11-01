"""Tests for the true energy rates and answer times of each tier."""

import pytest

from energy.three_tier import published_rates, published_speeds
from olt_energy import Energy


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


def test_seconds_on_the_onu_reads_the_prompt_then_generates(prices):
    """The ONU reads the prompt at its prefill speed, then generates at its decode speed."""
    speeds = published_speeds()["onu"]
    assert prices.seconds("onu", 1, 100, 200) == pytest.approx(
        100 / speeds["pf"] + 200 / speeds["dec"]
    )


def test_seconds_at_the_olt_depends_on_the_batch(curve, prices):
    """At the OLT a sequence's speed is its share of the batch's throughput."""
    assert prices.seconds("olt", 8, 100, 300) == pytest.approx(curve.service_s(8, 300))


def test_added_rates_with_the_added_signal_is_net_when_idle_and_slope_when_busy(
    marginal_curve, marginal_prices
):
    """One more query adds the net-of-idle rate to an idle OLT and the slope to a busy one."""
    assert marginal_prices.added_rates(0) == (marginal_curve.pf1_net, marginal_curve.dec1_net)
    assert marginal_prices.added_rates(5) == (marginal_curve.pf_slope, marginal_curve.dec_slope)


def test_added_rates_with_the_share_signal_is_the_share_of_the_batch_it_makes(curve, prices):
    """With the share signal a query counts its share of the batch it joins."""
    assert prices.added_rates(3) == curve.rates(4)


def test_added_rates_do_not_depend_on_the_accounting(curve, marginal_prices):
    """What a query adds excludes idle, so counting idle in the charge does not change it."""
    charged_on_average = Energy(
        curve, published_rates("average"), "average", published_speeds(), "added"
    )
    for running in (0, 1, 5):
        assert charged_on_average.added_rates(running) == marginal_prices.added_rates(running)


def test_decision_rates_can_differ_from_the_rates_charged(curve):
    """Charged on average accounting, the tiers still decide on the ONU's rates above idle."""
    above_idle = published_rates("marginal")
    energy = Energy(
        curve, published_rates("average"), "average", published_speeds(), "added", above_idle
    )
    assert energy.decision_rates("onu") == (above_idle["onu"]["pf"], above_idle["onu"]["dec"])
    assert energy.rates("onu", 1) != energy.decision_rates("onu")
