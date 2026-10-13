"""Tests for RecServe's confidence window, what households learn, and the routing rule."""

import random

import numpy as np
import pytest

from routing import cost_to_completion, Learned, on_arrival, on_escalation, OwnStats, Window

PROMPT_TOKENS = 100


def _learned(user_tokens=200, onu_tokens=200, olt_tokens=200, p_user=0.5, p_onu=0.5):
    """Return a household that knows its answer lengths and escalation rates."""
    learned = Learned(alpha=0.1, warmup=1)
    learned.tokens = {"user": user_tokens, "onu": onu_tokens, "olt": olt_tokens}
    learned.p_further = {"user": p_user, "onu": p_onu, "olt": 0.0}
    return learned


def _rates(user=0.1, onu=0.5, olt=0.3):
    """Return J per generated token for each tier, with free prompts."""
    return {"user": (0.0, user), "onu": (0.0, onu), "olt": (0.0, olt)}


def test_window_quantile_matches_numpy_percentile():
    """The interpolated quantile is numpy's default percentile."""
    rng = random.Random(3)
    window = Window(size=500)
    values = [rng.random() for _ in range(200)]
    for value in values:
        window.add(value)
    for quantile in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert window.quantile(quantile) == pytest.approx(np.percentile(values, quantile * 100))


def test_window_add_drops_the_oldest_once_full():
    """A full window forgets its oldest confidence, whatever its value."""
    window = Window(size=3)
    for value in (0.9, 0.1, 0.5, 0.7):
        window.add(value)
    assert len(window) == 3
    assert window.ordered == [0.1, 0.5, 0.7]


def test_learned_absorb_starts_from_the_first_report_then_averages():
    """The first report is taken as is; later ones move it by the EWMA weight."""
    learned = Learned(alpha=0.5, warmup=2, rate_alpha=0.25)
    learned.absorb({"olt": (1.0, 4.0, 100)}, "olt")
    learned.absorb({"olt": (1.0, 8.0, 300)}, "olt")
    assert learned.rates["olt"] == pytest.approx((1.0, 5.0))
    assert learned.tokens["olt"] == pytest.approx(200)


def test_learned_absorb_counts_tiers_the_query_went_past():
    """Tiers below the one that answered passed the query on; the answering tier did not."""
    learned = Learned(alpha=1.0, warmup=1)
    learned.absorb({"user": (0.1, 0.2, 50), "onu": (0.0, 0.5, 80)}, "onu")
    assert learned.p_further == {"user": 1.0, "onu": 0.0}


def test_learned_ready_waits_for_warmup_reports_from_every_tier():
    """A household decides only once each tier it would use has reported warmup times."""
    learned = Learned(alpha=0.1, warmup=2)
    learned.absorb({"user": (0.1, 0.2, 50), "olt": (0.1, 0.2, 50)}, "olt")
    assert not learned.ready(["user", "olt"])
    learned.absorb({"user": (0.1, 0.2, 50)}, "user")
    assert not learned.ready(["user", "olt"])
    learned.absorb({"olt": (0.1, 0.2, 50)}, "olt")
    assert learned.ready(["user", "olt"])


def test_learned_pool_shares_question_statistics_but_not_rates():
    """Pooled households share answer lengths and escalation rates, never energy rates."""
    pool = Learned(alpha=1.0, warmup=1)
    first, second = Learned(1.0, 1, pool=pool), Learned(1.0, 1, pool=pool)
    first.absorb({"olt": (0.1, 0.2, 50)}, "olt")
    assert second.tokens["olt"] == 50
    assert second.heard["olt"] == 1
    assert "olt" not in second.rates


def test_cost_to_completion_weights_each_tier_by_the_chance_of_reaching_it():
    """C(user) = E(user) + p(user) x (E(onu) + p(onu) x E(olt))."""
    expected = 200 * 0.1 + 0.5 * (200 * 0.5 + 0.5 * 200 * 0.3)
    assert cost_to_completion(0, PROMPT_TOKENS, _rates(), _learned()) == pytest.approx(expected)


def test_on_arrival_runs_here_when_this_tier_is_cheap():
    """A cheap phone that rarely escalates keeps the query."""
    assert on_arrival(0, PROMPT_TOKENS, _rates(), _learned(p_user=0.1), 0.0) == 0


def test_on_arrival_forwards_when_a_tier_above_is_cheaper():
    """A phone that almost always escalates is worth skipping for a cheap OLT."""
    learned = _learned(p_user=0.95)
    assert on_arrival(0, PROMPT_TOKENS, _rates(user=0.1, olt=0.05), learned, 0.0) == 2


def test_on_arrival_margin_keeps_a_small_saving_from_forwarding():
    """A forward saving less than delta of the cost of running here is not taken."""
    learned = _learned(p_user=0.95)
    rates = _rates(user=0.1, olt=0.098)
    assert on_arrival(0, PROMPT_TOKENS, rates, learned, 0.0) == 2
    assert on_arrival(0, PROMPT_TOKENS, rates, learned, 0.5) == 0


def test_on_arrival_at_the_top_stays():
    """The OLT has nowhere to forward to."""
    assert on_arrival(2, PROMPT_TOKENS, _rates(), _learned(), 0.0) == 2


def test_on_escalation_skips_an_expensive_onu():
    """An escalation from the phone goes straight to the OLT when the ONU costs more."""
    assert on_escalation(0, PROMPT_TOKENS, _rates(onu=0.5, olt=0.1), _learned(), 0.0) == 2


def test_on_escalation_takes_the_next_tier_when_it_is_cheaper():
    """A cheap ONU that rarely escalates is the next stop."""
    learned = _learned(p_onu=0.1)
    assert on_escalation(0, PROMPT_TOKENS, _rates(onu=0.05, olt=0.3), learned, 0.0) == 1


def test_learned_fixed_escalation_stays_at_beta():
    """With --escalation beta the packets never move the escalation rates."""
    learned = Learned(alpha=1.0, warmup=1, escalation=0.3)
    learned.absorb({"user": (0.1, 0.2, 50), "onu": (0.0, 0.5, 80)}, "onu")
    assert learned.p_further["user"] == learned.p_further["onu"] == 0.3
    assert learned.tokens == {"user": 50, "onu": 80}


def test_learned_ship_makes_a_household_ready_at_once():
    """Shipped lengths count as heard, so only the energy rates remain to arrive."""
    learned = Learned(alpha=0.1, warmup=20)
    learned.ship({"user": 100.0, "onu": 150.0, "olt": 200.0}, 0.5)
    assert learned.ready(["user", "onu", "olt"])
    assert learned.tokens["olt"] == 200.0


def test_learned_adopt_takes_what_each_tier_published():
    """A published packet replaces the lengths outright instead of averaging one answer in."""
    olt = OwnStats(alpha=0.5, shipped_tokens=100.0)
    olt.record(300, None)
    learned = Learned(alpha=0.01, warmup=1, escalation=0.4)
    learned.ship({"olt": 50.0}, 0.4)
    learned.adopt({"olt": (0.1, 0.2, olt)})
    assert learned.tokens["olt"] == pytest.approx(200.0)
    assert learned.rates["olt"] == (0.1, 0.2)
    assert learned.p_further["olt"] == 0.4


def test_own_stats_starts_from_the_shipped_length_and_tracks_escalations():
    """A tier's own averages start from its model's measured length."""
    own = OwnStats(alpha=0.5, shipped_tokens=100.0)
    own.record(200, True)
    own.record(200, False)
    assert own.tokens == pytest.approx(175.0)
    assert own.p_further == pytest.approx(0.5)
    assert own.answers == 2


def test_window_fill_starts_full_and_then_drops_the_oldest():
    """A window filled in advance reads quantiles at once and forgets its oldest value first."""
    window = Window(size=3)
    window.fill([0.9, 0.1, 0.5])
    assert window.quantile(0.5) == 0.5
    window.add(0.7)
    assert window.ordered == [0.1, 0.5, 0.7]
