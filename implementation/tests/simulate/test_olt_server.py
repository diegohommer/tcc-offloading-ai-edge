"""Tests for the OLT's continuous batching, against hand-worked cases."""

# pylint: disable=redefined-outer-name,protected-access

import random

import pytest

import olt_server
from olt_server import OltServer


class ConstantPowerCurve:  # pylint: disable=too-few-public-methods
    """Every sequence gets 10 tokens/s at any batch, so the batch makes 10 x batch tokens/s."""

    tps = None

    @staticmethod
    def _at(_values, batch):
        """Return the batch's total tokens per second."""
        return 10.0 * batch


class ConstantPowerPrices:  # pylint: disable=too-few-public-methods
    """A card drawing 10 W whatever the batch: 1 / batch J per generated token."""

    accounting = "average"

    @staticmethod
    def olt(batch):
        """Return (J per prompt token, J per generated token)."""
        return 0.5 / batch, 1.0 / batch


@pytest.fixture
def toy_server():
    """A server on the constant-power card, where every case can be worked by hand."""
    return OltServer(ConstantPowerCurve(), ConstantPowerPrices())


@pytest.fixture
def measured_server(curve, prices):
    """A server on the measured curve that never forgets its work."""
    return OltServer(curve, prices, report_window_s=1e12)


def _by_query(completions):
    """Index completions by query id: {query: (finished at, prefill J, decode J)}."""
    return {query: (done_at, prefill, decode) for query, done_at, prefill, decode in completions}


def test_admit_alone_runs_at_batch_one(toy_server):
    """A lone query takes tokens / speed and pays the batch-1 rate on every token."""
    toy_server.admit(0, prompt_tokens=20, generated_tokens=100)
    done_at, prefill, decode = _by_query(toy_server.drain())[0]
    assert done_at == pytest.approx(10.0)
    assert prefill == pytest.approx(20 * 0.5)
    assert decode == pytest.approx(100 * 1.0)


def test_admit_charges_prefill_at_the_batch_the_query_makes(toy_server):
    """A query joining one already running pays the batch-2 prompt rate."""
    toy_server.admit(0, 0, 100)
    toy_server.admit(1, 20, 100)
    assert _by_query(toy_server.drain())[1][1] == pytest.approx(20 * 0.25)


def test_advance_shares_the_steps_queries_overlap(toy_server):
    """A runs alone 0-5 s and with B 5-10 s; B then runs alone 10-15 s."""
    toy_server.admit(0, 0, 100)
    assert not toy_server.advance(5.0)
    toy_server.admit(1, 0, 100)
    done = _by_query(toy_server.drain())
    assert done[0][0] == pytest.approx(10.0)
    assert done[1][0] == pytest.approx(15.0)
    assert done[0][2] == pytest.approx(50 * 1.0 + 50 * 0.5)
    assert done[1][2] == pytest.approx(50 * 0.5 + 50 * 1.0)
    # constant power: 10 W over the 15 s the card was busy
    assert done[0][2] + done[1][2] == pytest.approx(150.0)


def test_advance_stops_at_the_target_without_finishing_early(toy_server):
    """Advancing part-way through an answer finishes nothing and moves the clock exactly."""
    toy_server.admit(0, 0, 100)
    assert not toy_server.advance(4.0)
    assert toy_server.clock == 4.0
    assert toy_server.batch == 1


def test_advance_finishes_an_empty_answer_at_once(toy_server):
    """A query that generates no tokens leaves on the next advance, having cost no decode."""
    toy_server.admit(0, 0, 0)
    assert _by_query(toy_server.advance(1.0))[0] == pytest.approx((0.0, 0.0, 0.0))


def test_advance_speed_follows_the_measured_batch(measured_server):
    """Two queries together each get the batch-2 per-sequence speed of the measured curve."""
    measured_server.admit(0, 0, 300)
    measured_server.admit(1, 0, 300)
    per_sequence = measured_server.curve._at(measured_server.curve.tps, 2) / 2
    done = _by_query(measured_server.drain())
    assert done[0][0] == pytest.approx(300 / per_sequence)
    assert done[1][0] == pytest.approx(300 / per_sequence)


def test_admit_queues_when_every_slot_is_busy(monkeypatch, toy_server):
    """With two slots, a third query waits for a slot to free, then runs alone."""
    monkeypatch.setattr(olt_server, "MAX_BATCH", 2)
    for query in range(3):
        toy_server.admit(query, 0, 100)
    assert toy_server.batch == 2
    assert toy_server.queued == 1
    assert toy_server.admitted == 3
    done = _by_query(toy_server.drain())
    assert done[0][0] == pytest.approx(10.0)
    assert done[1][0] == pytest.approx(10.0)
    assert done[2][0] == pytest.approx(20.0)
    assert done[2][2] == pytest.approx(100.0)


@pytest.mark.parametrize("accounting", ["average", "marginal"])
def test_drain_charges_queries_exactly_what_the_olt_spent(accounting, request):
    """Over random arrivals, the queries' joules sum to the OLT's own record of its work."""
    curve = request.getfixturevalue("curve" if accounting == "average" else "marginal_curve")
    energy = request.getfixturevalue("prices" if accounting == "average" else "marginal_prices")
    server = OltServer(curve, energy, report_window_s=1e12)
    rng = random.Random(7)
    clock, completions = 0.0, []
    for query in range(400):
        clock += rng.expovariate(0.5)
        completions += server.advance(clock)
        server.admit(query, rng.randint(50, 150), rng.randint(1, 500))
    completions += server.drain()
    assert len(completions) == 400
    _, _, prompt_joules, _, generated_joules = server._recent_totals
    assert sum(prefill for _, _, prefill, _ in completions) == pytest.approx(prompt_joules)
    assert sum(decode for _, _, _, decode in completions) == pytest.approx(generated_joules)


def test_mean_batch_is_weighted_by_time(toy_server):
    """One query alone for 5 s, then two for 5 s, then one for 5 s: a mean batch of 4/3."""
    toy_server.admit(0, 0, 100)
    toy_server.advance(5.0)
    toy_server.admit(1, 0, 100)
    toy_server.drain()
    assert toy_server.mean_batch == pytest.approx((5 * 1 + 5 * 2 + 5 * 1) / 15)


def test_reported_rate_is_the_mean_over_recent_work(toy_server):
    """Two queries generating together report the batch-2 rate."""
    toy_server.admit(0, 0, 100)
    toy_server.admit(1, 0, 100)
    toy_server.advance(5.0)
    assert toy_server.reported_rate()[1] == pytest.approx(0.5)


def test_reported_rate_of_an_idle_olt_is_the_lone_rate(curve, prices):
    """Once its work ages out of the window, the OLT reports what a lone query would pay."""
    server = OltServer(curve, prices, report_window_s=300)
    for query in range(8):
        server.admit(query, 100, 300)
    server.advance(60)
    assert server.reported_rate()[1] < server._rates(1)[1]
    server.drain()
    server.advance(server.clock + 301)
    assert server.reported_rate() == pytest.approx(server._rates(1))


def test_send_broadcast_holds_the_report_until_the_next_send(toy_server):
    """What was broadcast stays put while the OLT's own report moves on."""
    toy_server.admit(0, 0, 100)
    toy_server.advance(1.0)
    toy_server.send_broadcast()
    toy_server.admit(1, 0, 100)
    toy_server.advance(9.0)
    assert toy_server.last_broadcast[1] == pytest.approx(1.0)
    assert toy_server.reported_rate()[1] < 1.0


def test_reported_rate_under_marginal_accounting_is_what_one_more_query_adds(
    marginal_curve, marginal_prices
):
    """Busy for the whole window, the OLT reports the slope; idle, the net-of-idle rate."""
    server = OltServer(marginal_curve, marginal_prices, report_window_s=10.0)
    server.advance(100.0)
    assert server.reported_rate() == pytest.approx(marginal_prices.added_rates(0))
    for query in range(4):
        server.admit(query, 100, 10_000)
    server.advance(120.0)
    assert server.reported_rate() == pytest.approx(marginal_prices.added_rates(1))


def test_reported_rate_under_marginal_accounting_weights_by_busy_time(
    marginal_curve, marginal_prices
):
    """Busy for half the window, the OLT reports halfway between the slope and the net rate."""
    server = OltServer(marginal_curve, marginal_prices, report_window_s=10.0)
    server.advance(100.0)
    per_sequence = marginal_curve._at(marginal_curve.tps, 1)
    server.admit(0, 0, 5.0 * per_sequence)  # generates for exactly 5 s
    server.advance(110.0)
    idle, busy = marginal_prices.added_rates(0), marginal_prices.added_rates(1)
    assert server.reported_rate()[1] == pytest.approx((idle[1] + busy[1]) / 2)
