"""Continuous batching at the OLT, checked against hand-worked cases."""

# pylint: disable=wrong-import-position,protected-access,too-few-public-methods

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "simulate"))
import olt_server
from energy.three_tier import olt_factor, olt_reference, OltCurve
from olt_energy import Energy
from olt_server import OltServer


class ConstantPowerCurve:
    """Every sequence gets 10 tokens/s at any batch, and the card draws 10 W in total."""

    tps = None

    @staticmethod
    def _at(_values, batch):
        """Return the batch's total tokens per second."""
        return 10.0 * batch


class ConstantPowerPrices:
    """J per generated token at constant power: 10 W over 10 x batch tokens/s."""

    @staticmethod
    def olt(batch):
        """Return (J per prompt token, J per generated token)."""
        return 0.5, 1.0 / batch


def measured_server(accounting="average", report_window_s=1e12):
    """Return a server on the measured OLT curve."""
    curve = OltCurve(olt_reference()["batch_curve"], olt_factor("system", accounting))
    return OltServer(curve, Energy(curve, {}, accounting), report_window_s)


def by_query(completions):
    """Index completions by query id."""
    return {query: (done_at, prefill, decode) for query, done_at, prefill, decode in completions}


def test_lone_query_runs_at_batch_one():
    """A query alone takes tokens / speed and pays the batch-1 rate on every token."""
    server = OltServer(ConstantPowerCurve(), ConstantPowerPrices())
    server.admit(0, prompt_tokens=20, generated_tokens=100)
    done_at, prefill, decode = by_query(server.drain())[0]
    assert done_at == pytest.approx(10.0)
    assert prefill == pytest.approx(10.0)
    assert decode == pytest.approx(100.0)


def test_overlapping_queries_share_the_steps_they_overlap():
    """A runs alone 0-5 s, with B 5-10 s; B then runs alone 10-15 s."""
    server = OltServer(ConstantPowerCurve(), ConstantPowerPrices())
    server.admit(0, 0, 100)
    assert not server.advance(5.0)
    server.admit(1, 0, 100)
    done = by_query(server.drain())
    assert done[0][0] == pytest.approx(10.0)
    assert done[1][0] == pytest.approx(15.0)
    assert done[0][2] == pytest.approx(50 * 1.0 + 50 * 0.5)
    assert done[1][2] == pytest.approx(50 * 0.5 + 50 * 1.0)
    # constant power: the energy charged is 10 W over the 15 s the card was busy
    assert done[0][2] + done[1][2] == pytest.approx(150.0)


def test_speed_follows_the_measured_batch():
    """Two queries together each get the batch-2 per-sequence speed of the measured curve."""
    server = measured_server()
    server.admit(0, 0, 300)
    server.admit(1, 0, 300)
    per_sequence = server.curve._at(server.curve.tps, 2) / 2
    done = by_query(server.drain())
    assert done[0][0] == pytest.approx(300 / per_sequence)
    assert done[1][0] == pytest.approx(300 / per_sequence)


def test_full_server_queues_until_a_slot_frees(monkeypatch):
    """With two slots, a third query waits for the first to leave, then runs."""
    monkeypatch.setattr(olt_server, "MAX_BATCH", 2)
    server = OltServer(ConstantPowerCurve(), ConstantPowerPrices())
    for query in range(3):
        server.admit(query, 0, 100)
    assert server.batch == 2 and server.queued == 1
    done = by_query(server.drain())
    assert done[0][0] == pytest.approx(10.0)
    assert done[1][0] == pytest.approx(10.0)
    assert done[2][0] == pytest.approx(20.0)
    assert done[2][2] == pytest.approx(100.0)  # alone once it starts


@pytest.mark.parametrize("accounting", ["average", "marginal"])
def test_energy_charged_to_queries_is_the_energy_the_olt_spent(accounting):
    """Over random arrivals, the queries' prefill and decode joules sum to the OLT's own totals."""
    server = measured_server(accounting)
    rng = random.Random(7)
    clock, completions = 0.0, []
    for query in range(400):
        clock += rng.expovariate(0.5)
        completions += server.advance(clock)
        server.admit(query, rng.randint(50, 150), rng.randint(1, 500))
    completions += server.drain()
    assert len(completions) == 400
    _, prompt_joules, _, generated_joules = server._recent_totals
    assert sum(prefill for _, _, prefill, _ in completions) == pytest.approx(prompt_joules)
    assert sum(decode for _, _, _, decode in completions) == pytest.approx(generated_joules)


def test_marginal_batch_adds_net_plus_slope_per_extra_sequence():
    """Under marginal accounting a batch of n adds net + (n - 1) x slope per step, in total."""
    curve = OltCurve(olt_reference()["batch_curve"], olt_factor("system", "marginal"))
    for batch in (1, 2, 5, 64):
        _, generated = curve.marginal_rates(batch)
        assert generated * batch == pytest.approx(curve.dec1_net + (batch - 1) * curve.dec_slope)


def test_idle_olt_reports_the_lone_rate():
    """Once its work ages out of the window, the OLT reports what a lone query would pay."""
    server = measured_server(report_window_s=300)
    for query in range(8):
        server.admit(query, 100, 300)
    server.advance(60)
    assert server.reported_rate()[1] < server._rates(1)[1]
    server.drain()
    server.advance(server.clock + 301)
    assert server.reported_rate() == pytest.approx(server._rates(1))
