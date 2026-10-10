"""Tests for one simulated run of the cascade, on a small synthetic stream."""

# pylint: disable=redefined-outer-name

import math
import random
from types import SimpleNamespace

import pytest

import cascade
from cascade import believed_olt_rate, calibrate, POLICIES, run, RunSetup, STATIC
from energy.three_tier import TIERS


def _answers(questions=40, seed=1):
    """Return answers[question][tier] with confidence and accuracy rising up the tiers."""
    rng = random.Random(seed)
    answers = {}
    for question in range(questions):
        answers[question] = {
            tier: {
                "correct": rng.random() < 0.4 + 0.25 * level,
                "conf": min(rng.random() * 0.5 + 0.2 * level + 0.1, 1.0),
                "tp": 90 + level * 10,
                "tg": rng.randint(100, 300),
                "qb": 200,
                "ab": 600,
            }
            for level, tier in enumerate(TIERS)
        }
    return answers


def _settings(**overrides):
    """Return the settings run() reads, at the study's values."""
    values = {
        "window": 50,
        "alpha": 0.05,
        "warmup": 3,
        "rate_alpha": 0.3,
        "shared_stats": True,
        "report_window": 5.0,
        "report": "window",
        "broadcast_interval_s": 10.0,
        "delta": 0.0,
    }
    return SimpleNamespace(**{**values, **overrides})


@pytest.fixture
def setup(curve, prices):
    """A day of 600 queries from 30 households, with every static table filled in."""
    rng = random.Random(2)
    answers = _answers()
    stream = sorted(
        ((rng.choice(list(answers)), rng.uniform(0, 24), rng.randrange(30)) for _ in range(600)),
        key=lambda query: query[1],
    )
    run_setup = RunSetup(
        answers=answers,
        stream=stream,
        curve=curve,
        prices=prices,
        static_rates={},
        settings=_settings(),
        schedule_key=lambda hour: (0, int(hour) % 24),
    )
    table = calibrate(run_setup, 0.5)
    run_setup.static_rates = {
        **table,
        "stale_low": table["static_day"],
        "stale_high": table["static_day"],
    }
    return run_setup


@pytest.mark.parametrize("policy", POLICIES)
def test_run_settles_every_query_exactly_once(setup, policy):
    """Every query ends at one tier: the shares sum to 1 and the hours hold the whole stream."""
    metrics, per_hour, _ = run(setup, 0.5, policy)
    assert sum(metrics[f"final_{tier}"] for tier in TIERS) == pytest.approx(1.0)
    assert sum(hour["queries"] for hour in per_hour) == len(setup.stream)


def test_run_with_equal_confidences_answers_everything_on_the_phone(setup, prices):
    """When every phone answer is equally confident, none is below the window's quantile."""
    for answers in setup.answers.values():
        answers["user"]["conf"] = 0.5
    metrics, _, _ = run(setup, 0.5, "recserve")
    assert metrics["final_user"] == 1.0
    phone = [setup.answers[question]["user"] for question, _, _ in setup.stream]
    prompt_rate, generated_rate = prices.rates("user", 1)
    expected = sum(prompt_rate * answer["tp"] + generated_rate * answer["tg"] for answer in phone)
    assert metrics["J_per_query"] == pytest.approx(expected / len(phone))


def test_run_recserve_climbs_one_tier_at_a_time(setup):
    """Plain RecServe never forwards on arrival and never skips a tier."""
    metrics, _, _ = run(setup, 0.7, "recserve")
    assert metrics["forwarded_on_arrival"] == 0
    assert metrics["skipped_on_escalation"] == 0
    assert metrics["final_olt"] > 0


def test_run_recserve_no_onu_sends_every_escalation_to_the_olt(setup):
    """The control chain never answers at the ONU."""
    metrics, _, _ = run(setup, 0.7, "recserve_no_onu")
    assert metrics["final_onu"] == 0


@pytest.mark.parametrize(
    "policy, tier", [("user_alone", "user"), ("onu_alone", "onu"), ("olt_alone", "olt")]
)
def test_run_alone_answers_every_query_at_its_tier(setup, policy, tier):
    """A single-tier policy answers everything at its tier, with that tier's accuracy."""
    metrics, _, _ = run(setup, 0.5, policy)
    assert metrics[f"final_{tier}"] == 1.0
    correct = [setup.answers[question][tier]["correct"] for question, _, _ in setup.stream]
    assert metrics["accuracy"] == pytest.approx(sum(correct) / len(correct))


def test_run_fixed_chains_measure_no_rate_error(setup):
    """Policies that never route on energy have no belief to compare with what was paid."""
    metrics, _, _ = run(setup, 0.7, "recserve")
    assert math.isnan(metrics["olt_rate_error"])


def test_run_observed_cells_count_the_olt_answers(setup):
    """The timetable cells hold every query the OLT answered."""
    metrics, _, observed = run(setup, 0.7, "recserve")
    answered = sum(count for _, _, count in observed.values())
    assert answered == round(metrics["final_olt"] * len(setup.stream))


def test_calibrate_weights_static_day_by_olt_answers(monkeypatch, setup):
    """A busy cell counts for its answers, not as one cell among many."""
    observed = {(0, 3): (1.0, 10.0, 3), (0, 20): (2.0, 20.0, 1)}
    monkeypatch.setattr(cascade, "run", lambda *_: ({}, [], observed))
    table = calibrate(setup, 0.5)
    assert table["static_day"] == pytest.approx((1.25, 12.5))
    assert table["static_hour"] == {(0, 3): (1.0, 10.0), (0, 20): (2.0, 20.0)}


def test_calibrate_without_olt_answers_ships_nothing(monkeypatch, setup):
    """With nothing observed there is no table to ship."""
    monkeypatch.setattr(cascade, "run", lambda *_: ({}, [], {}))
    assert not calibrate(setup, 0.5)


def test_believed_olt_rate_static_hour_falls_back_to_static_day(setup):
    """An hour the calibration never saw uses the day's rate."""
    setup.static_rates = {"static_day": (1.0, 2.0), "static_hour": {(0, 3): (5.0, 6.0)}}
    assert believed_olt_rate("static_hour", setup, None, None, 3.5) == (5.0, 6.0)
    assert believed_olt_rate("static_hour", setup, None, None, 4.5) == (1.0, 2.0)


@pytest.mark.parametrize("policy", STATIC)
def test_believed_olt_rate_static_without_a_table_is_none(setup, policy):
    """A static policy with no table has nothing to route on."""
    setup.static_rates = {}
    assert believed_olt_rate(policy, setup, None, None, 3.0) is None


def test_believed_olt_rate_broadcast_is_the_last_send(setup):
    """Broadcast hears what the OLT last sent, not its current report."""
    server = SimpleNamespace(last_broadcast=(0.1, 0.2), reported_rate=lambda: (9.0, 9.0))
    assert believed_olt_rate("broadcast", setup, server, None, 3.0) == (0.1, 0.2)
    assert believed_olt_rate("oracle", setup, server, None, 3.0) == (9.0, 9.0)
