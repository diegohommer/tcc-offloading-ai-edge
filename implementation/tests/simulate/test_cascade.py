"""Tests for one simulated run of the cascade, on a small synthetic stream."""

# pylint: disable=redefined-outer-name

import math
import random
from types import SimpleNamespace

import pytest

import cascade
from cascade import believed_olt_rate, calibrate, POLICIES, run, RunSetup, STATIC
from energy.three_tier import published_rates, published_speeds, TIERS
from olt_energy import Energy


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
        "stats": "pooled",
        "escalation": "learned",
        "windows": "shared",
        "seed": 7,
        "timing": "fixed",
        "context": "none",
        "onu_queue": False,
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
    # each household's messages form one conversation, 30 seconds between them
    last_of = {}
    follow = [None] * len(stream)
    for index, (_, _, household) in enumerate(stream):
        if household in last_of:
            follow[last_of[household]] = (index, 30.0)
        last_of[household] = index
    run_setup = RunSetup(
        answers=answers,
        stream=stream,
        follow=follow,
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
@pytest.mark.parametrize("timing", ["fixed", "after-answer"])
def test_run_settles_every_query_exactly_once(setup, policy, timing):
    """Every query ends at one tier: the shares sum to 1 and the hours hold the whole stream."""
    setup.settings = _settings(timing=timing)
    metrics, per_hour, _ = run(setup, 0.5, policy)
    assert sum(metrics[f"final_{tier}"] for tier in TIERS) == pytest.approx(1.0)
    assert sum(hour["queries"] for hour in per_hour) == len(setup.stream)


@pytest.mark.parametrize("policy", ["static_hour", "broadcast", "piggyback"])
@pytest.mark.parametrize("stats", ["pooled", "household"])
def test_the_accounting_changes_the_energy_counted_but_not_the_routing(setup, curve, policy, stats):
    """Deciding on what a query adds, both accountings send every query to the same tier."""
    above_idle = published_rates("marginal")
    results = {}
    for accounting, charged in (("marginal", above_idle), ("average", published_rates("average"))):
        setup.prices = Energy(curve, charged, accounting, published_speeds(), "added", above_idle)
        setup.settings = _settings(stats=stats, timing="after-answer")
        setup.static_rates = calibrate(setup, 0.5)
        results[accounting] = run(setup, 0.5, policy)[0]
    for tier in TIERS:
        assert results["average"][f"final_{tier}"] == results["marginal"][f"final_{tier}"]
    assert results["average"]["J_per_query"] > results["marginal"]["J_per_query"]


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


# ==========================================
# Each hop happens at its own moment
# ==========================================
def _answer(conf, prompt_tokens=100, generated_tokens=200):
    """Return one tier's recorded answer."""
    return {
        "correct": True,
        "conf": conf,
        "tp": prompt_tokens,
        "tg": generated_tokens,
        "qb": 200,
        "ab": 600,
    }


SENT_S = 3600.0
"""When the message under test is sent, far from anything else in the stream."""


def _climbing_setup(curve, prices, **settings):
    """Return messages that give the phone and the ONU a history, then one that fails at both."""
    answers = {
        "sure": {tier: _answer(0.9) for tier in TIERS},
        "to_onu": {"user": _answer(0.1), "onu": _answer(0.9), "olt": _answer(0.9)},
        "unsure": {tier: _answer(0.1) for tier in TIERS},
    }
    stream = [
        ("sure", 0.0, 0),
        ("sure", 0.1, 0),
        ("sure", 0.2, 0),
        ("to_onu", 0.3, 0),
        ("to_onu", 0.4, 0),
        ("to_onu", 0.5, 0),
        ("unsure", SENT_S / 3600, 0),
    ]
    return RunSetup(
        answers=answers,
        stream=stream,
        curve=curve,
        prices=prices,
        static_rates={},
        settings=_settings(stats="published", escalation="beta", **settings),
        schedule_key=lambda hour: (0, int(hour) % 24),
    )


def _spy_on_rates(monkeypatch, rate_at):
    """Make the OLT's believed rate a function of time, and return the times it was read."""
    reads = []

    def believed(_policy, _setup, _server, _learned, hour):
        reads.append(hour * 3600)
        return rate_at(hour * 3600)

    monkeypatch.setattr(cascade, "believed_olt_rate", believed)
    return reads


DEAR, CHEAP = (1.0, 10.0), (1e-9, 1e-9)
"""OLT rates that make every tier below climb one step, or send the query straight up."""


def test_every_decision_reads_the_olt_rate_at_its_own_moment(monkeypatch, marginal_curve, marginal_prices):
    """Phone on arrival, phone done, ONU on arrival, ONU done: four moments, not one."""
    reads = _spy_on_rates(monkeypatch, lambda _now: DEAR)
    setup = _climbing_setup(marginal_curve, marginal_prices)
    run(setup, 0.5, "broadcast")
    phone_s = marginal_prices.seconds("user", 1, 100, 200)
    onu_s = marginal_prices.seconds("onu", 1, 100, 200)
    assert reads[-4:] == pytest.approx(
        [SENT_S, SENT_S + phone_s, SENT_S + phone_s, SENT_S + phone_s + onu_s]
    )


def test_a_rate_that_falls_while_the_phone_runs_changes_where_the_query_goes(
    monkeypatch, marginal_curve, marginal_prices
):
    """Dear when sent and cheap when the phone finishes: the query skips the ONU."""
    phone_s = marginal_prices.seconds("user", 1, 100, 200)
    _spy_on_rates(monkeypatch, lambda now: DEAR if now < SENT_S + phone_s / 2 else CHEAP)
    setup = _climbing_setup(marginal_curve, marginal_prices)
    metrics = run(setup, 0.5, "broadcast")[0]
    prompt_rate, generated_rate = marginal_prices.rates("onu", 1)
    one_onu_run = prompt_rate * 100 + generated_rate * 200
    assert metrics["skipped_on_escalation"] == pytest.approx(1 / 7)
    assert metrics["J_onu_per_query"] == pytest.approx(3 * one_onu_run / 7)  # the history only
    assert metrics["final_olt"] == pytest.approx(1 / 7)


def test_a_rate_that_falls_while_the_onu_runs_skips_nothing_but_was_read_late(
    monkeypatch, marginal_curve, marginal_prices
):
    """Cheap only once the ONU has run: the phone and the ONU both spent their energy."""
    phone_s = marginal_prices.seconds("user", 1, 100, 200)
    onu_s = marginal_prices.seconds("onu", 1, 100, 200)
    cutoff = SENT_S + phone_s + onu_s / 2
    _spy_on_rates(monkeypatch, lambda now: DEAR if now < cutoff else CHEAP)
    setup = _climbing_setup(marginal_curve, marginal_prices)
    metrics = run(setup, 0.5, "broadcast")[0]
    prompt_rate, generated_rate = marginal_prices.rates("onu", 1)
    one_onu_run = prompt_rate * 100 + generated_rate * 200
    assert metrics["J_onu_per_query"] == pytest.approx(4 * one_onu_run / 7)  # history and this one
    assert metrics["final_olt"] == pytest.approx(1 / 7)


def test_a_query_that_climbs_takes_each_tier_in_turn(marginal_curve, marginal_prices):
    """Its latency is exactly the phone's seconds, then the ONU's, then the OLT's."""
    setup = _climbing_setup(marginal_curve, marginal_prices)
    metrics = run(setup, 0.5, "recserve")[0]
    phone_s = marginal_prices.seconds("user", 1, 100, 200)
    onu_s = marginal_prices.seconds("onu", 1, 100, 200)
    olt_s = marginal_prices.seconds("olt", 1, 100, 200)
    answered_on_phone, answered_on_onu = 3 * phone_s, 3 * (phone_s + onu_s)
    expected = (answered_on_phone + answered_on_onu + phone_s + onu_s + olt_s) / 7
    assert metrics["latency_s_mean"] == pytest.approx(expected)


def test_the_onu_serves_queries_in_the_order_they_reach_it(marginal_curve, marginal_prices):
    """A message sent later that clears the phone first is served first, and the other waits."""
    long_prompt, short_prompt = 5000, 10
    answers = {
        "sure": {tier: _answer(0.9) for tier in TIERS},
        "slow": {"user": _answer(0.1, long_prompt), "onu": _answer(0.9), "olt": _answer(0.9)},
        "fast": {"user": _answer(0.1, short_prompt), "onu": _answer(0.9), "olt": _answer(0.9)},
    }
    stream = [
        ("sure", 0.0, 0),
        ("sure", 0.1, 0),
        ("sure", 0.2, 0),
        ("slow", SENT_S / 3600, 0),
        ("fast", (SENT_S + 1) / 3600, 0),
    ]
    setup = RunSetup(
        answers=answers,
        stream=stream,
        curve=marginal_curve,
        prices=marginal_prices,
        static_rates={},
        settings=_settings(stats="published", escalation="beta", onu_queue=True),
        schedule_key=lambda hour: (0, int(hour) % 24),
    )
    metrics = run(setup, 0.5, "recserve")[0]
    sure_s = marginal_prices.seconds("user", 1, 100, 200)
    slow_phone = marginal_prices.seconds("user", 1, long_prompt, 200)
    fast_phone = marginal_prices.seconds("user", 1, short_prompt, 200)
    onu_s = marginal_prices.seconds("onu", 1, 100, 200)
    assert 1 + fast_phone < slow_phone < 1 + fast_phone + onu_s  # the setup under test
    fast_latency = fast_phone + onu_s
    slow_latency = (1 + fast_phone + onu_s + onu_s) - 0.0
    expected = (3 * sure_s + fast_latency + slow_latency) / 5
    assert metrics["latency_s_mean"] == pytest.approx(expected)


@pytest.mark.parametrize("stats", ["pooled", "published", "household"])
@pytest.mark.parametrize("policy", ["static_hour", "broadcast", "oracle", "piggyback"])
def test_decisions_happen_in_time_order(monkeypatch, setup, policy, stats):
    """No decision reads the OLT's rate at a moment before one already read."""
    reads = []
    real = cascade.believed_olt_rate

    def spied(policy_name, run_setup, server, learned, hour):
        reads.append(hour)
        return real(policy_name, run_setup, server, learned, hour)

    monkeypatch.setattr(cascade, "believed_olt_rate", spied)
    setup.settings = _settings(stats=stats, timing="after-answer", escalation="beta")
    run(setup, 0.5, policy)
    assert reads == sorted(reads)


@pytest.mark.parametrize("interval", [10.0, 120.0])
def test_a_tier_never_reads_a_broadcast_older_than_the_interval(monkeypatch, setup, interval):
    """When a tier reads the broadcast, the OLT last sent it within one interval of that moment."""
    sent = []
    ages = []
    real_send = cascade.OltServer.send_broadcast
    real_belief = cascade.believed_olt_rate

    def send(server):
        sent.append(server.clock)
        real_send(server)

    def spied(policy_name, run_setup, server, learned, hour):
        if sent:
            ages.append(hour * 3600 - sent[-1])
        return real_belief(policy_name, run_setup, server, learned, hour)

    monkeypatch.setattr(cascade.OltServer, "send_broadcast", send)
    monkeypatch.setattr(cascade, "believed_olt_rate", spied)
    setup.settings = _settings(
        stats="published", escalation="beta", timing="after-answer", broadcast_interval_s=interval
    )
    run(setup, 0.5, "broadcast")
    assert ages
    assert 0.0 <= min(ages) and max(ages) < interval


@pytest.mark.parametrize("policy", POLICIES)
def test_the_energy_of_each_tier_adds_up_to_the_total(setup, policy):
    """However a query's hops are split in time, no joule is lost or counted twice."""
    setup.settings = _settings(timing="after-answer")
    metrics = run(setup, 0.5, policy)[0]
    split = sum(metrics[f"J_{tier}_per_query"] for tier in TIERS)
    assert split == pytest.approx(metrics["J_per_query"])


@pytest.mark.parametrize("policy", ["recserve", "static_hour", "broadcast", "piggyback"])
def test_a_run_is_exactly_repeatable(setup, policy):
    """Events at the same moment are always taken in the same order."""
    setup.settings = _settings(timing="after-answer")
    first, second = run(setup, 0.5, policy)[0], run(setup, 0.5, policy)[0]
    assert {key: value for key, value in first.items() if value == value} == {
        key: value for key, value in second.items() if value == value
    }


def test_run_observed_cells_hold_the_olt_report_over_each_cell(setup, curve):
    """Each cell lies between what joining adds and a lone query, the report's two ends."""
    above_idle = published_rates("marginal")
    setup.prices = Energy(curve, above_idle, "marginal", published_speeds(), "added", above_idle)
    _, _, observed = run(setup, 0.7, "recserve")
    idle, busy = setup.prices.added_rates(0), setup.prices.added_rates(1)
    assert len(observed) == 24
    for prompt_rate, generated_rate, _ in observed.values():
        assert busy[0] <= prompt_rate <= idle[0]
        assert busy[1] <= generated_rate <= idle[1]


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


def test_floor_tables_hold_the_cheapest_report_in_every_cell(curve, marginal_prices, prices):
    """A table started at the floor reads a full batch's rate at every hour."""
    added = cascade.floor_tables(marginal_prices)
    assert added["static_day"] == (marginal_prices.curve.pf_slope, marginal_prices.curve.dec_slope)
    assert added["static_hour"] == {}
    assert cascade.floor_tables(prices)["static_day"] == curve.rates(cascade.MAX_BATCH)


def test_table_change_compares_each_cell_with_what_it_ran_on():
    """A cell the round ran without is compared with the day's rate it fell back to."""
    before = {"static_day": (1.0, 2.0), "static_hour": {(0, 3): (2.0, 4.0)}}
    after = {"static_day": (1.1, 2.2), "static_hour": {(0, 3): (2.0, 4.0), (0, 4): (1.5, 2.0)}}
    assert cascade.table_change(before, after, "static_hour") == pytest.approx(0.5 / 4)
    assert cascade.table_change(before, after, "static_day") == pytest.approx(0.1)


def test_relearning_starts_at_the_floor_and_stops_once_settled(monkeypatch, setup):
    """The first round runs on the floor table; relearning stops when a round barely moves."""
    rounds = []
    observed = [
        {"static_day": (0.5, 1.0), "static_hour": {(0, 3): (0.5, 1.0)}},
        {"static_day": (0.6, 1.2), "static_hour": {(0, 3): (0.6, 1.2)}},
        {"static_day": (0.6, 1.2), "static_hour": {(0, 3): (0.601, 1.201)}},
        {"static_day": (9.0, 9.0), "static_hour": {(0, 3): (9.0, 9.0)}},
    ]

    def fake_calibrate(run_setup, _beta, _policy="recserve"):
        rounds.append(run_setup.static_rates)
        return observed[len(rounds) - 1]

    monkeypatch.setattr(cascade, "calibrate", fake_calibrate)
    tables = cascade.self_consistent_tables(setup, 0.5, "static_hour")
    assert rounds[0] == cascade.floor_tables(setup.prices)
    assert len(rounds) == 3
    assert tables == observed[2]


def test_relearning_stops_after_the_most_rounds(monkeypatch, setup):
    """A table that never settles is relearned SELF_CALIBRATION_ROUNDS times, no more."""
    calls = []

    def fake_calibrate(*_):
        calls.append(1)
        return {"static_day": (len(calls), len(calls)), "static_hour": {}}

    monkeypatch.setattr(cascade, "calibrate", fake_calibrate)
    cascade.self_consistent_tables(setup, 0.5, "static_day")
    assert len(calls) == cascade.SELF_CALIBRATION_ROUNDS


def test_relearning_rises_from_the_floor_to_what_the_olt_reports(setup):
    """On a real run the table settles above the floor, where the OLT's report agrees with it."""
    floor = cascade.floor_tables(setup.prices)["static_day"]
    tables = cascade.self_consistent_tables(setup, 0.5, "static_day")
    assert tables["static_day"][1] > floor[1]
    again = calibrate(type(setup)(**{**setup.__dict__, "static_rates": tables}), 0.5, "static_day")
    assert cascade.table_change(tables, again, "static_day") < 0.05


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


@pytest.mark.parametrize("stats", ["published", "pooled", "household"])
@pytest.mark.parametrize("escalation", ["beta", "learned"])
@pytest.mark.parametrize("windows", ["shared", "household"])
@pytest.mark.parametrize("timing", ["fixed", "after-answer"])
def test_run_settles_every_query_in_every_mode(setup, stats, escalation, windows, timing):
    """Every combination of --stats, --escalation, --windows and --timing serves the stream."""
    setup.settings = _settings(stats=stats, escalation=escalation, windows=windows, timing=timing)
    metrics, per_hour, _ = run(setup, 0.6, "broadcast")
    assert sum(metrics[f"final_{tier}"] for tier in TIERS) == pytest.approx(1.0)
    assert sum(hour["queries"] for hour in per_hour) == len(setup.stream)


def test_run_published_decides_from_the_first_query(setup):
    """Lengths shipped with the models leave only the OLT's rate to wait for."""
    setup.settings = _settings(stats="published", escalation="beta", warmup=10**6)
    metrics, _, _ = run(setup, 0.9, "oracle")
    assert metrics["forwarded_on_arrival"] + metrics["skipped_on_escalation"] > 0
    setup.settings = _settings(stats="household", escalation="beta", warmup=10**6)
    metrics, _, _ = run(setup, 0.9, "oracle")
    assert metrics["forwarded_on_arrival"] + metrics["skipped_on_escalation"] == 0


@pytest.mark.parametrize("timing, batched", [("after-answer", False), ("fixed", True)])
def test_run_after_answer_waits_for_the_answer_before_the_follow_up(setup, timing, batched):
    """Two messages sent together share the OLT's batch; after-answer sends the second later."""
    setup.stream = [(0, 1.0, 0), (1, 1.0, 0)]  # both at 01:00, the second linked to the first
    setup.follow = [(1, 0.0), None]
    setup.settings = _settings(timing=timing)
    metrics, _, _ = run(setup, 0.5, "olt_alone")
    assert metrics["final_olt"] == 1.0
    assert (metrics["mean_olt_batch"] > 1.0) == batched
    if not batched:
        assert metrics["mean_olt_batch"] == pytest.approx(1.0)


def test_run_context_carry_makes_a_follow_up_dearer(setup):
    """A follow-up whose prompt holds the first turn costs more prefill than one that does not."""
    setup.stream = [(0, 1.0, 0), (1, 1.0, 0)]
    setup.follow = [(1, 0.0), None]
    joules = {}
    for context in ("none", "carry"):
        setup.settings = _settings(timing="after-answer", context=context)
        joules[context] = run(setup, 0.5, "olt_alone")[0]["J_per_query"]
    assert joules["carry"] > joules["none"]


def test_run_onu_queue_makes_a_second_query_wait(setup):
    """Two queries at once from one home: with a queue, the second waits for the first."""
    setup.stream = [(0, 1.0, 0), (1, 1.0, 0)]
    setup.follow = [None, None]
    latency = {}
    for queue in (False, True):
        setup.settings = _settings(onu_queue=queue)
        latency[queue] = run(setup, 0.5, "onu_alone")[0]["latency_s_mean"]
    assert latency[True] > latency[False]


@pytest.mark.parametrize("policy", POLICIES)
def test_run_settles_every_query_under_the_realistic_settings(setup, policy):
    """Published lengths, beta, own windows, after-answer timing, context and the ONU queue."""
    setup.settings = _settings(
        stats="published",
        escalation="beta",
        windows="household",
        timing="after-answer",
        context="carry",
        onu_queue=True,
    )
    metrics, per_hour, _ = run(setup, 0.6, policy)
    assert sum(metrics[f"final_{tier}"] for tier in TIERS) == pytest.approx(1.0)
    assert sum(hour["queries"] for hour in per_hour) == len(setup.stream)


@pytest.mark.parametrize("policy", ["recserve", "broadcast", "olt_alone"])
def test_run_breakdowns_add_up_to_the_totals(setup, policy):
    """Energy by tier sums to J per query, and the burden by answering tier to the burden."""
    setup.settings = _settings(stats="published", escalation="beta", timing="after-answer")
    metrics, _, _ = run(setup, 0.7, policy)
    assert sum(metrics[f"J_{tier}_per_query"] for tier in TIERS) == pytest.approx(
        metrics["J_per_query"]
    )
    assert sum(metrics[f"comm_MB_{tier}_per_1k"] for tier in TIERS) == pytest.approx(
        metrics["comm_MB_per_1k_queries"]
    )
    assert metrics["comm_MB_user_per_1k"] == 0.0  # a phone's own answer crosses no hop
    assert 0 < metrics["comm_MB_followups_per_1k"] <= metrics["comm_MB_per_1k_queries"]
