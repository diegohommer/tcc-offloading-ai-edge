"""Tests for the case study's summary tables."""

import json

import pytest

import summarize_study
from summarize_study import at, at_column, header, percent, queued, saving


def _run(recserve_joules=200.0, broadcast_joules=150.0, queued_share=0.0):
    """Return one run's JSON for 1,000 households, RecServe and broadcast, two betas each."""
    rows = [
        {
            "subscribers": 1000,
            "policy": policy,
            "beta": beta,
            "accuracy": accuracy,
            "J_per_query": joules * scale,
            "latency_s_mean": 10.0 * scale,
            "pon_MB_per_1k_queries": 0.5,
            "comm_MB_per_1k_queries": 2.0,
            "olt_queued": queued_share,
        }
        for policy, joules in (("recserve", recserve_joules), ("broadcast", broadcast_joules))
        for beta, accuracy, scale in ((0.3, 0.7, 1.0), (0.9, 0.9, 2.0))
    ]
    return {
        "args": {
            "accounting": "marginal",
            "olt_scale": "1.0",
            "users_per_home": "1.0",
            "per_user_day": "3.6",
            "shared_stats": "True",
            "policies": "recserve,broadcast",
        },
        "fixed_J_per_query": {"onu": 91.0},
        "rows": rows,
        "frontiers": [
            {
                "subscribers": 1000,
                "policy": "recserve",
                "J_at_accuracy": {"0.70": recserve_joules, "0.80": 1.5 * recserve_joules},
            },
            {
                "subscribers": 1000,
                "policy": "broadcast",
                "J_at_accuracy": {"0.70": broadcast_joules, "0.80": 1.5 * broadcast_joules},
            },
        ],
    }


def test_saving_is_the_share_of_energy_saved():
    """150 J against 200 J saves a quarter; a missing side saves nothing measurable."""
    assert saving(150.0, 200.0) == pytest.approx(0.25)
    assert saving(None, 200.0) is None


def test_percent_shows_the_mean_and_the_spread_across_seeds():
    """Savings print as a signed mean, with (min..max) when there are several seeds."""
    assert percent([0.1, 0.3]) == "+20.0% (+10%..+30%)"
    assert percent([0.1, 0.3], spread=False) == "+20.0%"
    assert percent([None]) == "-"


def test_header_rule_has_one_cell_per_column():
    """The rule under a header has as many cells as the header."""
    assert header(["homes", "acc", "RecServe"]) == ["| homes | acc | RecServe |", "|---|---|---|"]


def test_at_reads_each_policy_frontier():
    """J per query at an accuracy comes from each policy's frontier."""
    assert at(_run(), 1000, "0.70") == {"recserve": 200.0, "broadcast": 150.0}


def test_at_column_interpolates_along_the_same_frontier():
    """Latency at 0.80 sits halfway between the betas at 0.70 and 0.90."""
    assert at_column(_run(), 1000, "0.80", "latency_s_mean")["recserve"] == pytest.approx(15.0)


def test_queued_flags_a_population_whose_olt_ran_out_of_slots():
    """Over 1% of arrivals waiting marks the population."""
    assert not queued([_run(queued_share=0.005)], 1000)
    assert queued([_run(queued_share=0.02)], 1000)


def test_main_writes_the_summary(monkeypatch, tmp_path):
    """The runs found in the study folder become SUMMARY.md, headline first."""
    for seed in (7, 8):
        (tmp_path / f"study_main_seed{seed}.json").write_text(json.dumps(_run()))
    monkeypatch.setattr(summarize_study, "STUDY", tmp_path)
    assert summarize_study.main() == 0
    summary = (tmp_path / "SUMMARY.md").read_text()
    assert summary.startswith("# Case study — summary")
    assert "| `main` | +25.0% · - |" in summary


def test_main_without_runs_fails(monkeypatch, tmp_path):
    """An empty study folder is an error, not an empty summary."""
    monkeypatch.setattr(summarize_study, "STUDY", tmp_path)
    assert summarize_study.main() == 1
