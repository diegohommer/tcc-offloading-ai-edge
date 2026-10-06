"""Tests for launching the case study on Modal."""

import json
import subprocess
import sys

from run_study_modal import IMPLEMENTATION, merge_pieces, study_jobs, study_sizes

SHORT = ["--test-days", "1", "--calibration-days", "1", "--betas", "0.5,0.9"]
"""A run short enough for a test."""


def _simulate(tmp_path, name, subscribers):
    """Run simulate.py on the study's settings and return its {suffix: bytes}."""
    out = tmp_path / f"{name}.csv"
    command = [sys.executable, "src/simulate/simulate.py", "--config", "config/study.yaml"]
    command += ["--seed", "7", *SHORT, "--subscribers", subscribers, "--out", str(out)]
    subprocess.run(command, cwd=IMPLEMENTATION, check=True, capture_output=True)
    return {suffix: out.with_suffix(suffix).read_bytes() for suffix in (".csv", ".json")} | {
        ".txt": b""
    }


def test_study_jobs_are_the_runs_run_study_sh_lists():
    """Every scenario runs at three seeds, each with the flags run_study.sh gives it."""
    jobs = study_jobs()
    assert len(jobs) == 30
    assert {seed for _, seed, _ in jobs} == {7, 8, 9}
    flags = {tag: tag_flags for tag, _, tag_flags in jobs}
    assert flags["main"] == []
    assert flags["burst_all"] == ["--burst-sigma", "0.457", "--burst-hours", "12.3"]
    assert flags["perhousehold"] == ["--no-shared-stats"]


def test_study_sizes_are_the_household_counts_of_the_study():
    """The pieces of a run are the household counts config/study.yaml sweeps."""
    assert study_sizes() == [1000, 2000, 5000, 10000, 20000]


def test_merge_pieces_equals_the_run_made_whole(tmp_path):
    """A run split by household count and merged gives the same table and results as whole."""
    whole = _simulate(tmp_path, "whole", "300,600")
    merged = merge_pieces(
        [_simulate(tmp_path, "small", "300"), _simulate(tmp_path, "large", "600")]
    )
    assert merged[".csv"] == whole[".csv"]
    whole_report, merged_report = json.loads(whole[".json"]), json.loads(merged[".json"])
    for key in ("rows", "frontiers", "configs", "accuracy", "fixed_J_per_query"):
        assert merged_report[key] == whole_report[key]
    assert merged_report["args"]["subscribers"] == "300,600"
