"""Tests for launching the case study on Modal."""

import json
import subprocess
import sys

from run_study_modal import (
    beta_part,
    IMPLEMENTATION,
    merge_finished,
    merge_pieces,
    piece_name,
    rows_csv,
    splice_static,
    study_jobs,
    study_sizes,
)

SHORT = ["--test-days", "1", "--calibration-days", "1", "--betas", "0.5,0.9"]
"""A run short enough for a test."""


def _simulate(tmp_path, name, subscribers, flags=()):
    """Run simulate.py on the study's settings and return its {suffix: bytes}."""
    out = tmp_path / f"{name}.csv"
    command = [sys.executable, "src/simulate/simulate.py", "--config", "config/study.yaml"]
    command += ["--seed", "7", *SHORT, "--subscribers", subscribers, *flags, "--out", str(out)]
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


def test_merge_finished_handles_tags_with_a_dot(tmp_path):
    """A tag like onu0.5 holds a dot that must not be taken for a file extension."""
    parts = tmp_path / "parts"
    parts.mkdir()
    for size, subscribers in ((300, "300"), (600, "600")):
        piece = _simulate(tmp_path, f"piece{size}", subscribers)
        for suffix, content in piece.items():
            (parts / f"{piece_name('onu0.5', 7, size)}{suffix}").write_bytes(content)
    assert not merge_finished([("onu0.5", 7, [])], [300, 600], tmp_path)
    assert (tmp_path / "study_onu0.5_seed7.json").exists()


def test_splice_static_equals_a_run_made_with_self_calibration(tmp_path):
    """Static rows relearned per beta and spliced in give the run self-calibrated whole."""
    whole = _simulate(tmp_path, "whole", "300,600", ["--calibration", "self"])
    report = json.loads(_simulate(tmp_path, "baseline", "300,600")[".json"])
    relearn = ["--calibration", "self", "--policies", "recserve,static_day,static_hour"]
    new_rows = []
    for size in ("300", "600"):
        for beta in (0.5, 0.9):
            stem = piece_name("main", 7, int(size)) + beta_part(beta)
            piece = _simulate(tmp_path, stem, size, relearn + ["--betas", f"{beta:g}"])
            new_rows += json.loads(piece[".json"])["rows"]
    spliced = splice_static(report, new_rows)
    whole_report = json.loads(whole[".json"])
    assert rows_csv(spliced["rows"]) == whole[".csv"]
    assert spliced["frontiers"] == whole_report["frontiers"]
    assert spliced["args"]["calibration"] == "self"
