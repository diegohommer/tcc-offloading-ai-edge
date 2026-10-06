"""Tests for launching the case study on Modal."""

from run_study_modal import study_jobs


def test_study_jobs_are_the_runs_run_study_sh_lists():
    """Every scenario runs at three seeds, each with the flags run_study.sh gives it."""
    jobs = study_jobs()
    assert len(jobs) == 30
    assert {seed for _, seed, _ in jobs} == {7, 8, 9}
    flags = {tag: tag_flags for tag, _, tag_flags in jobs}
    assert flags["main"] == []
    assert flags["burst_all"] == ["--burst-sigma", "0.457", "--burst-hours", "12.3"]
    assert flags["perhousehold"] == ["--no-shared-stats"]
