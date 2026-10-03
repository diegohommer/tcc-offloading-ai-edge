"""Tests for the simulator's settings."""

import sys

import pytest

from energy.three_tier import ROOT
from simulate import check_combinations, load_config, read_settings


def _read_settings_from(monkeypatch, *argv):
    """Parse settings as the command line `simulate.py <argv>` would."""
    monkeypatch.setattr(sys, "argv", ["simulate.py", *argv])
    return read_settings()


def test_load_config_flattens_groups_and_joins_lists(tmp_path):
    """Groups are only for reading: every setting lands at the top, lists as comma strings."""
    config = tmp_path / "run.yaml"
    config.write_text("traffic:\n  subscribers: [1000, 2000]\npolicy:\n  delta: 0.1\n")
    assert load_config(config) == {"subscribers": "1000,2000", "delta": 0.1}


def test_load_config_extends_a_base_file(tmp_path):
    """A file that extends another keeps the base's values and overrides its own."""
    (tmp_path / "base.yaml").write_text("policy:\n  delta: 0.0\n  stale_factor: 4.0\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\npolicy:\n  delta: 0.2\n")
    assert load_config(child) == {"delta": 0.2, "stale_factor": 4.0}


def test_check_combinations_requires_recserve():
    """Savings are read against RecServe, so it must run."""
    assert check_combinations(["recserve", "broadcast"])
    assert not check_combinations(["broadcast", "oracle"])


def test_check_combinations_rejects_unknown_policies():
    """A misspelled policy stops the run before it starts."""
    assert not check_combinations(["recserve", "brodcast"])


@pytest.mark.parametrize("config", ["simulation.yaml", "study.yaml"])
def test_read_settings_accepts_the_shipped_configs(monkeypatch, config):
    """Every setting in the shipped files has a flag, and every flag a setting."""
    settings = _read_settings_from(monkeypatch, "--config", str(ROOT / "config" / config))
    assert settings is not None


def test_read_settings_flag_overrides_the_file(monkeypatch):
    """A command-line flag beats the settings file for that run."""
    settings = _read_settings_from(monkeypatch, "--subscribers", "5000", "--accounting", "average")
    assert settings.subscribers == "5000"
    assert settings.accounting == "average"


def test_read_settings_rejects_an_unknown_setting(monkeypatch, tmp_path):
    """A setting with no flag behind it is reported rather than ignored."""
    config = tmp_path / "bad.yaml"
    config.write_text(
        f"extends: {ROOT / 'config' / 'simulation.yaml'}\ntraffic:\n  surge_factor: 2\n"
    )
    assert _read_settings_from(monkeypatch, "--config", str(config)) is None


def test_read_settings_rejects_a_value_outside_the_choices(monkeypatch, tmp_path):
    """A setting's value must be one of its flag's choices."""
    config = tmp_path / "bad.yaml"
    config.write_text(
        f"extends: {ROOT / 'config' / 'simulation.yaml'}\nenergy:\n  accounting: shared\n"
    )
    assert _read_settings_from(monkeypatch, "--config", str(config)) is None
