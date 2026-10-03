"""Tests for the helpers behind the tier-level tables."""

import math

import pytest

from tier_energy import crossing, fmt_batch, table, welch_t


def test_crossing_interpolates_log_log():
    """A curve falling from 100 J at batch 1 to 1 J at batch 100 crosses 10 J at batch 10."""
    assert crossing([1, 100], [100.0, 1.0], 10.0) == pytest.approx(10.0)


def test_crossing_never_reached_is_none():
    """A curve that stays above the target has no crossover."""
    assert crossing([1, 2, 4], [50.0, 40.0, 30.0], 10.0) is None


def test_welch_t_separates_distinct_means():
    """Two well-separated groups give a large t; groups of one give NaN."""
    assert welch_t([10.0, 11.0, 12.0], [1.0, 2.0, 3.0]) > 5
    assert math.isnan(welch_t([1.0], [2.0, 3.0]))


def test_table_writes_markdown_rows():
    """A header, its rule, then one line per row."""
    assert table(["a", "b"], [[1, 2]]) == ["| a | b |", "|---|---|", "| 1 | 2 |"]


def test_fmt_batch_names_a_missing_crossover():
    """A crossover prints approximately; none says it is beyond the measured range."""
    assert fmt_batch(5.24) == "≈ 5.2"
    assert fmt_batch(None) == "not within 64"
