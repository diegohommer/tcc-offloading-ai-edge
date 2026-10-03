"""Tests for comparing policies at equal accuracy."""

import math

import pytest

from frontier import frontier, iso_accuracy


def _row(policy, accuracy, joules):
    """Return one result row."""
    return {"policy": policy, "accuracy": accuracy, "J_per_query": joules}


def test_iso_accuracy_reads_recserve_at_the_same_accuracy():
    """A policy at 0.70 is compared with RecServe interpolated to 0.70."""
    rows = [
        _row("recserve", 0.6, 100.0),
        _row("recserve", 0.8, 200.0),
        _row("broadcast", 0.7, 120.0),
    ]
    iso_accuracy(rows)
    assert rows[2]["J_recserve_same_accuracy"] == pytest.approx(150.0)
    assert rows[2]["saving_same_accuracy"] == pytest.approx(0.2)


def test_iso_accuracy_outside_recserve_range_is_nan():
    """No saving is claimed beyond the accuracies RecServe reaches."""
    rows = [
        _row("recserve", 0.6, 100.0),
        _row("recserve", 0.8, 200.0),
        _row("broadcast", 0.9, 150.0),
    ]
    iso_accuracy(rows)
    assert math.isnan(rows[2]["saving_same_accuracy"])


def test_frontier_drops_points_another_beta_beats():
    """A point both less accurate and costlier than another plays no part."""
    rows = [_row("oracle", 0.6, 100.0), _row("oracle", 0.7, 300.0), _row("oracle", 0.8, 200.0)]
    assert frontier(rows, [0.7])["0.70"] == pytest.approx(150.0)


def test_frontier_below_the_range_is_the_cheapest_point():
    """Reaching at least a low accuracy costs the policy's least accurate point."""
    rows = [_row("oracle", 0.6, 100.0), _row("oracle", 0.8, 200.0)]
    assert frontier(rows, [0.5])["0.50"] == pytest.approx(100.0)


def test_frontier_above_the_range_is_none():
    """An accuracy the policy never reaches has no cost."""
    rows = [_row("oracle", 0.6, 100.0), _row("oracle", 0.8, 200.0)]
    assert frontier(rows, [0.85])["0.85"] is None
