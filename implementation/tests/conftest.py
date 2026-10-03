"""Shared fixtures: the OLT's measured curve and the true energy rates built on it."""

# pylint: disable=redefined-outer-name

import pytest

from energy.three_tier import (
    olt_factor,
    olt_reference,
    OltCurve,
    published_rates,
    published_speeds,
)
from olt_energy import Energy


@pytest.fixture
def curve():
    """The OLT's measured curve at the whole-system boundary, average accounting."""
    return OltCurve(olt_reference()["batch_curve"], olt_factor("system", "average"))


@pytest.fixture
def marginal_curve():
    """The OLT's measured curve at the whole-system boundary, marginal accounting."""
    return OltCurve(olt_reference()["batch_curve"], olt_factor("system", "marginal"))


@pytest.fixture
def prices(curve):
    """True energy rates for every tier, average accounting."""
    return Energy(curve, published_rates("average"), "average", published_speeds())


@pytest.fixture
def marginal_prices(marginal_curve):
    """True energy rates for every tier, marginal accounting."""
    return Energy(marginal_curve, published_rates("marginal"), "marginal", published_speeds())
