"""Tests for placing a WildChat user's country and state in a time zone."""

# pylint: disable=redefined-outer-name

import pytest

from local_time import LocalTime


@pytest.fixture(scope="module")
def mapper():
    """One mapper for every test: reading tzdata's tables once is enough."""
    return LocalTime()


@pytest.mark.parametrize(
    "country, state, zone",
    [
        ("Japan", None, "Asia/Tokyo"),
        ("United Kingdom", "England", "Europe/London"),
        ("China", "Guangdong", "Asia/Shanghai"),
        ("United States", "California", "America/Los_Angeles"),
        ("United States", "Arizona", "America/Phoenix"),
        ("Russia", "Novosibirsk Oblast", "Asia/Novosibirsk"),
        ("Russia", "Mariy-El Republic", "Europe/Moscow"),
        ("Canada", "Newfoundland and Labrador", "America/St_Johns"),
        ("Brazil", "Acre", "America/Rio_Branco"),
        ("Brazil", "Piaui", "America/Sao_Paulo"),
        ("Brazil", "Piauí", "America/Sao_Paulo"),
        ("Brazil", None, "America/Sao_Paulo"),
    ],
)
def test_zone_of_a_country_and_state(mapper, country, state, zone):
    """Single-zone countries by name, the large split ones by state, Brazil by its exceptions."""
    assert mapper.zone(country, state) == zone


@pytest.mark.parametrize(
    "country, state",
    [
        ("United States", None),
        ("United States", float("nan")),
        ("Russia", "Paris"),
        ("Atlantis", None),
    ],
)
def test_zone_is_none_when_it_cannot_be_told(mapper, country, state):
    """A split country without a known state, or an unknown country, is not guessed."""
    assert mapper.zone(country, state) is None
