"""Tests for GSM8K prompting and scoring."""

# pylint: disable=protected-access

import pytest

from gsm8k import _normalize_number, build_prompt, extract_answer, is_correct, parse_reference


@pytest.mark.parametrize(
    "raw, expected",
    [("1,000", "1000"), ("1000.0", "1000"), ("42.", "42"), ("-3", "-3"), ("2.5", "2.5")],
)
def test_normalize_number_canonical_form(raw, expected):
    """Different spellings of the same number compare equal."""
    assert _normalize_number(raw) == expected


def test_normalize_number_rejects_non_numbers():
    """Text that is not a number, or nothing at all, normalizes to None."""
    assert _normalize_number("abc") is None
    assert _normalize_number(None) is None


def test_build_prompt_carries_the_stripped_question():
    """The question is placed in the template without surrounding whitespace."""
    prompt = build_prompt("  How many apples?  \n")
    assert prompt.endswith("Problem: How many apples?")
    assert "#### <number>" in prompt


def test_extract_answer_prefers_the_last_marked_answer():
    """The last '#### n' wins over any other number in the text."""
    assert extract_answer("5 + 3 = 8\n#### 7\nactually\n#### 8") == "8"


def test_extract_answer_falls_back_to_the_last_number():
    """A model that ignores the format is scored on its last number."""
    assert extract_answer("She has 3 apples, then 1,200 more.") == "1200"


def test_extract_answer_without_numbers_is_none():
    """No number in the text means no answer."""
    assert extract_answer("I do not know.") is None


def test_is_correct_compares_normalized_numbers():
    """'#### 1,000.0' answers a gold '1000'."""
    assert is_correct("so #### 1,000.0", "1000")
    assert not is_correct("so #### 999", "1000")
    assert not is_correct("no idea", "1000")


def test_parse_reference_returns_final_number_and_steps():
    """The gold answer is the marked number; difficulty counts the calculator steps."""
    raw = "He buys <<2*3=6>>6 and <<6+4=10>>10.\n#### 10"
    assert parse_reference(raw) == ("10", 2)
