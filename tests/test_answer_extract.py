"""Unit tests for the answer-extraction / exact-match instrument (CPU, no downloads)."""
from __future__ import annotations

import pytest

from src.data.answer_extract import (
    answers_match,
    extract_final_number,
    normalize_gold,
    normalize_number,
)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("1,234", 1234.0),
        ("$50", 50.0),
        ("3.0", 3.0),
        ("12%", 12.0),
        ("-7", -7.0),
        ("360.", 360.0),
        ("", None),
        ("abc", None),
        ("+", None),
    ],
)
def test_normalize_number(token, expected):
    assert normalize_number(token) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is: 360", 360.0),
        ("...<<600-240=360>> The answer is: 360", 360.0),
        ("first 12 then 34 finally 56", 56.0),          # last number fallback
        ("The answer is 1,024 apples.", 1024.0),
        ("no numbers here", None),
        ("The answer is: -3.5", -3.5),
    ],
)
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


def test_answer_cue_beats_trailing_number():
    # Cue-following number wins even when another number appears later.
    assert extract_final_number("The answer is: 42 (see step 99)") == 42.0


def test_normalize_gold_gsm8k_main():
    gold = "Natural language reasoning...\n#### 18"
    assert normalize_gold(gold, "gsm8k_main") == 18.0


def test_normalize_gold_bare_number():
    assert normalize_gold("360", "gsm_hard") == 360.0
    assert normalize_gold(7.0, "svamp") == 7.0


@pytest.mark.parametrize(
    "pred,gold,ok",
    [
        ("The answer is: 360", 360.0, True),
        ("The answer is: 360.00004", 360.0, True),   # within tolerance
        ("The answer is: 361", 360.0, False),
        ("I don't know", 360.0, False),
        ("The answer is: 1,000", 1000.0, True),
    ],
)
def test_answers_match(pred, gold, ok):
    assert answers_match(pred, gold) is ok
