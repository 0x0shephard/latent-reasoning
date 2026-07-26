"""Official CODI accuracy-gate tests; no model or dataset downloads."""
from __future__ import annotations

from src.eval.official_codi_gate import (
    build_accuracy_gate,
    extract_official_answer_number,
    official_answers_match,
)


EXPECTED = {"gsm8k": 1319, "svamp": 1000}
PUBLISHED = {"gsm8k": 0.437, "svamp": 0.429}


def test_official_scorer_uses_last_number_and_strips_commas():
    assert extract_official_answer_number("First 5. The answer is 1,200.") == 1200.0
    assert official_answers_match("First 5. The answer is 1,200.", 1200)
    assert not official_answers_match("No numeric answer", 0)


def test_partial_evaluation_never_passes_gate():
    gate = build_accuracy_gate(
        results={"gsm8k": 0.44},
        evaluated_counts={"gsm8k": 200},
        expected_counts=EXPECTED,
        published_accuracy=PUBLISHED,
        primary_dataset="gsm8k",
        absolute_tolerance=0.03,
    )
    assert gate["status"] == "diagnostic_only_partial_evaluation"
    assert gate["comparisons"]["gsm8k"]["within_tolerance"] is False


def test_full_primary_evaluation_passes_within_tolerance():
    gate = build_accuracy_gate(
        results={"gsm8k": 0.42},
        evaluated_counts={"gsm8k": 1319},
        expected_counts=EXPECTED,
        published_accuracy=PUBLISHED,
        primary_dataset="gsm8k",
        absolute_tolerance=0.03,
    )
    assert gate["status"] == "passed"
    assert gate["comparisons"]["gsm8k"]["within_tolerance"] is True


def test_full_primary_evaluation_fails_outside_tolerance():
    gate = build_accuracy_gate(
        results={"gsm8k": 0.13},
        evaluated_counts={"gsm8k": 1319},
        expected_counts=EXPECTED,
        published_accuracy=PUBLISHED,
        primary_dataset="gsm8k",
        absolute_tolerance=0.03,
    )
    assert gate["status"] == "failed"


def test_nonprimary_dataset_without_comparable_reference_is_reported():
    gate = build_accuracy_gate(
        results={"gsm8k": 0.437, "svamp": 0.40},
        evaluated_counts={"gsm8k": 1319, "svamp": 1000},
        expected_counts=EXPECTED,
        published_accuracy={"gsm8k": 0.437},
        primary_dataset="gsm8k",
        absolute_tolerance=0.03,
    )
    assert gate["status"] == "passed"
    assert gate["comparisons"]["svamp"]["published_accuracy"] is None
    assert gate["comparisons"]["svamp"]["within_tolerance"] is None
