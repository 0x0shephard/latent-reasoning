"""Evaluation scoring tests (no model or dataset downloads)."""
from __future__ import annotations

import pytest

from src.data.answer_extract import normalize_gold
from src.eval.run_eval import score_generations


def test_score_generations_uses_numeric_exact_match():
    examples = [
        {"question": "q1", "gold": normalize_gold("1000000", "gsm_hard")},
        {"question": "q2", "gold": normalize_gold("42", "gsm_hard")},
    ]
    correct, accuracy = score_generations(
        ["The answer is: 1000100", "The answer is: 42"], examples
    )
    assert correct == 1
    assert accuracy == 0.5


def test_score_generation_count_must_match_examples():
    with pytest.raises(ValueError, match="count mismatch"):
        score_generations([], [{"question": "q", "gold": 1}])

