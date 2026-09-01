from __future__ import annotations

from src.mech.qwen_trajectory import (
    evenly_spaced_indices,
    final_answer_span,
    token_indices_overlapping_span,
)


def test_final_answer_span_prefers_last_balanced_box():
    text = r"A trial gives 9. Therefore the result is \boxed{\frac{21}{3}}."
    start, stop, rule = final_answer_span(text)
    assert text[start:stop] == r"\frac{21}{3}"
    assert rule == "boxed"


def test_final_answer_span_supports_explicit_marker_and_fallback():
    marked = "We used 12 objects. Final answer: $1,234.50"
    start, stop, rule = final_answer_span(marked)
    assert marked[start:stop] == "$1,234.50"
    assert rule == "final_marker"

    fallback = "After checking 3 cases, the result is 17."
    start, stop, rule = final_answer_span(fallback)
    assert fallback[start:stop] == "17"
    assert rule == "last_number_fallback"


def test_token_overlap_and_even_sampling_are_deterministic():
    offsets = [(0, 2), (2, 5), (6, 8), (8, 11)]
    assert token_indices_overlapping_span(offsets, (4, 9)) == [1, 2, 3]
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]
    assert evenly_spaced_indices(3, 4) == [0, 1, 2]
