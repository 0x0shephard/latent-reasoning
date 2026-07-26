"""Dependency-free accuracy gate for the official CODI reproduction."""
from __future__ import annotations

import re
from typing import Mapping


def extract_official_answer_number(text: str) -> float:
    """Reproduce the released ``test.py`` last-number extraction exactly."""
    matches = re.findall(r"-?\d+\.?\d*", str(text).replace(",", ""))
    return float(matches[-1]) if matches else float("inf")


def official_answers_match(text: str, gold: object) -> bool:
    return extract_official_answer_number(text) == float(str(gold))


def build_accuracy_gate(
    *,
    results: Mapping[str, float],
    evaluated_counts: Mapping[str, int],
    expected_counts: Mapping[str, int],
    published_accuracy: Mapping[str, float],
    primary_dataset: str,
    absolute_tolerance: float,
) -> dict:
    """Compare only full-benchmark results with the preregistered reference values."""
    comparisons = {}
    for name, accuracy in results.items():
        expected_count = int(expected_counts[name])
        full = int(evaluated_counts[name]) == expected_count
        reference_value = published_accuracy.get(name)
        reference = float(reference_value) if reference_value is not None else None
        delta = float(accuracy) - reference if reference is not None else None
        comparisons[name] = {
            "accuracy": float(accuracy),
            "published_accuracy": reference,
            "delta": delta,
            "absolute_delta": abs(delta) if delta is not None else None,
            "evaluated_count": int(evaluated_counts[name]),
            "expected_count": expected_count,
            "full_benchmark": full,
            "within_tolerance": (
                full and abs(delta) <= absolute_tolerance
                if delta is not None
                else None
            ),
        }

    if primary_dataset not in comparisons:
        status = "primary_dataset_not_evaluated"
    elif not comparisons[primary_dataset]["full_benchmark"]:
        status = "diagnostic_only_partial_evaluation"
    elif comparisons[primary_dataset]["published_accuracy"] is None:
        status = "primary_reference_missing"
    elif comparisons[primary_dataset]["within_tolerance"]:
        status = "passed"
    else:
        status = "failed"
    return {
        "status": status,
        "primary_dataset": primary_dataset,
        "absolute_tolerance": float(absolute_tolerance),
        "comparisons": comparisons,
        "interpretation": (
            "Passing establishes evaluator/checkpoint compatibility, not a new "
            "performance claim. Partial evaluations never pass the gate."
        ),
    }
