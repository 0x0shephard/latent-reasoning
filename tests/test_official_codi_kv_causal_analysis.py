"""Tests for paired official CODI KV causal analysis."""
from __future__ import annotations

import json

from src.eval.official_codi_kv_causal_analysis import (
    analyze_causal_interventions,
    holm_adjust,
    render_causal_markdown,
)


def _write_condition(root, name, correctness):
    path = root / name
    path.mkdir(parents=True)
    rows = [
        {
            "question": f"q{index}",
            "gold": str(index),
            "generation": str(index) if correct else "wrong",
            "correct": correct,
        }
        for index, correct in enumerate(correctness)
    ]
    (path / "gsm8k.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_holm_adjust_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == 0.03
    assert adjusted["c"] == 0.06
    assert adjusted["b"] == 0.06


def test_primary_gate_detects_retain_and_remove_specificity(tmp_path):
    count = 40
    baseline = [True] * 20 + [False] * 20
    _write_condition(tmp_path, "baseline", baseline)
    for position in (4, 5):
        # Learned retention solves all random-retention examples plus ten more.
        _write_condition(
            tmp_path,
            f"retain_learned_p{position}",
            [True] * 30 + [False] * 10,
        )
        _write_condition(
            tmp_path,
            f"retain_random_p{position}",
            [True] * 20 + [False] * 20,
        )
        # Removing learned directions breaks ten examples that random removal keeps.
        _write_condition(
            tmp_path,
            f"remove_learned_p{position}",
            [True] * 10 + [False] * 30,
        )
        _write_condition(
            tmp_path,
            f"remove_random_p{position}",
            [True] * 20 + [False] * 20,
        )

    report = analyze_causal_interventions(
        tmp_path,
        scopes=("p4", "p5"),
        bootstrap_samples=400,
        seed=4,
    )
    assert report["gate"]["status"] == "learned_subspace_causality_supported"
    assert set(report["gate"]["supported_primary_tests"]) == {
        "retain_p4",
        "remove_p4",
        "retain_p5",
        "remove_p5",
    }
    assert report["comparisons"]["retain_p4"]["learned_minus_random"][
        "accuracy_delta"
    ] == 0.25
    assert report["comparisons"]["remove_p4"]["learned_minus_random"][
        "accuracy_delta"
    ] == -0.25
    markdown = render_causal_markdown(report)
    assert "Retain tests sufficiency" in markdown
    assert "Holm p" in markdown


def test_primary_gate_rejects_learned_equal_to_random(tmp_path):
    baseline = [True, False] * 10
    _write_condition(tmp_path, "baseline", baseline)
    for position in (4, 5):
        for mode in ("retain", "remove"):
            _write_condition(
                tmp_path,
                f"{mode}_learned_p{position}",
                baseline,
            )
            _write_condition(
                tmp_path,
                f"{mode}_random_p{position}",
                baseline,
            )
    report = analyze_causal_interventions(
        tmp_path,
        scopes=("p4", "p5"),
        bootstrap_samples=50,
    )
    assert report["gate"]["status"] == "learned_subspace_causality_not_supported"
    assert report["gate"]["supported_primary_tests"] == []
