"""Paired evaluation analysis tests (no model or dataset downloads)."""
from __future__ import annotations

import json

import pytest

from src.eval.compare_runs import compare_runs, load_eval_run, render_markdown


def _write_run(root, rows_by_dataset):
    root.mkdir()
    for dataset, rows in rows_by_dataset.items():
        path = root / f"{dataset}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return load_eval_run(root)


def _rows(correct):
    generations = ["answer is 1", "", "no numeric answer", "answer is 4"]
    return [
        {
            "question": f"question {index}",
            "gold": str(index + 1),
            "generation": generations[index],
            "correct": value,
        }
        for index, value in enumerate(correct)
    ]


def test_paired_report_counts_disagreements_and_parse_failures(tmp_path):
    left = _write_run(tmp_path / "left", {"gsm8k": _rows([True, False, False, False])})
    right = _write_run(tmp_path / "right", {"gsm8k": _rows([True, True, True, True])})

    report = compare_runs(
        {"left": left, "right": right}, bootstrap_samples=200, seed=7
    )

    assert report["runs"]["left"]["datasets"]["gsm8k"] == {
        "count": 4,
        "correct": 1,
        "accuracy": 0.25,
        "blank_generations": 1,
        "unparseable_generations": 2,
    }
    paired = report["comparisons"][0]["datasets"]["gsm8k"]
    assert paired["accuracy_delta"] == 0.75
    assert paired["both_correct"] == 1
    assert paired["left_only_correct"] == 0
    assert paired["right_only_correct"] == 3
    assert paired["both_wrong"] == 0
    assert paired["mcnemar_exact_p"] == 0.25
    assert "right minus left" in render_markdown(report)


def test_bootstrap_report_is_deterministic(tmp_path):
    left = _write_run(tmp_path / "left", {"set": _rows([True, False, True, False])})
    right = _write_run(tmp_path / "right", {"set": _rows([False, True, True, True])})
    first = compare_runs({"a": left, "b": right}, bootstrap_samples=50, seed=11)
    second = compare_runs({"a": left, "b": right}, bootstrap_samples=50, seed=11)
    assert first == second


def test_alignment_accepts_different_row_order(tmp_path):
    left_rows = _rows([True, False, False, False])
    right_rows = list(reversed(_rows([True, False, False, False])))
    left = _write_run(tmp_path / "left", {"set": left_rows})
    right = _write_run(tmp_path / "right", {"set": right_rows})

    report = compare_runs({"left": left, "right": right}, bootstrap_samples=10)
    assert report["comparisons"][0]["datasets"]["set"]["accuracy_delta"] == 0


def test_alignment_accepts_equivalent_numeric_gold_formatting(tmp_path):
    left_rows = _rows([True, False, False, False])
    right_rows = _rows([True, False, False, False])
    for row in right_rows:
        row["gold"] = f'{row["gold"]}.0'
    left = _write_run(tmp_path / "left", {"set": left_rows})
    right = _write_run(tmp_path / "right", {"set": right_rows})

    report = compare_runs({"left": left, "right": right}, bootstrap_samples=10)
    assert report["comparisons"][0]["datasets"]["set"]["accuracy_delta"] == 0


def test_alignment_rejects_question_mismatch(tmp_path):
    left_rows = _rows([True, False, False, False])
    right_rows = _rows([True, False, False, False])
    right_rows[2]["question"] = "a different question"
    left = _write_run(tmp_path / "left", {"set": left_rows})
    right = _write_run(tmp_path / "right", {"set": right_rows})

    with pytest.raises(ValueError, match="example mismatch: left_only=1, right_only=1"):
        compare_runs({"left": left, "right": right}, bootstrap_samples=10)
