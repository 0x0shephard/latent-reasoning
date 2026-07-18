"""Position-sweep and causal difference-in-differences tests."""
from __future__ import annotations

from pathlib import Path

from src.eval.causal_compare import (
    analyze_position_sweep,
    compare_intervention_effects,
    render_did_markdown,
    render_position_sweep_markdown,
)
from src.eval.compare_runs import EvalRecord, EvalRun


def _run(name: str, correctness: list[bool], order=None) -> EvalRun:
    records = [
        EvalRecord(f"q{index}", str(index), str(index), correct)
        for index, correct in enumerate(correctness)
    ]
    if order is not None:
        records = [records[index] for index in order]
    return EvalRun(Path(name), {"set": tuple(records)})


def test_position_sweep_reports_only_baseline_effects():
    baseline = _run("baseline", [True, True, False, False])
    p0 = _run("p0", [False, True, False, False])
    p1 = _run("p1", [False, False, False, False])

    report = analyze_position_sweep(
        baseline, {0: p0, 1: p1}, bootstrap_samples=50, seed=3
    )

    assert report["positions"]["0"]["macro_effect"] == -0.25
    assert report["positions"]["1"]["macro_effect"] == -0.5
    assert report["most_harmful_position"] == 1
    assert "Most harmful position: **1**" in render_position_sweep_markdown(report)


def test_difference_in_differences_is_question_paired_and_reorders():
    left_baseline = _run("left_base", [True, True, False, False])
    left_changed = _run("left_changed", [False, True, False, False])
    right_baseline = _run("right_base", [True, True, True, False])
    # Deliberately reverse storage order; alignment must recover question identity.
    right_changed = _run(
        "right_changed", [False, False, False, False], order=[3, 2, 1, 0]
    )

    report = compare_intervention_effects(
        left_name="codi",
        left_baseline=left_baseline,
        left_intervention=left_changed,
        right_name="kava",
        right_baseline=right_baseline,
        right_intervention=right_changed,
        intervention_name="shuffle",
        bootstrap_samples=100,
        seed=9,
    )

    assert report["macro"]["codi_effect"] == -0.25
    assert report["macro"]["kava_effect"] == -0.75
    assert report["macro"]["difference_in_differences"] == -0.5
    rendered = render_did_markdown(report)
    assert "kava is harmed more than codi" in rendered
    assert "training-seed variance" in rendered
