from __future__ import annotations

import pytest

from src.eval.compare_runs import EvalRecord, EvalRun
from src.eval.seed_compare import compare_seeded_runs, render_seed_markdown


def _run(tmp_path, name, correctness):
    records = tuple(
        EvalRecord(
            question=f"question {index}",
            gold=str(index + 1),
            generation=str(index + 1) if correct else "wrong",
            correct=correct,
        )
        for index, correct in enumerate(correctness)
    )
    return EvalRun(path=tmp_path / name, datasets={"math": records})


def test_seed_comparison_reports_paired_training_variation(tmp_path):
    runs = {
        "codi": {
            0: _run(tmp_path, "c0", [True, False]),
            1: _run(tmp_path, "c1", [False, False]),
            2: _run(tmp_path, "c2", [True, True]),
        },
        "kava": {
            0: _run(tmp_path, "k0", [True, True]),
            1: _run(tmp_path, "k1", [True, False]),
            2: _run(tmp_path, "k2", [True, True]),
        },
    }
    report = compare_seeded_runs(runs)

    assert report["seeds"] == [0, 1, 2]
    assert report["aggregates"]["codi"]["macro_mean"]["mean"] == pytest.approx(0.5)
    assert report["aggregates"]["kava"]["macro_mean"]["mean"] == pytest.approx(5 / 6)
    assert report["paired_seed_deltas"]["macro_mean"]["mean"] == pytest.approx(1 / 3)
    assert report["paired_seed_deltas"]["macro_mean"]["sample_sd"] == pytest.approx(
        0.2886751346
    )
    markdown = render_seed_markdown(report)
    assert "Multi-seed CODI vs KaVa" in markdown
    assert "0.3333" in markdown


def test_seed_comparison_rejects_unmatched_seed_sets(tmp_path):
    runs = {
        "codi": {0: _run(tmp_path, "c0", [True])},
        "kava": {1: _run(tmp_path, "k1", [True])},
    }
    with pytest.raises(ValueError, match="matched seeds"):
        compare_seeded_runs(runs)


def test_seed_comparison_refuses_premature_two_seed_report(tmp_path):
    runs = {
        "codi": {
            0: _run(tmp_path, "c0", [True]),
            1: _run(tmp_path, "c1", [True]),
        },
        "kava": {
            0: _run(tmp_path, "k0", [True]),
            1: _run(tmp_path, "k1", [True]),
        },
    }
    with pytest.raises(ValueError, match="at least three"):
        compare_seeded_runs(runs)
