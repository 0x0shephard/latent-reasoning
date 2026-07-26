from __future__ import annotations

from scripts.run_official_codi_kv_gradient_signal import (
    sample_three_group_disjoint_splits,
)


class TinyDataset:
    def __init__(self, rows):
        self.rows = rows
        self._fingerprint = "tiny"

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]


def test_three_splits_exclude_prior_question_groups():
    rows = [
        {
            "question": f" Question {index} ",
            "answer": str(index + 1),
        }
        for index in range(12)
    ]
    dataset = TinyDataset(rows)
    calibration, update, validation, metadata = (
        sample_three_group_disjoint_splits(
            dataset,
            examples_per_split=3,
            seed=4,
            excluded_normalized_questions={"question 0", "question 1"},
        )
    )
    selected = calibration + update + validation
    assert len(selected) == 9
    assert len(set(selected)) == 9
    assert 0 not in selected and 1 not in selected
    assert metadata["excluded_prior_question_groups"] == 2
