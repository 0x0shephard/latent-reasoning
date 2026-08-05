from __future__ import annotations

import torch

from scripts.collect_official_codi_endpoint_tsvc import _normalized_question
from scripts.run_official_codi_endpoint_retention import (
    sample_fresh_retention_training,
)
from src.eval.official_codi_endpoint_retention_analysis import (
    analyze_endpoint_retention,
)
from src.mech.endpoint_retention import (
    RETENTION_METHODS,
    RETENTION_TRAINING_ARMS,
    RetentionBasis,
    _artifact_metadata,
    endpoint_retention_loss,
    retention_basis_for_arm,
    validate_retention_basis,
)


def _basis(name: str) -> RetentionBasis:
    value = torch.zeros(13, 768, 3)
    value[11] = torch.eye(768)[:, :3]
    value[12] = torch.eye(768)[:, :3]
    ranks = torch.zeros(13, dtype=torch.int64)
    ranks[11:] = 3
    return RetentionBasis(name, value, ranks, "basis.pt", "a", "b", "contract")


class TinyDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


def test_registered_retention_bases_are_rank_matched():
    for name in RETENTION_METHODS:
        validate_retention_basis(_basis(name))


def test_older_corrected_export_merges_manifest_and_parity_sidecars(tmp_path):
    basis_path = tmp_path / "basis.pt"
    basis_path.write_bytes(b"identity-only")
    (tmp_path / "run_manifest.json").write_text(
        '{"state":"complete","contract":"energy-contract","calibration_examples":5000}'
    )
    (tmp_path / "native_loss_gradient_parity.json").write_text(
        '{"status":"passed"}'
    )
    metadata = _artifact_metadata(basis_path, {"metadata": {"rank": 77}})
    assert metadata["contract"] == "energy-contract"
    assert metadata["calibration_examples"] == 5000
    assert metadata["native_parity_gate"]["status"] == "passed"


def test_selected_and_complement_ignore_opposite_coordinates():
    basis = _basis("energy")
    teacher = torch.randn(2, 13, 768)
    selected_student = teacher.clone()
    selected_student[:, 11:, :3] += 2
    complement_student = teacher.clone()
    complement_student[:, 11:, 3:] += 2
    selected_on_selected = endpoint_retention_loss(
        selected_student, teacher, mode="projected", basis=basis.basis, ranks=basis.ranks
    )
    complement_on_selected = endpoint_retention_loss(
        selected_student, teacher, mode="complement", basis=basis.basis, ranks=basis.ranks
    )
    selected_on_complement = endpoint_retention_loss(
        complement_student, teacher, mode="projected", basis=basis.basis, ranks=basis.ranks
    )
    complement_on_complement = endpoint_retention_loss(
        complement_student, teacher, mode="complement", basis=basis.basis, ranks=basis.ranks
    )
    assert selected_on_selected > 0 and complement_on_complement > 0
    assert torch.allclose(complement_on_selected, torch.zeros_like(complement_on_selected))
    assert torch.allclose(selected_on_complement, torch.zeros_like(selected_on_complement))


def test_arm_mapping_is_explicit():
    values = {name: _basis(name) for name in RETENTION_METHODS}
    assert retention_basis_for_arm("answer_only", values) == (None, "none")
    assert retention_basis_for_arm("full_common", values) == (None, "full")
    assert retention_basis_for_arm("parameter_aware_selected", values)[1] == "projected"
    assert retention_basis_for_arm("energy_complement", values)[1] == "complement"


def test_analysis_uses_paired_accuracy_and_reports_no_architectural_speedup():
    runs = []
    for arm_index, arm in enumerate(RETENTION_TRAINING_ARMS):
        for seed in (53, 59):
            correctness = [1, 1, int(arm_index % 2 == 0), 0]
            runs.append(
                {
                    "arm": arm,
                    "training_seed": seed,
                    "correctness": correctness,
                    "examples_per_second": 10 + arm_index,
                }
            )
    report = analyze_endpoint_retention(
        runs, bootstrap_samples=50, bootstrap_seed=7, noninferiority_margin=0.01
    )
    assert report["evaluated_examples_per_run"] == 4
    assert set(report["selected_vs_full_common"]) == set(RETENTION_METHODS)
    assert set(report["selected_vs_answer_only"]) == set(RETENTION_METHODS)
    assert not report["inference_speed_interpretation"]["expected_speedup_from_retention"]


def test_training_partition_excludes_all_three_completed_experiments():
    dataset = TinyDataset(
        [
            {"question": f"Question {index}", "cot": "work", "answer": "1"}
            for index in range(10_700)
        ]
    )
    selected, metadata = sample_fresh_retention_training(
        dataset, training_examples=8, seed=53
    )
    assert len(selected) == 8
    assert metadata["excluded_unique_questions"] == 10_632
    selected_questions = {
        _normalized_question(dataset[index]["question"]) for index in selected
    }
    assert len(selected_questions) == 8
