from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from scripts.collect_official_codi_endpoint_tsvc import (
    sample_endpoint_tsvc_partitions,
    verify_full_reproduction_gate,
)
from scripts.run_official_codi_endpoint_tsvc_utility import _completed_batch
from src.mech.endpoint_tsvc import (
    create_endpoint_moments,
    endpoint_tsvc_loss,
    fit_endpoint_tsvc_bases,
    match_gradient_norm,
    project_endpoint_residual,
    scope_layers,
    update_endpoint_moments,
    validate_endpoint_bases,
)


class TinyDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]


def test_three_way_question_partitions_are_disjoint_and_deterministic():
    rows = [
        {
            "question": f"  Problem   {index // 2} ",
            "answer": str(index + 1),
        }
        for index in range(24)
    ]
    dataset = TinyDataset(rows)
    first, metadata = sample_endpoint_tsvc_partitions(
        dataset,
        calibration_examples=4,
        update_examples=3,
        validation_examples=2,
        seed=11,
    )
    second, _ = sample_endpoint_tsvc_partitions(
        dataset,
        calibration_examples=4,
        update_examples=3,
        validation_examples=2,
        seed=11,
    )
    assert first == second
    assert not set(first["calibration"]) & set(first["update"])
    assert not set(first["calibration"]) & set(first["validation"])
    assert not set(first["update"]) & set(first["validation"])
    assert set(metadata["partition_sha256"]) == {
        "calibration",
        "update",
        "validation",
    }


def test_per_layer_svd_bases_are_orthonormal_and_unpooled():
    moments = create_endpoint_moments(layers=2, hidden_size=4)
    teacher = torch.zeros(8, 2, 4)
    student = torch.zeros_like(teacher)
    student[:, 0, 0] = torch.arange(1, 9)
    student[:, 1, 3] = torch.arange(1, 9) * 2
    update_endpoint_moments(moments, student, teacher)
    bases = fit_endpoint_tsvc_bases(moments, rank=2, random_seed=17)
    repeated = fit_endpoint_tsvc_bases(moments, rank=2, random_seed=17)
    validate_endpoint_bases(bases, layers=2, hidden_size=4)
    assert bases.top.shape == (2, 4, 2)
    assert abs(float(bases.top[0, 0, 0])) > 0.99
    assert abs(float(bases.top[1, 3, 0])) > 0.99
    assert torch.equal(bases.random, repeated.random)


def test_projection_and_complement_reconstruct_residual():
    residual = torch.randn(3, 2, 4)
    q, _ = torch.linalg.qr(torch.randn(2, 4, 2))
    projected = project_endpoint_residual(residual, q)
    complement = residual - projected
    assert torch.allclose(projected + complement, residual, atol=1e-6)


def test_full_rank_projected_loss_equals_native_full_loss():
    student = torch.randn(3, 12, 4, requires_grad=True)
    teacher = torch.randn(3, 12, 4)
    identity = torch.eye(4).expand(12, 4, 4).clone()
    native = endpoint_tsvc_loss(
        student,
        teacher,
        scope="endpoint_all_layers",
        mode="full",
    )
    projected = endpoint_tsvc_loss(
        student,
        teacher,
        scope="endpoint_all_layers",
        mode="projected",
        basis=identity,
    )
    assert torch.allclose(native, projected, atol=1e-6)


def test_layer11_scope_selects_only_final_block():
    student = torch.zeros(2, 12, 4)
    teacher = torch.zeros_like(student)
    student[:, 0] = 100.0
    student[:, 11] = 2.0
    teacher[:, 11, 0] = torch.tensor([0.0, 2.0])
    loss = endpoint_tsvc_loss(
        student,
        teacher,
        scope="endpoint_layer11",
        mode="full",
    )
    assert torch.isfinite(loss)
    assert scope_layers("endpoint_layer11", 12) == (11,)


def test_auxiliary_gradient_norm_is_matched_to_full_reference():
    gradients = (torch.tensor([3.0, 4.0]), None)
    reference = (torch.tensor([0.0, 10.0]), None)
    matched, report = match_gradient_norm(gradients, reference)
    assert torch.allclose(torch.linalg.vector_norm(matched[0]), torch.tensor(10.0))
    assert report["raw_auxiliary_gradient_norm"] == 5.0
    assert report["matched_auxiliary_gradient_norm"] == 10.0


def test_endpoint_gate_rejects_partial_reproduction_summary(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "gate": "passed",
                "checkpoint_revision": "revision",
                "evaluated_counts": {"gsm8k": 200},
            }
        )
    )
    cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(revision="revision"),
        eval=SimpleNamespace(
            expected_counts=SimpleNamespace(gsm8k=1319)
        ),
    )
    try:
        verify_full_reproduction_gate(summary, cfg)
    except RuntimeError as error:
        assert "complete official GSM8K" in str(error)
    else:
        raise AssertionError("partial reproduction summary was accepted")


def test_completed_batch_resume_rejects_changed_request(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_sha256": "original",
                "batch_index": 0,
                "update_indices": [1, 2],
                "validation_indices": [3, 4],
            }
        )
    )
    assert _completed_batch(
        path,
        request_sha256="original",
        batch_index=0,
        update_indices=[1, 2],
        validation_indices=[3, 4],
    )
    try:
        _completed_batch(
            path,
            request_sha256="changed",
            batch_index=0,
            update_indices=[1, 2],
            validation_indices=[3, 4],
        )
    except RuntimeError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("changed resume request was accepted")
