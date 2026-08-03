from __future__ import annotations

import json

import torch

from scripts.run_official_codi_endpoint_tsvc_corrected_utility import (
    _completed_batch,
)
from src.mech.endpoint_tsvc import (
    create_endpoint_moments,
    fit_endpoint_tsvc_bases,
    project_endpoint_residual,
    update_endpoint_moments,
)
from src.mech.endpoint_tsvc_corrected import (
    corrected_endpoint_tsvc_loss,
    corrected_scope_indices,
    relative_gradient_error,
    source_faithful_native_endpoint_loss,
    validate_corrected_bases,
)


def test_corrected_native_full_loss_and_gradient_parity():
    torch.manual_seed(3)
    student = torch.randn(4, 13, 8, requires_grad=True)
    teacher = torch.randn(4, 13, 8)
    reference = source_faithful_native_endpoint_loss(student, teacher)
    candidate = corrected_endpoint_tsvc_loss(
        student,
        teacher,
        scope="endpoint_all_states",
        mode="full",
    )
    reference_gradient = torch.autograd.grad(
        reference,
        student,
        retain_graph=True,
    )
    candidate_gradient = torch.autograd.grad(candidate, student)
    relative, cosine = relative_gradient_error(
        candidate_gradient,
        reference_gradient,
    )
    assert abs(float((reference - candidate).detach())) <= 1e-7
    assert relative <= 1e-6
    assert cosine >= 0.999999


def test_corrected_projection_and_complement_reconstruct_residual():
    torch.manual_seed(5)
    residual = torch.randn(3, 13, 8)
    sample = torch.randn(13, 8, 3)
    basis, _ = torch.linalg.qr(sample, mode="reduced")
    projected = project_endpoint_residual(residual, basis)
    complement = residual - projected
    assert torch.allclose(projected + complement, residual, atol=1e-6)


def test_corrected_full_rank_projection_reproduces_native_loss():
    torch.manual_seed(7)
    student = torch.randn(3, 13, 8, requires_grad=True)
    teacher = torch.randn(3, 13, 8)
    identity = torch.eye(8).expand(13, 8, 8).clone()
    native = corrected_endpoint_tsvc_loss(
        student,
        teacher,
        scope="endpoint_all_states",
        mode="full",
    )
    projected = corrected_endpoint_tsvc_loss(
        student,
        teacher,
        scope="endpoint_all_states",
        mode="projected",
        basis=identity,
    )
    assert torch.allclose(native, projected, atol=1e-6)


def test_corrected_layer11_is_hidden_state_tuple_index_12():
    assert corrected_scope_indices("endpoint_all_states", 13) == tuple(range(13))
    assert corrected_scope_indices("endpoint_layer11", 13) == (12,)


def test_corrected_basis_contract_uses_thirteen_independent_states():
    moments = create_endpoint_moments(layers=13, hidden_size=8)
    teacher = torch.zeros(16, 13, 8)
    student = torch.zeros_like(teacher)
    for state in range(13):
        student[:, state, state % 8] = torch.arange(1, 17)
    update_endpoint_moments(moments, student, teacher)
    bases = fit_endpoint_tsvc_bases(moments, rank=3, random_seed=20260803)
    validate_corrected_bases(bases, hidden_size=8)
    assert bases.top.shape == (13, 8, 3)


def test_corrected_resume_rejects_changed_request(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
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
        assert "mismatch" in str(error)
    else:
        raise AssertionError("changed corrected utility request was accepted")
