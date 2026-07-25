"""CODI/KaVa trajectory-loss shape and stop-gradient tests."""
from __future__ import annotations

import pytest
import torch

from src.losses.trajectory_match import (
    TrajectoryMatchLoss,
    hidden_match_loss,
    key_match_loss,
    kv_match_loss,
    projected_key_match_loss,
)


def test_identical_hidden_and_kv_trajectories_have_zero_loss():
    hidden = torch.randn(2, 3, 7)
    kv = torch.randn(2, 3, 2, 4, 5)
    module = TrajectoryMatchLoss(hidden_weight=1, kv_weight=1)
    output = module(
        student_hidden=hidden,
        teacher_hidden=hidden.clone(),
        student_keys=kv,
        student_values=kv,
        teacher_keys=kv.clone(),
        teacher_values=kv.clone(),
    )
    assert output.total.item() == pytest.approx(0.0)


def test_teacher_is_stop_gradient_and_selected_layers_work():
    student = torch.zeros(2, 3, 4, requires_grad=True)
    teacher = torch.ones(2, 3, 4, requires_grad=True)
    loss = hidden_match_loss(
        student,
        teacher,
        layers=[1],
        normalize_teacher_std=False,
        layer_reduction="mean",
    )
    loss.backward()
    assert student.grad is not None and student.grad[:, 1].abs().sum() > 0
    assert teacher.grad is None
    assert student.grad[:, [0, 2]].abs().sum() == 0


def test_kv_mask_excludes_padded_teacher_slots():
    student = torch.zeros(1, 1, 1, 2, 1)
    teacher = torch.tensor([[[[[1.0], [1000.0]]]]])
    loss = kv_match_loss(
        student,
        student,
        teacher,
        teacher,
        mask=torch.tensor([[True, False]]),
    )
    assert loss.item() == pytest.approx(1.0)


def test_projected_key_loss_uses_only_selected_orthonormal_directions():
    student = torch.zeros(1, 1, 1, 2, 3)
    teacher = torch.tensor([[[[[2.0, 7.0, 11.0], [4.0, 8.0, 12.0]]]]])
    basis = torch.zeros(1, 1, 2, 3, 1)
    basis[..., 0, 0] = 1.0
    projected = projected_key_match_loss(
        student,
        teacher,
        basis,
        metric="mse",
    )
    full = key_match_loss(student, teacher, metric="mse")
    assert projected.item() == pytest.approx((4.0 + 16.0) / 2)
    assert full.item() > projected.item()


def test_trajectory_module_accepts_key_only_without_value_tensors():
    hidden = torch.zeros(1, 1, 2)
    student = torch.zeros(1, 1, 1, 1, 2)
    teacher = torch.ones_like(student)
    module = TrajectoryMatchLoss(
        hidden_weight=0,
        kv_weight=1,
        kv_target="key",
        kv_metric="mse",
    )
    output = module(
        student_hidden=hidden,
        teacher_hidden=hidden,
        student_keys=student,
        teacher_keys=teacher,
    )
    assert output.kv.item() == pytest.approx(1.0)
