"""CODI/KaVa trajectory-loss shape and stop-gradient tests."""
from __future__ import annotations

import pytest
import torch

from src.losses.trajectory_match import TrajectoryMatchLoss, hidden_match_loss, kv_match_loss


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
