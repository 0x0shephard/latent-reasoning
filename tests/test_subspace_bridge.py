from __future__ import annotations

import torch
import torch.nn as nn

from src.mech.subspace_bridge import (
    SubspaceInterventionVocabularyHead,
    edit_hidden_subspace,
    orthonormal_row_space,
    subspace_overlap,
)


def test_row_space_is_invariant_to_bottleneck_reparameterization():
    generator = torch.Generator().manual_seed(7)
    down = torch.randn(5, 13, generator=generator)
    change = torch.randn(5, 5, generator=generator) + 3.0 * torch.eye(5)
    first = orthonormal_row_space(down)
    second = orthonormal_row_space(change @ down)
    report = subspace_overlap(first, second)
    assert report.reference_rank == 5
    assert report.reference_capture_fraction > 1.0 - 1e-5


def test_overlap_distinguishes_capture_from_candidate_occupancy():
    identity = torch.eye(8)
    reference = identity[:, :2]
    candidate = identity[:, :5]
    report = subspace_overlap(reference, candidate)
    assert abs(report.reference_capture_fraction - 1.0) < 1e-12
    assert abs(report.candidate_occupancy_fraction - 0.4) < 1e-12
    assert abs(report.mean_squared_cosine - 1.0) < 1e-12


def test_retain_and_remove_reconstruct_the_original_hidden_state():
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(4, 9, generator=generator)
    centre = torch.randn(9, generator=generator)
    basis = torch.linalg.qr(torch.randn(9, 3, generator=generator)).Q
    retained = edit_hidden_subspace(hidden, basis, centre, mode="retain")
    removed = edit_hidden_subspace(hidden, basis, centre, mode="remove")
    assert torch.allclose(retained + removed - centre, hidden, atol=1e-6)


class _PositionAwareLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 6, bias=False)
        self.position = None

    def set_answer_position(self, position):
        self.position = position

    def forward(self, hidden):
        return self.linear(hidden)


def test_intervention_head_edits_only_the_requested_answer_position():
    inner = _PositionAwareLinear()
    basis = torch.eye(4)[:, :1]
    centre = torch.zeros(4)
    wrapper = SubspaceInterventionVocabularyHead(
        inner, basis, centre, mode="retain", vocabulary_size=6,
        active_positions={0},
    )
    hidden = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    wrapper.set_answer_position(None)
    assert inner.position is None
    assert torch.allclose(wrapper(hidden), inner(hidden))
    wrapper.set_answer_position(0)
    assert inner.position == 0
    expected = inner(torch.tensor([[2.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(wrapper(hidden), expected)
    wrapper.set_answer_position(1)
    assert torch.allclose(wrapper(hidden), inner(hidden))
