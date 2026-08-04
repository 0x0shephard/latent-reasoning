from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from scripts.collect_official_codi_endpoint_parameter_aware import (
    _math_sdpa_context,
    _shuffled_answer_batch,
    sample_fresh_parameter_aware_partitions,
)
from src.data.official_codi_training import OfficialCODIKVBatch
from src.mech.endpoint_answer_conditioned import (
    create_answer_alignment_moments,
    update_answer_alignment_moments,
)
from src.mech.endpoint_parameter_aware import (
    fit_parameter_aware_bases,
    parameter_gradient_cosines,
    residual_pc_candidate_losses,
    validate_parameter_aware_bases,
)


class TinyDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_math_sdpa_context_supports_double_backward_after_context_exit():
    device = torch.device("cuda")
    query = torch.randn(1, 2, 4, 8, device=device, requires_grad=True)
    key = torch.randn(1, 2, 4, 8, device=device, requires_grad=True)
    value = torch.randn(1, 2, 4, 8, device=device, requires_grad=True)
    with _math_sdpa_context(device):
        output = F.scaled_dot_product_attention(query, key, value)
    first = torch.autograd.grad(output.square().sum(), query, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), query)[0]
    assert torch.isfinite(second).all()


def test_parameter_gradient_cosines_match_exact_two_dimensional_geometry():
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    losses = torch.stack(
        [parameter[0] + 2 * parameter[1], 3 * parameter[0] - parameter[1]]
    )
    probes = [
        (torch.tensor([1.0, 1.0]),),
        (torch.tensor([1.0, -1.0]),),
    ]
    result = parameter_gradient_cosines(
        losses,
        [parameter],
        {
            "x": (torch.tensor([1.0, 0.0]),),
            "y": (torch.tensor([0.0, 1.0]),),
        },
        hutchinson_probes=2,
        seed=3,
        probe_directions=probes,
    )
    assert torch.allclose(
        result["cosines"]["x"],
        torch.tensor([1 / math.sqrt(5), 3 / math.sqrt(10)], dtype=torch.float64),
    )
    assert torch.allclose(
        result["cosines"]["y"],
        torch.tensor([2 / math.sqrt(5), -1 / math.sqrt(10)], dtype=torch.float64),
    )


def test_rank_one_candidate_losses_cover_only_registered_states_and_pcs():
    torch.manual_seed(2)
    student = torch.randn(3, 13, 8, requires_grad=True)
    teacher = torch.randn(3, 13, 8)
    eigenvectors = torch.eye(8).expand(13, 8, 8).clone()
    losses, identities = residual_pc_candidate_losses(
        student,
        teacher,
        eigenvectors,
        candidate_states=(11, 12),
        candidate_pc_count=4,
    )
    assert losses.shape == (8,)
    assert identities.tolist() == [
        [11, 0], [11, 1], [11, 2], [11, 3],
        [12, 0], [12, 1], [12, 2], [12, 3],
    ]
    losses.sum().backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[:, :11, :]) == 0


def test_parameter_aware_selection_cannot_escape_final_two_states():
    states, hidden = 13, 8
    moments = create_answer_alignment_moments(states, hidden)
    actual = torch.zeros(40, states, hidden)
    shuffled = torch.zeros_like(actual)
    actual[:, 4, 0] = 5.0
    actual[:, 11, 2] = 0.5
    actual[:, 12, 1] = 0.4
    update_answer_alignment_moments(
        moments,
        actual,
        shuffled,
        torch.arange(40) % 2,
    )
    eigenvalues = torch.arange(hidden, 0, -1).float().expand(states, hidden).clone()
    eigenvectors = torch.eye(hidden).expand(states, hidden, hidden).clone()
    bases = fit_parameter_aware_bases(
        eigenvalues,
        eigenvectors,
        moments,
        candidate_states=(11, 12),
        candidate_pc_count=4,
        hutchinson_probes=2,
        minimum_split_z=1.645,
        selection_fdr=0.05,
        maximum_rank_per_state=2,
        random_seed=7,
        residual_fit_count=64,
        direction_selection_examples=320,
    )
    validate_parameter_aware_bases(
        bases, states=states, hidden_size=hidden, require_candidate=True
    )
    assert bases.active_states == (11, 12)
    assert int(bases.selected_pc_indices[11, 0]) == 2
    assert int(bases.selected_pc_indices[12, 0]) == 1


def test_shuffled_answer_keeps_questions_and_permutes_teacher_fields():
    batch = OfficialCODIKVBatch(
        student_question_ids=torch.tensor([[1], [2], [3]]),
        student_question_mask=torch.ones(3, 1, dtype=torch.long),
        teacher_ids=torch.tensor([[10], [20], [30]]),
        teacher_mask=torch.ones(3, 1, dtype=torch.long),
        teacher_trace_start=torch.tensor([1, 2, 3]),
        teacher_trace_end=torch.tensor([4, 5, 6]),
        teacher_answer_start=torch.tensor([7, 8, 9]),
        teacher_endpoint=torch.tensor([10, 11, 12]),
    )
    permutation = torch.tensor([1, 2, 0])
    shuffled = _shuffled_answer_batch(batch, permutation)
    assert torch.equal(shuffled.student_question_ids, batch.student_question_ids)
    assert shuffled.teacher_ids.flatten().tolist() == [20, 30, 10]
    assert shuffled.teacher_answer_start.tolist() == [8, 9, 7]


def test_new_partitions_exclude_seed11_and_seed29_questions():
    dataset = TinyDataset(
        [
            {"question": f"Question {index}", "cot": "work", "answer": "1"}
            for index in range(8200)
        ]
    )
    partitions, metadata = sample_fresh_parameter_aware_partitions(
        dataset,
        residual_fit_examples=8,
        direction_selection_examples=8,
        update_examples=4,
        validation_examples=4,
        seed=41,
    )
    assert sum(len(values) for values in partitions.values()) == 24
    assert metadata["excluded_unique_questions"] == 8072
    assert metadata["legacy_exclusion"]["seed"] == 11
    assert metadata["answer_conditioned_exclusion"]["seed"] == 29
