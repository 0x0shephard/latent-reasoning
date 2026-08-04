from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.collect_official_codi_endpoint_answer_conditioned import (
    _answer_gradients_at_colon,
    sample_fresh_answer_conditioned_partitions,
)
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    sample_endpoint_tsvc_partitions,
)
from src.mech.endpoint_answer_conditioned import (
    answer_conditioned_endpoint_loss,
    create_answer_alignment_moments,
    fit_answer_conditioned_bases,
    project_variable_rank_residual,
    update_answer_alignment_moments,
    validate_answer_conditioned_bases,
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


def test_answer_gradient_collection_skips_frozen_embedding_state():
    batch_size, width, hidden = 2, 3, 768
    embedding = torch.zeros(batch_size, width, hidden)
    blocks = tuple(
        torch.randn(batch_size, width, hidden, requires_grad=True)
        for _ in range(12)
    )
    loss = sum(value.sum() for value in blocks) / batch_size
    output = SimpleNamespace(
        mean_loss=loss,
        student_answer_hidden_states=(embedding, *blocks),
    )
    batch = SimpleNamespace(
        teacher_answer_start=torch.tensor([2, 2]),
        teacher_trace_end=torch.tensor([1, 1]),
    )
    gradients = _answer_gradients_at_colon(output, batch)
    assert gradients.shape == (batch_size, 13, hidden)
    assert torch.count_nonzero(gradients[:, 0, :]) == 0
    assert torch.allclose(gradients[:, 1:, :], torch.ones_like(gradients[:, 1:, :]))


def test_split_stable_positive_direction_is_selected_and_embedding_is_excluded():
    states, hidden = 13, 8
    moments = create_answer_alignment_moments(states, hidden)
    products = torch.zeros(40, states, hidden)
    shuffled = torch.zeros_like(products)
    products[:, 0, 1] = 3.0
    products[:, 4, 3] = 2.0
    products[:, 4, 5] = torch.linspace(-1.0, 1.0, 40)
    split_ids = torch.arange(40) % 2
    update_answer_alignment_moments(
        moments, products, shuffled, split_ids
    )
    eigenvalues = torch.arange(hidden, 0, -1).float().expand(states, hidden).clone()
    eigenvectors = torch.eye(hidden).expand(states, hidden, hidden).clone()
    bases = fit_answer_conditioned_bases(
        eigenvalues,
        eigenvectors,
        moments,
        minimum_split_z=1.645,
        selection_fdr=0.05,
        maximum_rank_per_state=4,
        random_seed=9,
        residual_fit_count=64,
    )
    validate_answer_conditioned_bases(
        bases, states=states, hidden_size=hidden
    )
    assert int(bases.ranks[0]) == 0
    assert int(bases.ranks[4]) == 1
    assert int(bases.selected_pc_indices[4, 0]) == 3


def test_variable_rank_projection_and_complement_reconstruct_blocks():
    torch.manual_seed(4)
    residual = torch.randn(3, 13, 8)
    ranks = torch.zeros(13, dtype=torch.int64)
    ranks[2] = 2
    ranks[12] = 3
    basis = torch.zeros(13, 8, 3)
    basis[2, :, :2] = torch.eye(8)[:, :2]
    basis[12, :, :3] = torch.eye(8)[:, :3]
    projected = project_variable_rank_residual(residual, basis, ranks)
    assert torch.allclose(projected[:, 1], torch.zeros_like(projected[:, 1]))
    assert torch.allclose(projected[:, 2, :2], residual[:, 2, :2])
    assert torch.allclose(projected + (residual - projected), residual)


def test_block_only_loss_ignores_embedding_state():
    torch.manual_seed(5)
    student = torch.randn(4, 13, 8, requires_grad=True)
    teacher = torch.randn(4, 13, 8)
    reference = answer_conditioned_endpoint_loss(
        student, teacher, mode="full_blocks"
    )
    changed = student.detach().clone()
    changed[:, 0, :] += 1000
    observed = answer_conditioned_endpoint_loss(
        changed, teacher, mode="full_blocks"
    )
    assert torch.allclose(reference.detach(), observed, atol=1e-6)


def test_new_partitions_exclude_completed_seed11_questions():
    dataset = TinyDataset(
        [
            {"question": f"Question {index}", "cot": "work", "answer": "1"}
            for index in range(5600)
        ]
    )
    legacy, _ = sample_endpoint_tsvc_partitions(
        dataset,
        calibration_examples=5000,
        update_examples=256,
        validation_examples=256,
        seed=11,
    )
    fresh, metadata = sample_fresh_answer_conditioned_partitions(
        dataset,
        residual_fit_examples=8,
        direction_selection_examples=8,
        update_examples=4,
        validation_examples=4,
        seed=29,
    )
    old_questions = {
        _normalized_question(dataset[index]["question"])
        for name in legacy
        for index in legacy[name]
    }
    new_questions = {
        _normalized_question(dataset[index]["question"])
        for name in fresh
        for index in fresh[name]
    }
    assert old_questions.isdisjoint(new_questions)
    assert metadata["legacy_exclusion"]["excluded_unique_questions"] == 5512
