from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.func import functional_call

from scripts.run_official_codi_kv_target_utility import (
    sample_group_disjoint_indices,
)
from src.mech.official_codi_target_utility import (
    OfficialCODIAnswerScorer,
    build_official_student_answer_io,
)


class TinyDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]


class TinyCausalCache(nn.Module):
    def __init__(self, hidden: int = 4, vocab: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.output = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        **_,
    ):
        hidden = (
            self.embedding(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        batch, width, feature = hidden.shape
        new_key = hidden.view(batch, 1, width, feature)
        new_value = (0.5 * hidden).view(batch, 1, width, feature)
        if past_key_values is None:
            key, value = new_key, new_value
        else:
            key = torch.cat((past_key_values[0][0], new_key), dim=2)
            value = torch.cat((past_key_values[0][1], new_value), dim=2)
        return SimpleNamespace(
            past_key_values=((key, value),),
            hidden_states=(hidden, hidden),
            logits=self.output(hidden),
        )


class TinyOfficialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.codi = TinyCausalCache()
        self.prj = nn.Linear(4, 4)
        self.eot_id = 50
        self.pad_token_id = 0

    def input_embeddings(self):
        return self.codi.embedding


def test_student_answer_teacher_forcing_starts_with_eot():
    batch = SimpleNamespace(
        teacher_ids=torch.tensor(
            [
                [10, 11, 20, 21, 22, 99],
                [12, 30, 31, 99, 0, 0],
            ]
        ),
        teacher_mask=torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
            ]
        ),
        teacher_trace_end=torch.tensor([2, 1]),
        teacher_answer_start=torch.tensor([4, 2]),
    )
    inputs, targets, mask = build_official_student_answer_io(
        batch,
        eot_token_id=50,
        pad_token_id=0,
    )
    assert inputs.tolist() == [
        [50, 20, 21, 22],
        [50, 30, 31, 0],
    ]
    assert targets.tolist() == [
        [20, 21, 22, 99],
        [30, 31, 99, 0],
    ]
    assert mask.tolist() == [
        [False, False, True, True],
        [False, True, True, False],
    ]


def test_discovery_and_validation_question_groups_are_disjoint():
    rows = [
        {
            "question": f" Question   {index // 2} ",
            "cot": "a b",
            "answer": str(index + 1),
        }
        for index in range(12)
    ]
    dataset = TinyDataset(rows)
    discovery, validation, metadata = sample_group_disjoint_indices(
        dataset,
        examples_per_split=3,
        seed=7,
    )
    discovery_questions = {
        " ".join(dataset[index]["question"].casefold().split())
        for index in discovery
    }
    validation_questions = {
        " ".join(dataset[index]["question"].casefold().split())
        for index in validation
    }
    assert discovery_questions.isdisjoint(validation_questions)
    assert metadata["eligible_unique_question_groups"] == 6


def test_official_student_scorer_is_differentiable_and_stateless_callable():
    model = TinyOfficialModel()
    scorer = OfficialCODIAnswerScorer(model, latent_positions=2)
    batch = SimpleNamespace(
        student_question_ids=torch.tensor([[3, 4], [5, 6]]),
        student_question_mask=torch.ones(2, 2, dtype=torch.long),
        teacher_ids=torch.tensor(
            [
                [3, 7, 20, 21, 99],
                [5, 8, 22, 23, 99],
            ]
        ),
        teacher_mask=torch.ones(2, 5, dtype=torch.long),
        teacher_trace_end=torch.tensor([2, 2]),
        teacher_answer_start=torch.tensor([3, 3]),
    )
    output = scorer(batch, return_kv=True)
    assert output.per_example_loss.shape == (2,)
    assert output.student_keys.shape == (2, 1, 1, 2, 4)
    assert output.student_values.shape == (2, 1, 1, 2, 4)
    gradients = torch.autograd.grad(
        output.mean_loss,
        tuple(parameter for parameter in scorer.parameters()),
        allow_unused=True,
    )
    assert any(value is not None for value in gradients)

    name, parameter = next(iter(scorer.named_parameters()))
    original = parameter.detach().clone()
    updated = {name: parameter.detach() + 0.01}
    with torch.no_grad():
        changed = functional_call(
            scorer,
            updated,
            (batch,),
            {"return_kv": False},
            strict=False,
        )
    assert changed.per_example_loss.shape == (2,)
    assert torch.equal(next(iter(scorer.named_parameters()))[1], original)
