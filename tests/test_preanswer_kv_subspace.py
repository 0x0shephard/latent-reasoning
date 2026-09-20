from types import SimpleNamespace

import torch
import torch.nn as nn

from src.mech.direct_layerwise_kv import LayerwiseEigensystem
from src.mech.preanswer_kv_subspace import (
    _bh_qvalues,
    _cache_gradients,
    cache_as_legacy_tuple,
    latent_cache_tensor,
    official_codi_preanswer_kv_forward,
    select_variable_layerwise_bases,
)


class ContextUsingCacheLM(nn.Module):
    def __init__(self, hidden=4, vocabulary=128):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.output = nn.Linear(hidden, vocabulary, bias=False)

    def forward(self, *, input_ids=None, inputs_embeds=None, past_key_values=None, **_):
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if past_key_values is not None:
            context = (
                past_key_values[0][0].mean(2) + past_key_values[0][1].mean(2)
            ).reshape(hidden.shape[0], 1, -1)
            hidden = hidden + context
        batch, width, feature = hidden.shape
        new_key = hidden.reshape(batch, 1, width, feature)
        new_value = (0.5 * hidden).reshape(batch, 1, width, feature)
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


class TinyContextCODI(nn.Module):
    def __init__(self):
        super().__init__()
        self.codi = ContextUsingCacheLM()
        self.prj = nn.Linear(4, 4)
        self.eot_id = 50
        self.pad_token_id = 0

    def input_embeddings(self):
        return self.codi.embedding


def test_exact_cache_gradients_are_connected_and_keep_last_latent_positions():
    keys = [torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3)]
    values = [torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3)]
    cache = tuple(zip(keys, values))
    loss = sum((key[:, :, -2:] ** 2).mean() + (value[:, :, -2:] ** 2).mean()
               for key, value in cache)
    key_gradient, value_gradient, connected, full_key, full_value = _cache_gradients(
        loss, cache, latent_positions=2, batch_scale=1
    )
    assert key_gradient.shape == (2, 3, 2, 8)
    assert value_gradient.shape == (2, 3, 2, 8)
    assert connected.shape == (3, 2)
    assert full_key.shape == (2, 3, 5, 8)
    assert full_value.shape == (2, 3, 5, 8)
    assert bool(connected.all())
    assert float(key_gradient.abs().sum()) > 0
    assert float(value_gradient.abs().sum()) > 0


def test_preanswer_forward_differentiates_the_same_cache_used_by_answer_decoder():
    model = TinyContextCODI()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batch = SimpleNamespace(
        student_question_ids=torch.tensor([[3, 4], [5, 6]]),
        student_question_mask=torch.ones(2, 2, dtype=torch.long),
        teacher_ids=torch.tensor([[3, 7, 20, 21, 99], [5, 8, 22, 23, 99]]),
        teacher_mask=torch.ones(2, 5, dtype=torch.long),
        teacher_trace_end=torch.tensor([2, 2]),
        teacher_answer_start=torch.tensor([3, 3]),
    )
    output = official_codi_preanswer_kv_forward(
        model, batch, latent_positions=2, return_gradients=True
    )
    assert output.per_example_loss.shape == (2,)
    assert output.key_states.shape == (2, 1, 2, 4)
    assert output.value_states.shape == (2, 1, 2, 4)
    assert output.key_gradients.shape == (2, 1, 2, 4)
    assert output.value_gradients.shape == (2, 1, 2, 4)
    assert bool(output.gradient_connected.all())
    assert float(output.key_gradients.abs().sum()) > 0
    assert float(output.value_gradients.abs().sum()) > 0


def test_preanswer_forward_supports_label_free_top1_margin_gradients():
    model = TinyContextCODI()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batch = SimpleNamespace(
        student_question_ids=torch.tensor([[3, 4], [5, 6]]),
        student_question_mask=torch.ones(2, 2, dtype=torch.long),
        teacher_ids=torch.tensor([[3, 7, 20, 21, 99], [5, 8, 22, 23, 99]]),
        teacher_mask=torch.ones(2, 5, dtype=torch.long),
        teacher_trace_end=torch.tensor([2, 2]),
        teacher_answer_start=torch.tensor([3, 3]),
    )
    output = official_codi_preanswer_kv_forward(
        model,
        batch,
        latent_positions=2,
        return_gradients=True,
        gradient_objective="first_token_margin",
    )
    assert bool(output.gradient_connected.all())
    assert float(output.key_gradients.abs().sum()) > 0
    assert float(output.value_gradients.abs().sum()) > 0


def test_preanswer_forward_can_return_full_cache_gradients_for_allocation():
    model = TinyContextCODI()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batch = SimpleNamespace(
        student_question_ids=torch.tensor([[3, 4], [5, 6]]),
        student_question_mask=torch.ones(2, 2, dtype=torch.long),
        teacher_ids=torch.tensor([[3, 7, 20, 21, 99], [5, 8, 22, 23, 99]]),
        teacher_mask=torch.ones(2, 5, dtype=torch.long),
        teacher_trace_end=torch.tensor([2, 2]),
        teacher_answer_start=torch.tensor([3, 3]),
    )
    output = official_codi_preanswer_kv_forward(
        model,
        batch,
        latent_positions=2,
        return_gradients=True,
        return_full_cache_gradients=True,
    )
    assert output.full_key_gradients.shape == (2, 1, 4, 4)
    assert output.full_value_gradients.shape == (2, 1, 4, 4)
    assert torch.allclose(output.key_gradients, output.full_key_gradients[:, :, -2:])
    assert torch.allclose(output.value_gradients, output.full_value_gradients[:, :, -2:])


def test_cache_conversion_preserves_tensor_identity():
    key = torch.randn(1, 2, 3, 4, requires_grad=True)
    value = torch.randn(1, 2, 3, 4, requires_grad=True)
    legacy = cache_as_legacy_tuple([(key, value)])
    assert legacy[0][0] is key
    assert legacy[0][1] is value
    stacked_key, stacked_value = latent_cache_tensor(legacy, latent_positions=2)
    assert stacked_key.shape == (1, 1, 2, 8)
    assert stacked_value.shape == (1, 1, 2, 8)


def test_bh_qvalues_are_monotone_in_sorted_p_values():
    p_values = torch.tensor([[0.001, 0.02, 0.2, 0.8]])
    q_values = _bh_qvalues(p_values)
    assert q_values.shape == p_values.shape
    assert bool((q_values >= p_values).all())
    assert bool((q_values[:, 1:] >= q_values[:, :-1]).all())


def test_variable_selection_can_return_different_ranks_and_zero():
    identity = torch.eye(768)
    eigen = LayerwiseEigensystem(
        means=torch.zeros(12, 2, 768),
        eigenvalues=torch.ones(12, 768),
        eigenvectors=identity.repeat(12, 1, 1),
    )
    z = torch.zeros(12, 768)
    effect = torch.zeros(12, 768)
    positive = torch.zeros(12, 768, dtype=torch.bool)
    q = torch.ones(12, 768)
    z[1, :2] = 5; effect[1, :2] = torch.tensor([0.9, 0.1]); positive[1, :2] = True; q[1, :2] = 0.001
    z[2, :3] = 5; effect[2, :3] = 1; positive[2, :3] = True; q[2, :3] = 0.001
    bases, indices, ranks = select_variable_layerwise_bases(
        eigen,
        {"split_stable_z": z, "excess_predicted_removal_damage": effect,
         "positive_both_splits": positive, "q_values": q},
        maximum_rank=64, minimum_split_z=1.645, fdr_q=0.05,
        retained_effect_fraction=0.95,
    )
    assert ranks[0] == 0
    assert ranks[1] == 2
    assert ranks[2] == 3
    assert bases[2].shape == (768, 3)
    assert indices[0].numel() == 0


def test_variable_selection_uses_disjoint_validation_scores_for_rank():
    identity = torch.eye(768)
    eigen = LayerwiseEigensystem(
        means=torch.zeros(12, 2, 768), eigenvalues=torch.ones(12, 768),
        eigenvectors=identity.repeat(12, 1, 1),
    )
    z = torch.zeros(12, 768); effect = torch.zeros(12, 768)
    positive = torch.zeros(12, 768, dtype=torch.bool); q = torch.ones(12, 768)
    z[0, :3] = 5; effect[0, :3] = torch.tensor([3.0, 2.0, 1.0])
    positive[0, :3] = True; q[0, :3] = 0.001
    validation_effect = effect.clone(); validation_effect[0, :3] = torch.tensor([0.9, -0.2, 0.1])
    validation_positive = positive.clone(); validation_positive[0, 1] = False
    common = {"split_stable_z": z, "positive_both_splits": positive,
              "q_values": q, "excess_predicted_removal_damage": effect}
    validation = {**common, "positive_both_splits": validation_positive,
                  "excess_predicted_removal_damage": validation_effect}
    _, indices, ranks = select_variable_layerwise_bases(
        eigen, common, maximum_rank=64, minimum_split_z=1.645, fdr_q=0.05,
        retained_effect_fraction=0.95, validation_scores=validation,
    )
    assert ranks[0] == 2
    assert indices[0].tolist() == [0, 2]
