"""Tiny GPT-2 shape/backprop checks for the shared Phase-2 latent model."""
from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.data.teacher_cache import LatentBatch
from src.models.latent_lm import LatentCausalLM


def _batch() -> LatentBatch:
    return LatentBatch(
        question_ids=torch.tensor([[4, 5, 30, 0], [6, 7, 8, 30]]),
        question_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        student_segment_ids=torch.tensor([[31, 9, 10, 11], [31, 9, 12, 0]]),
        student_segment_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        student_labels=torch.tensor([[-100, -100, 10, 11], [-100, -100, 12, -100]]),
        student_endpoint=torch.tensor([1, 1]),
        teacher_ids=torch.tensor([[4, 5, 13, 9, 10, 11], [6, 7, 8, 13, 9, 12]]),
        teacher_mask=torch.ones(2, 6, dtype=torch.long),
        teacher_labels=torch.tensor(
            [[-100, -100, 13, 9, 10, 11], [-100, -100, -100, 13, 9, 12]]
        ),
        teacher_endpoint=torch.tensor([3, 4]),
        teacher_trace_start=torch.tensor([2, 3]),
        teacher_trace_end=torch.tensor([3, 4]),
        teacher_answer_start=torch.tensor([4, 5]),
        reasoning_truncated=0,
    )


def _model(mechanism="autoregressive"):
    config = GPT2Config(
        vocab_size=32,
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        attn_implementation="eager",
    )
    backbone = GPT2LMHeadModel(config)
    backbone.set_attn_implementation("eager")
    return LatentCausalLM(
        backbone,
        bot_token_id=30,
        eot_token_id=31,
        latent_steps=3,
        mechanism=mechanism,
        jacobi_iterations=2,
    )


def test_autoregressive_latent_forward_shapes_and_projection_gradient():
    model = _model()
    output = model.forward_student(_batch())
    assert output.hidden_endpoint.shape == (2, 2, 16)
    assert output.latent_hidden.shape == (2, 2, 3, 16)
    assert output.latent_keys.shape == output.latent_values.shape == (2, 2, 2, 3, 8)
    output.answer_loss.backward()
    assert model.projection[1].weight.grad is not None


def test_jacobi_ablation_uses_the_same_output_contract():
    model = _model("jacobi")
    output = model.forward_student(_batch())
    assert torch.isfinite(output.answer_loss)
    assert output.latent_keys.shape[-2] == 3


def test_latent_forward_is_seed_deterministic():
    torch.manual_seed(17)
    first = _model()
    torch.manual_seed(99)
    first_loss = first.forward_student(_batch()).answer_loss.detach()
    torch.manual_seed(17)
    second = _model()
    torch.manual_seed(99)
    second_loss = second.forward_student(_batch()).answer_loss.detach()
    assert torch.equal(first_loss, second_loss)


def test_latent_student_can_overfit_one_tiny_batch():
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=32,
        n_positions=64,
        n_ctx=64,
        n_embd=16,
        n_layer=1,
        n_head=1,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = LatentCausalLM(
        GPT2LMHeadModel(config),
        bot_token_id=30,
        eot_token_id=31,
        latent_steps=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = model.forward_student(_batch()).answer_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.4
