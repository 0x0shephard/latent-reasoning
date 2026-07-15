"""Phase-2 sequence and explicit-teacher target tests (no downloads)."""
from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.data.prompts import PromptStyle
from src.data.teacher_cache import (
    collate_latent_rows,
    drop_last_reasoning_step,
    encode_latent_example,
    extract_teacher_targets,
)


class CharTokenizer:
    eos_token_id = 0
    pad_token_id = 255

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


def test_drop_last_equation_and_natural_language_steps():
    assert drop_last_reasoning_step("<<2+2=4>> <<4*3=12>>", "eq_only") == "<<2+2=4>>"
    assert (
        drop_last_reasoning_step("First compute 2+2. Then multiply by 3.", "natural_language")
        == "First compute 2+2."
    )


def test_latent_encoding_records_exact_endpoints_and_masks_answer_only():
    tok = CharTokenizer()
    row = {"question": "2+2?", "cot": "<<2+2=4>> <<4*1=4>>", "answer": "4"}
    encoded = encode_latent_example(
        tok,
        row,
        PromptStyle(),
        bot_token_id=253,
        eot_token_id=254,
        trace_style="eq_only",
        max_length=128,
        latent_steps=3,
    )
    assert encoded.question_ids[-1] == 253
    assert encoded.student_segment_ids[0] == 254
    assert encoded.student_segment_ids[encoded.student_endpoint] == ord(":")
    assert encoded.teacher_ids[encoded.teacher_endpoint] == ord(":")
    assert encoded.teacher_trace_end > encoded.teacher_trace_start
    assert all(label == -100 for label in encoded.student_labels[: encoded.student_endpoint + 1])
    assert encoded.student_labels[-1] == tok.eos_token_id
    assert encoded.teacher_labels[-1] == tok.eos_token_id


def test_latent_collation_right_pads_without_moving_boundaries():
    tok = CharTokenizer()
    rows = [
        {"question": "Q?", "cot": "<<1+1=2>> <<2+2=4>>", "answer": "4"},
        {"question": "A longer Q?", "cot": "<<3+3=6>> <<6/2=3>>", "answer": "3"},
    ]
    batch = collate_latent_rows(
        tok,
        rows,
        PromptStyle(),
        bot_token_id=253,
        eot_token_id=254,
        trace_style="eq_only",
        max_length=128,
        latent_steps=3,
    )
    assert batch.question_ids.shape == batch.question_mask.shape
    assert batch.teacher_ids.shape == batch.teacher_mask.shape == batch.teacher_labels.shape
    assert (batch.student_labels[batch.student_segment_mask == 0] == -100).all()
    assert (batch.teacher_ids[range(2), batch.teacher_endpoint] == ord(":" )).all()


def test_real_causal_lm_output_extracts_detached_teacher_targets():
    tok = CharTokenizer()
    batch = collate_latent_rows(
        tok,
        [{"question": "Q?", "cot": "<<1+1=2>> <<2+2=4>>", "answer": "4"}],
        PromptStyle(),
        bot_token_id=253,
        eot_token_id=254,
        trace_style="eq_only",
        max_length=128,
        latent_steps=3,
    )
    config = GPT2Config(
        vocab_size=256,
        n_positions=128,
        n_ctx=128,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config)
    model.set_attn_implementation("eager")
    outputs = model(
        input_ids=batch.teacher_ids,
        attention_mask=batch.teacher_mask,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    targets = extract_teacher_targets(outputs, batch)
    assert targets.hidden_endpoint.shape == (1, 2, 16)
    assert targets.trace_keys.shape[:3] == (1, 2, 2)
    assert torch.allclose(
        targets.importance.sum(dim=-1),
        torch.ones_like(targets.importance.sum(dim=-1)),
    )
    assert not targets.hidden_endpoint.requires_grad
