"""Differentiable official-CODI student answer and teacher KV target paths."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.teacher_cache import cache_to_tensors
from src.losses.kv_compress import rkv_compress


@dataclass(frozen=True)
class OfficialCODIStudentAnswerOutput:
    per_example_loss: torch.Tensor
    mean_loss: torch.Tensor
    student_keys: torch.Tensor | None
    student_values: torch.Tensor | None
    # Historical diagnostic captured immediately after the final continuous latent
    # step, before EOT and the answer cue.  This is not CODI's native distillation
    # endpoint and is retained only for backwards compatibility.
    student_endpoint_hidden: torch.Tensor | None
    # Source-faithful CODI endpoint: embedding state plus every transformer-block
    # state at the colon in the student's teacher-forced ``The answer is:`` cue.
    student_answer_endpoint_hidden: torch.Tensor | None


@dataclass(frozen=True)
class OfficialCODITeacherKVTargets:
    keys: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class OfficialCODITeacherEndpointTargets:
    """Detached block-only and complete hidden tuples at the teacher answer cue."""

    hidden: torch.Tensor
    all_hidden: torch.Tensor


def build_official_student_answer_io(
    batch,
    *,
    eot_token_id: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build teacher forcing while scoring only numeric-answer tokens and EOS.

    The fixed ``The answer is:`` cue remains in the input context but is masked from
    the loss so it cannot dominate target-utility rankings.
    """
    target_rows: list[list[int]] = []
    score_rows: list[list[int]] = []
    for row in range(batch.teacher_ids.shape[0]):
        start = int(batch.teacher_trace_end[row])
        answer_start = int(batch.teacher_answer_start[row])
        end = int(batch.teacher_mask[row].sum())
        if not start < answer_start < end:
            raise ValueError("official numeric-answer boundary is invalid")
        values = [
            int(value)
            for value in batch.teacher_ids[row, start:end].detach().cpu().tolist()
        ]
        if not values:
            raise ValueError("official student answer target cannot be empty")
        target_rows.append(values)
        score_rows.append(
            [0] * (answer_start - start)
            + [1] * (end - answer_start)
        )
    width = max(len(values) for values in target_rows)
    inputs = []
    targets = []
    masks = []
    for values, score_mask in zip(target_rows, score_rows):
        amount = width - len(values)
        inputs.append(
            [int(eot_token_id), *values[:-1]]
            + [int(pad_token_id)] * amount
        )
        targets.append(values + [int(pad_token_id)] * amount)
        masks.append(score_mask + [0] * amount)
    device = batch.teacher_ids.device
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def official_codi_student_answer_forward(
    model,
    batch,
    *,
    latent_positions: int,
    return_kv: bool,
    return_endpoint_hidden: bool = False,
    return_answer_endpoint_hidden: bool = False,
) -> OfficialCODIStudentAnswerOutput:
    """Run the released six-step student path with differentiable answer NLL."""
    if latent_positions <= 0:
        raise ValueError("latent_positions must be positive")
    encoded = model.codi(
        input_ids=batch.student_question_ids,
        attention_mask=batch.student_question_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    cache = encoded.past_key_values
    latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
    endpoint_hidden = None
    for position in range(latent_positions):
        latent_output = model.codi(
            inputs_embeds=latent,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = latent_output.past_key_values
        if return_endpoint_hidden and position + 1 == latent_positions:
            # Hugging Face returns the embedding state followed by one tensor per
            # transformer block. CODI distils only the block outputs.
            endpoint_hidden = torch.stack(
                [state[:, -1, :] for state in latent_output.hidden_states[1:]],
                dim=1,
            )
        latent = model.prj(
            latent_output.hidden_states[-1][:, -1, :].unsqueeze(1)
        )

    answer_inputs, answer_targets, answer_mask = build_official_student_answer_io(
        batch,
        eot_token_id=model.eot_id,
        pad_token_id=model.pad_token_id,
    )
    embeddings = model.input_embeddings()(answer_inputs)
    decoded = model.codi(
        inputs_embeds=embeddings,
        past_key_values=cache,
        # Transformers 4.52 converts a legacy tuple cache to ``DynamicCache`` only
        # when cache output is enabled. The author-released CODI path returns legacy
        # tuples, so disabling cache output here makes GPT-2 call ``get_seq_length``
        # directly on a tuple. Keep cache output enabled exactly as in official greedy
        # generation; the returned extension is unused and logits are unchanged.
        use_cache=True,
        output_hidden_states=return_answer_endpoint_hidden,
        return_dict=True,
    )
    answer_endpoint_hidden = None
    if return_answer_endpoint_hidden:
        if not decoded.hidden_states:
            raise RuntimeError("student answer pass returned no hidden states")
        # ``answer_inputs`` begins with EOT, followed by the exact answer cue.  The
        # colon is therefore at index ``len(answer_prompt_ids)``.  Reconstruct that
        # length from the independently tokenized teacher boundaries and verify the
        # gathered input token is identical to the teacher's endpoint token.
        endpoints = (
            batch.teacher_answer_start - batch.teacher_trace_end
        ).to(device=answer_inputs.device)
        row = torch.arange(answer_inputs.shape[0], device=answer_inputs.device)
        if bool((endpoints < 1).any()) or bool((endpoints >= answer_inputs.shape[1]).any()):
            raise RuntimeError("student answer-cue endpoint is outside decoder inputs")
        student_tokens = answer_inputs[row, endpoints]
        teacher_tokens = batch.teacher_ids[
            row.to(device=batch.teacher_ids.device),
            batch.teacher_endpoint.to(device=batch.teacher_ids.device),
        ].to(device=student_tokens.device)
        if not torch.equal(student_tokens, teacher_tokens):
            raise RuntimeError("teacher and student answer-cue endpoint tokens differ")
        all_states = torch.stack(decoded.hidden_states, dim=1)
        answer_endpoint_hidden = all_states[row, :, endpoints, :]
        if not torch.isfinite(answer_endpoint_hidden).all():
            raise RuntimeError("student answer endpoint states contain non-finite values")
    token_loss = F.cross_entropy(
        decoded.logits.transpose(1, 2),
        answer_targets,
        reduction="none",
    )
    weights = answer_mask.to(dtype=token_loss.dtype)
    per_example = (token_loss * weights).sum(dim=-1) / weights.sum(
        dim=-1
    ).clamp_min(1)

    student_keys = None
    student_values = None
    if return_kv:
        keys, values = cache_to_tensors(cache)
        student_keys = keys[:, :, :, -latent_positions:, :]
        student_values = values[:, :, :, -latent_positions:, :]
    return OfficialCODIStudentAnswerOutput(
        per_example_loss=per_example,
        mean_loss=per_example.mean(),
        student_keys=student_keys,
        student_values=student_values,
        student_endpoint_hidden=endpoint_hidden,
        student_answer_endpoint_hidden=answer_endpoint_hidden,
    )


class OfficialCODIAnswerScorer(nn.Module):
    """Module wrapper that makes the answer path compatible with functional_call."""

    def __init__(self, model, *, latent_positions: int) -> None:
        super().__init__()
        self.model = model
        self.latent_positions = int(latent_positions)

    def forward(
        self,
        batch,
        *,
        return_kv: bool = False,
        return_endpoint_hidden: bool = False,
        return_answer_endpoint_hidden: bool = False,
    ):
        return official_codi_student_answer_forward(
            self.model,
            batch,
            latent_positions=self.latent_positions,
            return_kv=return_kv,
            return_endpoint_hidden=return_endpoint_hidden,
            return_answer_endpoint_hidden=return_answer_endpoint_hidden,
        )


def extract_official_teacher_endpoint_targets(
    model,
    batch,
) -> OfficialCODITeacherEndpointTargets:
    """Gather detached all-block states at the exact official answer-cue endpoint."""
    with torch.no_grad():
        outputs = model.codi(
            input_ids=batch.teacher_ids,
            attention_mask=batch.teacher_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        all_states = torch.stack(outputs.hidden_states, dim=1)
        block_states = all_states[:, 1:, :, :]
        if block_states.ndim != 4:
            raise RuntimeError("official teacher hidden states must be [B,L,T,D]")
        row = torch.arange(block_states.shape[0], device=block_states.device)
        endpoints = batch.teacher_endpoint.to(device=block_states.device)
        hidden = block_states[row, :, endpoints, :]
        all_hidden = all_states[row, :, endpoints, :]
        if not torch.isfinite(hidden).all():
            raise RuntimeError("official teacher endpoint states contain non-finite values")
        if not torch.isfinite(all_hidden).all():
            raise RuntimeError("official teacher all-state endpoints contain non-finite values")
    return OfficialCODITeacherEndpointTargets(
        hidden=hidden.detach(),
        all_hidden=all_hidden.detach(),
    )


def extract_official_teacher_kv_targets(
    model,
    batch,
    *,
    latent_positions: int,
    importance_weight: float,
) -> OfficialCODITeacherKVTargets:
    """Extract detached R-KV-selected explicit-trace targets."""
    with torch.no_grad():
        outputs = model.codi(
            input_ids=batch.teacher_ids,
            attention_mask=batch.teacher_mask,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=True,
            return_dict=True,
        )
        if not outputs.attentions:
            raise RuntimeError(
                "official teacher returned no attentions; force eager attention"
            )
        keys, values = cache_to_tensors(outputs.past_key_values)
        attentions = torch.stack(outputs.attentions, dim=1)
        batch_size, layers, heads, _, head_dim = keys.shape
        lengths = batch.teacher_trace_end - batch.teacher_trace_start
        max_trace = max(1, int(lengths.max()))
        trace_keys = keys.new_zeros(
            (batch_size, layers, heads, max_trace, head_dim)
        )
        trace_values = values.new_zeros(trace_keys.shape)
        importance = keys.new_zeros(
            (batch_size, layers, heads, max_trace)
        )
        trace_mask = torch.zeros(
            (batch_size, max_trace),
            dtype=torch.bool,
            device=keys.device,
        )
        for index in range(batch_size):
            start = int(batch.teacher_trace_start[index])
            end = int(batch.teacher_trace_end[index])
            endpoint = int(batch.teacher_endpoint[index])
            answer_start = int(batch.teacher_answer_start[index])
            sequence_end = int(batch.teacher_mask[index].sum())
            count = end - start
            if count <= 0:
                continue
            trace_keys[index, :, :, :count] = keys[index, :, :, start:end]
            trace_values[index, :, :, :count] = values[
                index, :, :, start:end
            ]
            trace_mask[index, :count] = True
            rows = attentions[
                index, :, :, answer_start:sequence_end, start:end
            ]
            if rows.shape[-2] == 0:
                rows = attentions[
                    index, :, :, endpoint : endpoint + 1, start:end
                ]
            scores = rows.mean(dim=-2)
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            importance[index, :, :, :count] = scores
        compressed = rkv_compress(
            trace_keys,
            trace_values,
            importance,
            trace_mask,
            latent_positions,
            importance_weight=importance_weight,
        )
    return OfficialCODITeacherKVTargets(
        keys=compressed.keys.detach(),
        values=compressed.values.detach(),
        mask=compressed.mask.detach(),
    )
