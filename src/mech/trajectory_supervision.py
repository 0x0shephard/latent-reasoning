"""Trajectory-level supervision for warm-started official CODI (ledger §84).

CODI supervises only the answer-cue endpoint.  KaVa adds key/value targets at
R-KV-selected teacher positions for every latent slot.  Ledger §55 showed what the
latent trajectory actually holds: the odd slots (0-based 1, 3, 5) store the
solution's intermediate values, unordered.  This module implements the
mechanism-selected alternative, teacher positions where the explicit trace emits
an intermediate value assigned to the value-holding slots, together with the
selector and slot controls and a generative-objective comparison.

Everything here operates on batches produced by
``collate_official_codi_kv_rows`` and on the released model wrapper.  Data
sampling, optimisation, and evaluation live in the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
from typing import Sequence

import torch
import torch.nn.functional as F

from src.data.teacher_cache import cache_to_tensors
from src.losses.kv_compress import rkv_compress
from src.losses.trajectory_match import kv_match_loss
from src.mech.endpoint_tsvc import match_gradient_norm
from src.mech.kv_target_utility import autograd_gradients, combine_gradients
from src.models.official_codi import official_codi_base_model


ODD_SLOTS = (1, 3, 5)
EVEN_SLOTS = (0, 2, 4)
ALL_SLOTS = (0, 1, 2, 3, 4, 5)
ALL_STATES = tuple(range(13))
VALUE_PATTERN = re.compile(r"<<[^<>]*?=([^<>]*?)>>")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    auxiliary: str | None  # None, "kv", or "recon"
    selector: str | None  # None, "rkv", "value", or "random"
    slots: tuple[int, ...]


ARMS: dict[str, ArmSpec] = {
    "codi": ArmSpec("codi", None, None, ()),
    "kava": ArmSpec("kava", "kv", "rkv", ALL_SLOTS),
    "value_odd": ArmSpec("value_odd", "kv", "value", ODD_SLOTS),
    "random_odd": ArmSpec("random_odd", "kv", "random", ODD_SLOTS),
    "value_even": ArmSpec("value_even", "kv", "value", EVEN_SLOTS),
    "recon_odd": ArmSpec("recon_odd", "recon", "value", ODD_SLOTS),
}
ARM_ORDER = tuple(ARMS)


# --------------------------------------------------------------------------- data


def equation_value_spans(cot: str) -> list[tuple[int, int]]:
    """Character spans of the result inside every ``<<... = result>>`` equation."""
    return [match.span(1) for match in VALUE_PATTERN.finditer(cot)]


def _decode(tokenizer, ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(ids), clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(list(ids))


def value_token_indices(tokenizer, cot_text: str, cot_ids: Sequence[int]) -> list[int] | None:
    """Index within ``cot_ids`` of the first token of every equation result.

    Alignment uses incremental decoding so it holds for any tokenizer whose
    decode inverts encode on this text.  Returns ``None`` when the token sequence
    does not reproduce ``cot_text`` exactly, so the caller can exclude the row
    rather than supervise a misaligned position.
    """
    ids = [int(value) for value in cot_ids]
    if not ids:
        return []
    if _decode(tokenizer, ids) != cot_text:
        return None
    ends = []
    for index in range(len(ids)):
        prefix = _decode(tokenizer, ids[: index + 1])
        if not cot_text.startswith(prefix):
            return None
        ends.append(len(prefix))
    indices = []
    for start, stop in equation_value_spans(cot_text):
        if stop <= start:
            continue
        token = next((index for index, end in enumerate(ends) if end > start), None)
        if token is None:
            return None
        indices.append(token)
    return indices


# ------------------------------------------------------------------ teacher trace


@dataclass(frozen=True)
class TeacherTrace:
    """Detached teacher trace K/V ``[B,L,H,N,D]``, mask ``[B,N]``, importance ``[B,L,H,N]``."""

    keys: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    importance: torch.Tensor


def extract_teacher_trace(model, batch) -> TeacherTrace:
    """Teacher forward over question + truncated trace + answer; keep trace K/V.

    Importance is the mean attention from the answer tokens onto each trace
    position, exactly as the completed KV-target extraction computed it.
    """
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
            raise RuntimeError("teacher returned no attentions; force eager attention")
        keys, values = cache_to_tensors(outputs.past_key_values)
        attentions = torch.stack(outputs.attentions, dim=1)
        batch_size, layers, heads, _, head_dim = keys.shape
        lengths = batch.teacher_trace_end - batch.teacher_trace_start
        width = max(1, int(lengths.max()))
        trace_keys = keys.new_zeros((batch_size, layers, heads, width, head_dim))
        trace_values = values.new_zeros(trace_keys.shape)
        importance = keys.new_zeros((batch_size, layers, heads, width))
        mask = torch.zeros((batch_size, width), dtype=torch.bool, device=keys.device)
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
            trace_values[index, :, :, :count] = values[index, :, :, start:end]
            mask[index, :count] = True
            rows = attentions[index, :, :, answer_start:sequence_end, start:end]
            if rows.shape[-2] == 0:
                rows = attentions[index, :, :, endpoint : endpoint + 1, start:end]
            scores = rows.mean(dim=-2)
            importance[index, :, :, :count] = scores / scores.sum(-1, keepdim=True).clamp_min(1e-8)
    return TeacherTrace(
        keys=trace_keys.detach(), values=trace_values.detach(),
        mask=mask.detach(), importance=importance.detach(),
    )


def teacher_endpoint_states(model, batch) -> torch.Tensor:
    """Detached ``[B,13,D]`` teacher states at the answer-cue endpoint."""
    with torch.no_grad():
        outputs = model.codi(
            input_ids=batch.teacher_ids, attention_mask=batch.teacher_mask,
            use_cache=False, output_hidden_states=True, return_dict=True,
        )
        states = torch.stack(outputs.hidden_states, dim=1)
        row = torch.arange(states.shape[0], device=states.device)
        endpoint = states[row, :, batch.teacher_endpoint.to(states.device), :]
    return endpoint.detach()


# ------------------------------------------------------------------ student path


@dataclass(frozen=True)
class StudentTrajectory:
    per_example_loss: torch.Tensor
    mean_loss: torch.Tensor
    latent_keys: torch.Tensor  # [B,L,H,M,D]
    latent_values: torch.Tensor
    latent_states: torch.Tensor  # [B,M,D] post-ln_f state of each latent slot
    answer_endpoint_hidden: torch.Tensor  # [B,13,D]


def _student_answer_io(batch, *, eot_token_id: int, pad_token_id: int):
    target_rows, score_rows = [], []
    for row in range(batch.teacher_ids.shape[0]):
        start = int(batch.teacher_trace_end[row])
        answer_start = int(batch.teacher_answer_start[row])
        end = int(batch.teacher_mask[row].sum())
        if not start < answer_start < end:
            raise ValueError("official numeric-answer boundary is invalid")
        values = [int(v) for v in batch.teacher_ids[row, start:end].detach().cpu().tolist()]
        target_rows.append(values)
        score_rows.append([0] * (answer_start - start) + [1] * (end - answer_start))
    width = max(len(values) for values in target_rows)
    inputs, targets, masks = [], [], []
    for values, score in zip(target_rows, score_rows):
        pad = width - len(values)
        inputs.append([int(eot_token_id), *values[:-1]] + [int(pad_token_id)] * pad)
        targets.append(values + [int(pad_token_id)] * pad)
        masks.append(score + [0] * pad)
    device = batch.teacher_ids.device
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def student_trajectory_forward(model, batch, *, latent_positions: int) -> StudentTrajectory:
    """Released six-step student path returning slot K/V, slot states, and the endpoint."""
    if latent_positions <= 0:
        raise ValueError("latent_positions must be positive")
    encoded = model.codi(
        input_ids=batch.student_question_ids, attention_mask=batch.student_question_mask,
        use_cache=True, output_hidden_states=True, return_dict=True,
    )
    cache = encoded.past_key_values
    latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
    slot_states = []
    for _ in range(latent_positions):
        out = model.codi(
            inputs_embeds=latent, past_key_values=cache, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
        cache = out.past_key_values
        state = out.hidden_states[-1][:, -1, :]
        slot_states.append(state)
        latent = model.prj(state.unsqueeze(1))
    answer_inputs, answer_targets, answer_mask = _student_answer_io(
        batch, eot_token_id=model.eot_id, pad_token_id=model.pad_token_id
    )
    decoded = model.codi(
        inputs_embeds=model.input_embeddings()(answer_inputs), past_key_values=cache,
        use_cache=True, output_hidden_states=True, return_dict=True,
    )
    endpoints = (batch.teacher_answer_start - batch.teacher_trace_end).to(answer_inputs.device)
    row = torch.arange(answer_inputs.shape[0], device=answer_inputs.device)
    if bool((endpoints < 1).any()) or bool((endpoints >= answer_inputs.shape[1]).any()):
        raise RuntimeError("student answer-cue endpoint is outside decoder inputs")
    student_tokens = answer_inputs[row, endpoints]
    teacher_tokens = batch.teacher_ids[row, batch.teacher_endpoint.to(answer_inputs.device)]
    if not torch.equal(student_tokens, teacher_tokens.to(student_tokens.device)):
        raise RuntimeError("teacher and student answer-cue endpoint tokens differ")
    all_states = torch.stack(decoded.hidden_states, dim=1)
    endpoint_hidden = all_states[row, :, endpoints, :]
    token_loss = F.cross_entropy(decoded.logits.transpose(1, 2), answer_targets, reduction="none")
    weights = answer_mask.to(token_loss.dtype)
    per_example = (token_loss * weights).sum(-1) / weights.sum(-1).clamp_min(1)
    keys, values = cache_to_tensors(cache)
    return StudentTrajectory(
        per_example_loss=per_example,
        mean_loss=per_example.mean(),
        latent_keys=keys[:, :, :, -latent_positions:, :],
        latent_values=values[:, :, :, -latent_positions:, :],
        latent_states=torch.stack(slot_states, dim=1),
        answer_endpoint_hidden=endpoint_hidden,
    )


def endpoint_hidden_loss(
    student: torch.Tensor, teacher: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Official endpoint distillation on ``[B, states, D]`` tensors.

    Per state: smooth-L1 (beta 1) of the residual against zero, divided by the
    teacher's unbiased standard deviation, then averaged over states.  This is the
    formula the completed retention runs used (``endpoint_retention_loss`` with
    ``mode="full"``), written without that function's fixed GPT-2 shape check so
    the same code runs on the tiny test model.
    """
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("endpoint states must share a [B, states, D] shape")
    target = teacher.detach()
    residual = student - target
    losses = []
    for state in range(student.shape[1]):
        value = F.smooth_l1_loss(
            residual[:, state, :], torch.zeros_like(residual[:, state, :]),
            reduction="mean", beta=1.0,
        )
        scale = target[:, state, :].float().std(unbiased=True).clamp_min(eps)
        losses.append(value / scale.to(value.dtype))
    return torch.stack(losses).mean()


# ------------------------------------------------------------- slot assignment


def assign_slots(cost: torch.Tensor) -> list[int]:
    """Minimum-cost injective assignment of slots (rows) to candidates (columns).

    Returns one candidate index per slot, or ``-1`` for an unassigned slot.  When
    there are fewer candidates than slots, the cheapest slot subset is chosen.
    Sizes are tiny (at most six slots, a handful of candidates), so exhaustive
    search is exact and cheap.
    """
    if cost.ndim != 2:
        raise ValueError("cost must be [slots, candidates]")
    slots, candidates = cost.shape
    if candidates == 0 or slots == 0:
        return [-1] * slots
    matrix = cost.detach().double().cpu()
    k = min(slots, candidates)
    best_total, best = None, None
    for slot_subset in itertools.combinations(range(slots), k):
        for candidate_perm in itertools.permutations(range(candidates), k):
            total = sum(float(matrix[s, c]) for s, c in zip(slot_subset, candidate_perm))
            if best_total is None or total < best_total - 1e-12:
                best_total, best = total, (slot_subset, candidate_perm)
    assignment = [-1] * slots
    for s, c in zip(*best):
        assignment[s] = int(c)
    return assignment


def kv_slot_cost(
    student_keys: torch.Tensor, student_values: torch.Tensor,
    trace: TeacherTrace, *, slots: Sequence[int], candidates: Sequence[int], row: int,
) -> torch.Tensor:
    """Detached mean-L1 cost ``[len(slots), len(candidates)]`` for one example."""
    if not candidates:
        return torch.zeros(len(slots), 0)
    sk = student_keys[row][:, :, list(slots), :].detach().float()  # [L,H,S,D]
    sv = student_values[row][:, :, list(slots), :].detach().float()
    tk = trace.keys[row][:, :, list(candidates), :].float()  # [L,H,C,D]
    tv = trace.values[row][:, :, list(candidates), :].float()
    key_cost = (sk.unsqueeze(3) - tk.unsqueeze(2)).abs().mean((0, 1, 4))
    value_cost = (sv.unsqueeze(3) - tv.unsqueeze(2)).abs().mean((0, 1, 4))
    return 0.5 * (key_cost + value_cost)


def build_slot_targets(
    student_keys: torch.Tensor, student_values: torch.Tensor, trace: TeacherTrace,
    *, slots: Sequence[int], candidates_by_row: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    """Assign candidates to slots per example; return aligned targets and a slot mask.

    Outputs are ``teacher_keys``/``teacher_values`` ``[B,L,H,M,D]`` aligned with the
    student slot axis, a boolean ``[B,M]`` mask of supervised slots, and the
    per-row assignment (candidate trace index per slot, ``-1`` if none).
    """
    batch, layers, heads, positions, head_dim = student_keys.shape
    teacher_keys = student_keys.new_zeros(student_keys.shape)
    teacher_values = student_values.new_zeros(student_values.shape)
    mask = torch.zeros(batch, positions, dtype=torch.bool, device=student_keys.device)
    assignments = []
    for row in range(batch):
        candidates = [int(c) for c in candidates_by_row[row] if bool(trace.mask[row, int(c)])]
        cost = kv_slot_cost(
            student_keys, student_values, trace, slots=slots, candidates=candidates, row=row
        )
        assignment = assign_slots(cost)
        record = [-1] * positions
        for slot, choice in zip(slots, assignment):
            if choice < 0:
                continue
            position = candidates[choice]
            teacher_keys[row, :, :, slot] = trace.keys[row, :, :, position].to(teacher_keys.dtype)
            teacher_values[row, :, :, slot] = trace.values[row, :, :, position].to(teacher_values.dtype)
            mask[row, slot] = True
            record[slot] = position
        assignments.append(record)
    return teacher_keys.detach(), teacher_values.detach(), mask, assignments


def random_trace_positions(
    trace_mask: torch.Tensor, counts: Sequence[int], *, generator: torch.Generator
) -> list[list[int]]:
    """Per example, ``counts[row]`` distinct valid trace positions, seeded."""
    result = []
    for row, count in enumerate(counts):
        valid = torch.nonzero(trace_mask[row].cpu(), as_tuple=False).flatten()
        take = min(int(count), int(valid.numel()))
        if take <= 0:
            result.append([])
            continue
        order = torch.randperm(int(valid.numel()), generator=generator)[:take]
        result.append(sorted(int(valid[i]) for i in order))
    return result


def rkv_slot_targets(trace: TeacherTrace, *, slots: int, importance_weight: float):
    """KaVa targets: R-KV per layer/head, chronological, aligned to all slots."""
    compressed = rkv_compress(
        trace.keys, trace.values, trace.importance, trace.mask, slots,
        importance_weight=importance_weight,
    )
    return compressed.keys.detach(), compressed.values.detach(), compressed.mask, compressed.indices


def rkv_value_overlap(indices: torch.Tensor, selected_mask: torch.Tensor,
                      value_positions: Sequence[Sequence[int]]) -> float:
    """Fraction of R-KV picks (over B,L,H,M) that land on a value-token position."""
    hits = total = 0
    width = int(indices.max()) + 1 if indices.numel() else 0
    for row, positions in enumerate(value_positions):
        chosen = indices[row][selected_mask[row]].cpu().long()
        if chosen.numel() == 0:
            continue
        table = torch.zeros(max(width, 1), dtype=torch.bool)
        for position in positions:
            if 0 <= int(position) < table.numel():
                table[int(position)] = True
        hits += int(table[chosen].sum())
        total += int(chosen.numel())
    return hits / total if total else float("nan")


# ------------------------------------------------------- generative comparison


def slot_readout_logits(model, latent_states: torch.Tensor) -> torch.Tensor:
    """Apply the (tied) vocabulary head to post-ln_f slot states: ``[B,M,V]``."""
    head = official_codi_base_model(model).get_output_embeddings()
    return head(latent_states)


def reconstruction_loss(
    logits: torch.Tensor, *, slots: Sequence[int],
    value_token_ids_by_row: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Cross-entropy of each supervised slot's readout toward its assigned value token."""
    batch, positions, vocab = logits.shape
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    mask = torch.zeros(batch, positions, dtype=torch.bool, device=logits.device)
    losses = []
    assignments = []
    for row in range(batch):
        tokens = [int(t) for t in value_token_ids_by_row[row] if 0 <= int(t) < vocab]
        record = [-1] * positions
        if tokens:
            cost = -log_probs[row][list(slots)][:, tokens].detach()  # [S, C]
            assignment = assign_slots(cost)
            for slot, choice in zip(slots, assignment):
                if choice < 0:
                    continue
                losses.append(-log_probs[row, slot, tokens[choice]])
                mask[row, slot] = True
                record[slot] = tokens[choice]
        assignments.append(record)
    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = logits.sum() * 0.0
    return loss, mask, assignments


# -------------------------------------------------------------- training step


@dataclass
class StepResult:
    gradients: tuple
    answer_loss: float
    endpoint_loss: float
    auxiliary_loss: float | None
    auxiliary_scale: float | None
    supervised_slots: float | None
    rkv_value_overlap: float | None


def trajectory_training_step(
    model, batch, spec: ArmSpec, parameters: Sequence[torch.Tensor], *,
    latent_positions: int, value_positions: Sequence[Sequence[int]],
    value_token_ids: Sequence[Sequence[int]], random_generator: torch.Generator | None,
    importance_weight: float,
) -> StepResult:
    """One arm's gradients: answer NLL + endpoint loss + norm-matched auxiliary.

    The auxiliary gradient is rescaled to the endpoint-loss gradient norm, so arms
    that differ in *what* they supervise receive the same gradient pressure.
    """
    teacher_endpoint = teacher_endpoint_states(model, batch)
    trace = extract_teacher_trace(model, batch) if spec.auxiliary == "kv" else None
    student = student_trajectory_forward(model, batch, latent_positions=latent_positions)
    endpoint = endpoint_hidden_loss(student.answer_endpoint_hidden, teacher_endpoint)
    needs_auxiliary = spec.auxiliary is not None
    base = autograd_gradients(student.mean_loss, parameters, retain_graph=True)
    endpoint_gradients = autograd_gradients(endpoint, parameters, retain_graph=needs_auxiliary)
    total = combine_gradients(base, endpoint_gradients)
    auxiliary_value = scale = supervised = overlap = None
    if spec.auxiliary == "kv":
        if spec.selector == "rkv":
            tk, tv, mask, indices = rkv_slot_targets(
                trace, slots=latent_positions, importance_weight=importance_weight
            )
            overlap = rkv_value_overlap(indices, mask, value_positions)
            supervised = float(mask.float().mean())
        else:
            if spec.selector == "value":
                candidates = value_positions
            elif spec.selector == "random":
                if random_generator is None:
                    raise ValueError("random selector needs a generator")
                candidates = random_trace_positions(
                    trace.mask, [len(v) for v in value_positions], generator=random_generator
                )
            else:
                raise ValueError(f"unknown selector {spec.selector!r}")
            tk, tv, mask, _ = build_slot_targets(
                student.latent_keys, student.latent_values, trace,
                slots=spec.slots, candidates_by_row=candidates,
            )
            supervised = float(mask.float().sum() / (mask.shape[0] * len(spec.slots)))
        auxiliary = kv_match_loss(
            student.latent_keys, student.latent_values, tk, tv, mask=mask, metric="l1"
        )
    elif spec.auxiliary == "recon":
        logits = slot_readout_logits(model, student.latent_states)
        auxiliary, mask, _ = reconstruction_loss(
            logits, slots=spec.slots, value_token_ids_by_row=value_token_ids
        )
        supervised = float(mask.float().sum() / (mask.shape[0] * len(spec.slots)))
    if needs_auxiliary:
        auxiliary_value = float(auxiliary.detach())
        raw = autograd_gradients(auxiliary, parameters, retain_graph=False)
        if any(g is not None and bool(g.abs().sum() > 0) for g in raw):
            matched, matching = match_gradient_norm(raw, endpoint_gradients)
            scale = float(matching["auxiliary_scale"])
            total = combine_gradients(total, matched)
        else:
            scale = 0.0
    return StepResult(
        gradients=total,
        answer_loss=float(student.mean_loss.detach()),
        endpoint_loss=float(endpoint.detach()),
        auxiliary_loss=auxiliary_value,
        auxiliary_scale=scale,
        supervised_slots=supervised,
        rkv_value_overlap=overlap,
    )
