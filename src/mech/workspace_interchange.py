"""Interchange interventions on CODI's latent workspace (ledger §105 go/no-go).

The §55 workspace finding says the odd thought slots hold the solution's intermediate
values.  Before training a student by interchange-intervention (DIITO-style) on those
slots, this module measures on the *official* model whether swapping a slot's state,
or a low-rank subspace of it, between two questions changes the answer; whether that
effect is specific to the odd slots; and whether a gradient score over the slot's
principal directions ranks them the way the interventions do.  Only if the slot
mediates the answer, through a low-rank subspace, and gradients disagree with
interventions, can an intervention-selected trajectory target differ from a
gradient-selected one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from src.data.official_codi_training import collate_official_codi_kv_rows
from src.mech.causal_subspace_distillation import TeacherPCA, fit_teacher_pca
from src.mech.trajectory_supervision import _student_answer_io
from src.models.official_codi import generate_official_codi


@dataclass
class SlotRecording:
    states: torch.Tensor      # [N, S, D] last hidden state of each thought, before the projector
    outputs: list[str]        # native greedy generations


def record_slot_states(model, tokenizer, questions: Sequence[str], *, latent_iterations: int,
                       batch_size: int, device, max_new_tokens: int = 64) -> SlotRecording:
    """Run the released latent path once and keep every thought's state."""
    store: dict[int, dict[int, torch.Tensor]] = {}

    def hook(state, position, start):
        store.setdefault(start, {})[position] = state.detach().float().cpu()
        return state

    with torch.inference_mode():
        outputs = generate_official_codi(model, tokenizer, list(questions), latent_iterations=latent_iterations,
                                         max_new_tokens=max_new_tokens, batch_size=batch_size, device=device,
                                         latent_state_hook=hook)
    chunks = []
    for start in sorted(store):
        chunks.append(torch.stack([store[start][p] for p in range(latent_iterations)], dim=1))
    return SlotRecording(states=torch.cat(chunks), outputs=outputs)


def interchange_hook(slot: int, donor_states: torch.Tensor, basis: torch.Tensor | None):
    """Hook that, at ``slot``, moves each row toward its donor's state inside ``basis``
    (orthonormal columns ``[D, r]``); ``basis=None`` swaps the whole state."""
    def hook(state, position, start):
        if position != slot:
            return state
        donor = donor_states[start:start + state.shape[0]].to(state.device, state.dtype)
        if basis is None:
            return donor
        b = basis.to(state.device, state.dtype)
        return state + ((donor - state) @ b) @ b.T
    return hook


def generate_with_interchange(model, tokenizer, questions: Sequence[str], donor_states: torch.Tensor, *,
                              slot: int, basis: torch.Tensor | None, latent_iterations: int, batch_size: int,
                              device, max_new_tokens: int = 64) -> list[str]:
    with torch.inference_mode():
        return generate_official_codi(model, tokenizer, list(questions), latent_iterations=latent_iterations,
                                      max_new_tokens=max_new_tokens, batch_size=batch_size, device=device,
                                      latent_state_hook=interchange_hook(slot, donor_states, basis))


def _latent_path_first_token_logits(model, batch, *, latent_positions: int, slot: int | None = None,
                                    donor: torch.Tensor | None = None, basis: torch.Tensor | None = None,
                                    grad_slot: int | None = None):
    """Teacher-forced latent path to the answer cue; returns (first-token logits [B,V],
    gold first token [B], slot state that requires grad or None)."""
    encoded = model.codi(input_ids=batch.student_question_ids, attention_mask=batch.student_question_mask,
                         use_cache=True, output_hidden_states=True, return_dict=True)
    cache = encoded.past_key_values
    latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
    grad_state = None
    for position in range(latent_positions):
        out = model.codi(inputs_embeds=latent, past_key_values=cache, use_cache=True,
                         output_hidden_states=True, return_dict=True)
        cache = out.past_key_values
        state = out.hidden_states[-1][:, -1, :]
        if slot is not None and position == slot and donor is not None:
            d = donor.to(state.device, state.dtype)
            if basis is None:
                state = d
            else:
                b = basis.to(state.device, state.dtype)
                state = state + ((d - state) @ b) @ b.T
        if grad_slot is not None and position == grad_slot:
            state = state.detach().requires_grad_(True)
            grad_state = state
        latent = model.prj(state.unsqueeze(1))
    answer_inputs, _, _ = _student_answer_io(batch, eot_token_id=model.eot_id, pad_token_id=model.pad_token_id)
    decoded = model.codi(inputs_embeds=model.input_embeddings()(answer_inputs), past_key_values=cache,
                         use_cache=True, output_hidden_states=False, return_dict=True)
    endpoints = (batch.teacher_answer_start - batch.teacher_trace_end).to(answer_inputs.device)
    row = torch.arange(answer_inputs.shape[0], device=answer_inputs.device)
    logits = decoded.logits[row, endpoints, : model.eot_id]
    gold = batch.teacher_ids[row, batch.teacher_answer_start.to(batch.teacher_ids.device)].to(logits.device)
    return logits, gold, grad_state


@torch.no_grad()
def first_token_outcomes(model, tokenizer, rows: Sequence[dict], *, latent_positions: int, batch_size: int,
                         device, slot: int | None = None, donor_states: torch.Tensor | None = None,
                         basis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(predicted first answer token, gold first token) under the forced cue, optionally
    with an interchange at ``slot`` toward ``donor_states`` inside ``basis``."""
    model.eval(); preds, golds = [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(tokenizer, rows[start:start + batch_size], bot_token_id=model.bot_id,
                                              enforce_answer_eligibility=False).to(device)
        donor = None if donor_states is None else donor_states[start:start + batch_size]
        logits, gold, _ = _latent_path_first_token_logits(model, batch, latent_positions=latent_positions,
                                                          slot=slot, donor=donor, basis=basis)
        preds.append(logits.argmax(-1).cpu()); golds.append(gold.cpu())
    return torch.cat(preds), torch.cat(golds)


def slot_gradient_scores(model, tokenizer, rows: Sequence[dict], *, latent_positions: int, slot: int,
                         pca: TeacherPCA, batch_size: int, device, candidates: int) -> torch.Tensor:
    """Relevance-style score per principal direction of the slot state: mean squared
    projection of the gradient of the gold first-token log-probability (the §86
    relevance selector, moved to the trajectory)."""
    model.eval()
    basis = pca.basis[:, :candidates].to(device, torch.float32)
    total = torch.zeros(candidates, dtype=torch.float64)
    n = 0
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(tokenizer, rows[start:start + batch_size], bot_token_id=model.bot_id,
                                              enforce_answer_eligibility=False).to(device)
        with torch.enable_grad():
            logits, gold, state = _latent_path_first_token_logits(model, batch, latent_positions=latent_positions,
                                                                  grad_slot=slot)
            log_prob = F.log_softmax(logits.float(), dim=-1).gather(1, gold[:, None]).sum()
            grad, = torch.autograd.grad(log_prob, state, allow_unused=True)
        if grad is None:  # the terminal slot's output state is never consumed by the released path
            grad = torch.zeros_like(state)
        proj = grad.detach().float() @ basis
        total += (proj ** 2).sum(0).double().cpu()
        n += grad.shape[0]
    return total / max(1, n)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = torch.tensor(list(a), dtype=torch.float64); b = torch.tensor(list(b), dtype=torch.float64)
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = float((ra.norm() * rb.norm()).clamp_min(1e-12))
    return float((ra @ rb) / denom)


def change_rate(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((before != after).double().mean())


def derangement(n: int, seed: int) -> list[int]:
    """A pairing with no fixed point, so every question gets a different donor."""
    g = torch.Generator().manual_seed(int(seed))
    while True:
        perm = torch.randperm(n, generator=g).tolist()
        if all(i != p for i, p in enumerate(perm)):
            return perm


def fit_slot_pca(states: torch.Tensor) -> TeacherPCA:
    return fit_teacher_pca(states.float())
