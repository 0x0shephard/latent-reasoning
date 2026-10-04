"""Cache-carrier audit on CODI's latent trajectory (ledger §108).

Every thought slot reaches the answer by two routes: its output state, projected into
the next thought's input (route a, the object a linear monitor reads), and the K/V it
writes at every layer, which all later positions attend to (route b).  §106 swapped
route (a) only.  This module swaps either route, or both, between derangement-paired
questions on the teacher-forced latent path and on native decoding, with no training.

Slot ``s``'s K/V are the cache entries written by the forward pass that produced
slot ``s``'s state: after that pass they are the last cache position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.data.official_codi_training import collate_official_codi_kv_rows
from src.mech.kv_risk_cache import cache_to_legacy
from src.mech.trajectory_supervision import _student_answer_io
from src.models.official_codi import generate_official_codi


@dataclass(frozen=True)
class CacheEdit:
    """Replace, at ``slot``, the keys and/or values of ``layers`` by the donor's."""
    slot: int
    layers: tuple[int, ...]
    keys: bool
    values: bool


@dataclass
class Trajectory:
    states: torch.Tensor   # [N, S, D]      pre-projector output state of each thought
    keys: torch.Tensor     # [N, S, L, H, h] keys written by each thought at every layer
    values: torch.Tensor   # [N, S, L, H, h]


def _last_position_kv(cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V at the newest cache position, stacked over layers: ``[B, L, H, h]``."""
    legacy = cache_to_legacy(cache)
    keys = torch.stack([layer[0][:, :, -1, :] for layer in legacy], dim=1)
    values = torch.stack([layer[1][:, :, -1, :] for layer in legacy], dim=1)
    return keys, values


def patch_last_position(cache: Any, *, donor_keys: torch.Tensor | None, donor_values: torch.Tensor | None,
                        layers: Sequence[int]) -> Any:
    """In place: overwrite the newest cache position's K and/or V at ``layers`` with the
    donor's ``[B, L, H, h]`` tensors.  Returns the same cache object."""
    legacy = cache_to_legacy(cache)
    for layer in layers:
        k, v = legacy[layer]
        if donor_keys is not None:
            k[:, :, -1, :] = donor_keys[:, layer].to(k.device, k.dtype)
        if donor_values is not None:
            v[:, :, -1, :] = donor_values[:, layer].to(v.device, v.dtype)
    return cache


def apply_edits(cache: Any, slot: int, edits: Sequence[CacheEdit], donor: Trajectory | None, rows: slice) -> Any:
    for edit in edits:
        if edit.slot != slot:
            continue
        if donor is None:
            raise ValueError("cache edits need donor trajectories")
        patch_last_position(cache, donor_keys=donor.keys[rows, slot] if edit.keys else None,
                            donor_values=donor.values[rows, slot] if edit.values else None, layers=edit.layers)
    return cache


def latent_path(model, batch, *, latent_positions: int, hidden_donors: dict[int, torch.Tensor] | None = None,
                cache_edits: Sequence[CacheEdit] = (), donor: Trajectory | None = None, rows: slice = slice(None),
                record: bool = False):
    """Teacher-forced latent path to the answer cue with optional route swaps.

    ``hidden_donors[slot]`` is ``[N, D]`` (indexed by ``rows``) and replaces the state
    fed to the projector at that slot.  ``cache_edits`` replace K/V of the position
    that slot wrote.  Returns ``(first-token logits [B, V], gold first token [B],
    recorded Trajectory or None)``.
    """
    encoded = model.codi(input_ids=batch.student_question_ids, attention_mask=batch.student_question_mask,
                         use_cache=True, output_hidden_states=True, return_dict=True)
    cache = encoded.past_key_values
    latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
    states, keys, values = [], [], []
    for position in range(latent_positions):
        out = model.codi(inputs_embeds=latent, past_key_values=cache, use_cache=True,
                         output_hidden_states=True, return_dict=True)
        cache = apply_edits(out.past_key_values, position, cache_edits, donor, rows)
        state = out.hidden_states[-1][:, -1, :]
        if record:
            k, v = _last_position_kv(cache)
            states.append(state.detach().float().cpu()); keys.append(k.detach().float().cpu()); values.append(v.detach().float().cpu())
        if hidden_donors and position in hidden_donors:
            state = hidden_donors[position][rows].to(state.device, state.dtype)
        latent = model.prj(state.unsqueeze(1))
    answer_inputs, _, _ = _student_answer_io(batch, eot_token_id=model.eot_id, pad_token_id=model.pad_token_id)
    decoded = model.codi(inputs_embeds=model.input_embeddings()(answer_inputs), past_key_values=cache,
                         use_cache=True, output_hidden_states=False, return_dict=True)
    endpoints = (batch.teacher_answer_start - batch.teacher_trace_end).to(answer_inputs.device)
    row = torch.arange(answer_inputs.shape[0], device=answer_inputs.device)
    logits = decoded.logits[row, endpoints, : model.eot_id]
    gold = batch.teacher_ids[row, batch.teacher_answer_start.to(batch.teacher_ids.device)].to(logits.device)
    recorded = None
    if record:
        recorded = Trajectory(states=torch.stack(states, 1), keys=torch.stack(keys, 1), values=torch.stack(values, 1))
    return logits, gold, recorded


@torch.no_grad()
def record_trajectory(model, tokenizer, rows: Sequence[dict], *, latent_positions: int, batch_size: int, device):
    """Baseline pass: ``(Trajectory, predicted first token [N], gold first token [N])``."""
    model.eval(); parts, preds, golds = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(tokenizer, rows[start:start + batch_size], bot_token_id=model.bot_id,
                                              enforce_answer_eligibility=False).to(device)
        logits, gold, rec = latent_path(model, batch, latent_positions=latent_positions, record=True)
        parts.append(rec); preds.append(logits.argmax(-1).cpu()); golds.append(gold.cpu())
    traj = Trajectory(states=torch.cat([p.states for p in parts]), keys=torch.cat([p.keys for p in parts]),
                      values=torch.cat([p.values for p in parts]))
    return traj, torch.cat(preds), torch.cat(golds)


@torch.no_grad()
def outcomes_under(model, tokenizer, rows: Sequence[dict], *, latent_positions: int, batch_size: int, device,
                   hidden_donors: dict[int, torch.Tensor] | None = None, cache_edits: Sequence[CacheEdit] = (),
                   donor: Trajectory | None = None) -> torch.Tensor:
    """Predicted first answer token for every row under the given route swaps."""
    model.eval(); preds = []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(tokenizer, rows[start:start + batch_size], bot_token_id=model.bot_id,
                                              enforce_answer_eligibility=False).to(device)
        logits, _, _ = latent_path(model, batch, latent_positions=latent_positions, hidden_donors=hidden_donors,
                                   cache_edits=cache_edits, donor=donor, rows=slice(start, start + batch_size))
        preds.append(logits.argmax(-1).cpu())
    return torch.cat(preds)


class _ChunkTracker:
    """``generate_official_codi`` calls its hooks per chunk and per thought without a
    chunk offset for ``kv_intervention``; recover it from the call order."""

    def __init__(self):
        self.start, self.size = 0, None

    def rows(self, position: int, batch_rows: int) -> slice:
        if position == 0:
            if self.size is not None:
                self.start += self.size
            self.size = batch_rows
        return slice(self.start, self.start + batch_rows)


def record_native_trajectory(model, tokenizer, questions: Sequence[str], *, latent_iterations: int, batch_size: int,
                             device, max_new_tokens: int = 64) -> tuple[Trajectory, list[str]]:
    """Native greedy decoding with the released path, recording every thought's state and K/V."""
    states: list[torch.Tensor] = []; keys: list[torch.Tensor] = []; values: list[torch.Tensor] = []
    chunk_states: list[torch.Tensor] = []; chunk_k: list[torch.Tensor] = []; chunk_v: list[torch.Tensor] = []

    def kv_hook(cache, position):
        k, v = _last_position_kv(cache)
        chunk_k.append(k.detach().float().cpu()); chunk_v.append(v.detach().float().cpu())
        return cache

    def state_hook(state, position, start):
        chunk_states.append(state.detach().float().cpu())
        if position == latent_iterations - 1:
            states.append(torch.stack(chunk_states, 1)); keys.append(torch.stack(chunk_k, 1)); values.append(torch.stack(chunk_v, 1))
            chunk_states.clear(); chunk_k.clear(); chunk_v.clear()
        return state

    with torch.inference_mode():
        outputs = generate_official_codi(model, tokenizer, list(questions), latent_iterations=latent_iterations,
                                         max_new_tokens=max_new_tokens, batch_size=batch_size, device=device,
                                         kv_intervention=kv_hook, latent_state_hook=state_hook)
    return Trajectory(states=torch.cat(states), keys=torch.cat(keys), values=torch.cat(values)), outputs


def generate_under(model, tokenizer, questions: Sequence[str], *, latent_iterations: int, batch_size: int, device,
                   hidden_donors: dict[int, torch.Tensor] | None = None, cache_edits: Sequence[CacheEdit] = (),
                   donor: Trajectory | None = None, max_new_tokens: int = 64) -> list[str]:
    """Native greedy decoding with the route swaps applied inside the released path."""
    tracker = _ChunkTracker()
    current: dict[str, slice] = {}

    def kv_hook(cache, position):
        current["rows"] = tracker.rows(position, cache_to_legacy(cache)[0][0].shape[0])
        return apply_edits(cache, position, cache_edits, donor, current["rows"])

    def state_hook(state, position, start):
        if hidden_donors and position in hidden_donors:
            return hidden_donors[position][start:start + state.shape[0]].to(state.device, state.dtype)
        return state

    with torch.inference_mode():
        return generate_official_codi(model, tokenizer, list(questions), latent_iterations=latent_iterations,
                                      max_new_tokens=max_new_tokens, batch_size=batch_size, device=device,
                                      kv_intervention=kv_hook, latent_state_hook=state_hook)


def donor_view(trajectory: Trajectory, perm: Sequence[int]) -> Trajectory:
    index = torch.tensor(list(perm))
    return Trajectory(states=trajectory.states[index], keys=trajectory.keys[index], values=trajectory.values[index])
