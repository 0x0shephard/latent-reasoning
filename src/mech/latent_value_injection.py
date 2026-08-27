"""Write values into CODI's latent workspace during the latent loop.

§55 confirmed the six thoughts store the solution's intermediate values at the odd
slots, read through the model's own vocabulary. This module holds the causal tier:
an intervention that *adds a value token's readout direction* to the propagating
hidden state at the value slots, during the released latent loop. Repair arms add
the gold intermediates on top of a wrong run; corruption arms add plausible wrong
values on top of a correct run; matched random numeric tokens are the control.

The edit enters at state 11 (block 10's output) of the value-slot latent passes, so
it propagates through block 11's KV contribution, ``ln_f``, the projection, and
every later pass — the same entry point §50 validated for state-space edits.
"""
from __future__ import annotations

import re

import torch

from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE


VALUE_INJECTION_SCHEMA_VERSION = 1
VALUE_INJECTION_CONTRACT = "frozen_checkpoint_latent_value_injection_v1"

#: The odd thoughts are the measured value slots (§55: hits [0, 263, 0, 228, 1, 227]).
VALUE_SLOTS = (1, 3, 5)

_NUMERIC_TOKEN = re.compile(r"^-?\d+$")


def _transformer_parts(model):
    transformer = getattr(model.codi, "base_model", model.codi)
    transformer = getattr(transformer, "model", transformer)
    transformer = getattr(transformer, "transformer", transformer)
    blocks = getattr(transformer, "h", None)
    if blocks is None or len(blocks) != 12:
        raise RuntimeError("unexpected GPT-2 module layout")
    return blocks


def value_token_id(tokenizer, value: str) -> int:
    """First token of `` {value}`` — the same convention the lens decodes."""
    ids = tokenizer(f" {value}", add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"value {value!r} does not tokenize")
    return int(ids[0])


def numeric_token_pool(tokenizer, vocabulary_size: int) -> list[int]:
    """All single tokens that decode to a bare integer string."""
    pool = [
        token
        for token in range(vocabulary_size)
        if _NUMERIC_TOKEN.match(tokenizer.decode([token]).strip())
    ]
    if not pool:
        raise RuntimeError("no numeric tokens found in the vocabulary")
    return pool


def build_slot_tokens(
    intermediates_by_row: list[list[str]],
    tokenizer,
    *,
    arm: str,
    vocabulary_size: int,
    random_seed: int,
    slots: tuple[int, ...] = VALUE_SLOTS,
) -> torch.Tensor:
    """Per-question, per-slot target token ids; ``-1`` marks no injection.

    ``gold`` injects the first ``len(slots)`` gold intermediates in order;
    ``offset`` injects those values plus one (a plausible wrong computation);
    ``random`` injects seeded draws from the numeric token pool with exactly the
    same slot mask, so every arm applies the same number of identically scaled
    edits and differs only in *which value* it writes.
    """
    if arm not in ("gold", "offset", "random"):
        raise ValueError(f"unknown injection arm {arm!r}")
    generator = torch.Generator().manual_seed(int(random_seed))
    pool = numeric_token_pool(tokenizer, vocabulary_size)
    pool_tensor = torch.tensor(pool, dtype=torch.long)
    targets = torch.full(
        (len(intermediates_by_row), len(slots)), -1, dtype=torch.long
    )
    for row, intermediates in enumerate(intermediates_by_row):
        for slot_index in range(min(len(slots), len(intermediates))):
            if arm == "gold":
                value = intermediates[slot_index]
            elif arm == "offset":
                try:
                    value = str(int(float(intermediates[slot_index])) + 1)
                except ValueError:
                    value = intermediates[slot_index] + "1"
            else:
                value = None
            if value is None:
                choice = pool_tensor[
                    int(torch.randint(len(pool), (1,), generator=generator))
                ]
                targets[row, slot_index] = int(choice)
            else:
                targets[row, slot_index] = value_token_id(tokenizer, value)
    return targets


class OfficialCODILatentValueInjection:
    """Additive readout-direction edit at state 11 of the value-slot passes.

    The object is also the ``kv_intervention`` callable of
    ``generate_official_codi``: the released latent loop invokes it once per
    latent position, which is how it tracks both the position inside the loop
    and the row cursor across batch chunks. It returns the cache unchanged.

    For row ``r`` at slot ``k`` with target token ``t``: ``h += beta * rms(h) *
    W[t]/||W[t]||``. Rows whose target is ``-1`` are untouched. Answer-decoding
    passes are provably skipped: they occur while the tracked next-position is 0,
    and 0 is never a value slot.
    """

    def __init__(
        self,
        model,
        *,
        readout: torch.Tensor,
        slot_tokens: torch.Tensor,
        beta: float,
        latent_iterations: int,
        slots: tuple[int, ...] = VALUE_SLOTS,
    ) -> None:
        if readout.ndim != 2 or readout.shape[1] != GPT2_HIDDEN_SIZE:
            raise ValueError("readout must be [V, 768]")
        if slot_tokens.ndim != 2 or slot_tokens.shape[1] != len(slots):
            raise ValueError("slot tokens must be [rows, len(slots)]")
        if 0 in slots:
            raise ValueError("slot 0 would collide with answer decoding")
        if not all(0 < slot < latent_iterations for slot in slots):
            raise ValueError("slots must lie inside the latent loop")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        blocks = _transformer_parts(model)
        directions = readout.float()
        self.directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        self.slot_tokens = slot_tokens.long()
        self.beta = float(beta)
        self.slots = tuple(int(s) for s in slots)
        self.latent_iterations = int(latent_iterations)
        self.next_position = 0
        self.row_cursor = 0
        self.rows_edited = 0
        self.edit_squared_norm = 0.0
        self._last_batch = 0
        self._handle = blocks[10].register_forward_hook(self._hook())

    def _hook(self):
        def edit(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.shape[1] != 1 or self.next_position not in self.slots:
                return output
            if self.next_position >= self.latent_iterations:
                return output
            batch = hidden.shape[0]
            self._last_batch = batch
            rows = slice(self.row_cursor, self.row_cursor + batch)
            slot_index = self.slots.index(self.next_position)
            tokens = self.slot_tokens[rows, slot_index].to(hidden.device)
            mask = tokens >= 0
            if self.beta == 0.0 or not bool(mask.any()):
                return output
            state = hidden[:, -1, :]
            directions = self.directions.to(
                device=hidden.device, dtype=state.dtype
            )[tokens.clamp_min(0)]
            scale = state.pow(2).mean(dim=1, keepdim=True).sqrt() * self.beta
            delta = directions * scale * mask.unsqueeze(1)
            hidden = hidden.clone()
            hidden[:, -1, :] = state + delta
            self.rows_edited += int(mask.sum())
            self.edit_squared_norm += float(delta.double().pow(2).sum())
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            if isinstance(output, list):
                return [hidden] + list(output[1:])
            return hidden

        return edit

    def __call__(self, cache, latent_position: int):
        position = int(latent_position)
        if position != self.next_position:
            raise RuntimeError(
                f"latent loop reported position {position}, expected "
                f"{self.next_position}"
            )
        if position == self.latent_iterations - 1:
            self.next_position = 0
            self.row_cursor += self._last_batch
        else:
            self.next_position = position + 1
        return cache

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def diagnostics(self) -> dict:
        return {
            "beta": self.beta,
            "slots": list(self.slots),
            "rows_edited": int(self.rows_edited),
            "edit_rms_norm": (
                (self.edit_squared_norm / self.rows_edited) ** 0.5
                if self.rows_edited
                else 0.0
            ),
        }
