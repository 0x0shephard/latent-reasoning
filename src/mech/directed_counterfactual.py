"""Directed counterfactual analysis on CODI's latent trajectory (ledger §110)."""
from __future__ import annotations

from typing import Sequence

import torch

from src.mech.latent_workspace import decode_thought_numbers
from src.models.official_codi import official_codi_base_model


def readout_matrix(model) -> torch.Tensor:
    """The tied output head ``[V, D]`` the released decoder reads thoughts with."""
    head = official_codi_base_model(model).get_output_embeddings()
    return head.weight.detach().float().cpu()


def slot_numbers(states: torch.Tensor, readout: torch.Tensor, tokenizer, *, top_k: int = 5) -> list[list[set[str]]]:
    """``[row][slot] -> numeric strings`` among each slot state's top-k readout tokens."""
    return decode_thought_numbers(states.unsqueeze(2), readout, tokenizer, state=0, top_k=top_k)


def locate_pairs(numbers: list[list[set[str]]], partner: Sequence[int], changed_values: Sequence[Sequence[tuple[str, str]]],
                 odd_slots: Sequence[int]) -> list[list[int]]:
    """For each row, the odd slots where some changed result's own value is decoded in
    the row and the partner's value is decoded at the same slot in the partner."""
    located = []
    for i, j in enumerate(partner):
        slots = []
        for s in odd_slots:
            mine, theirs = numbers[i][s], numbers[j][s]
            if any(old in mine and new in theirs for old, new in changed_values[i]):
                slots.append(s)
        located.append(slots)
    return located


def outcome(pred: torch.Tensor, own_gold: torch.Tensor, target_gold: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    n = int(mask.sum())
    if n == 0:
        return {"target": None, "retain": None, "other": None, "n": 0}
    target = float((pred[mask] == target_gold[mask]).double().mean())
    retain = float((pred[mask] == own_gold[mask]).double().mean())
    return {"target": target, "retain": retain, "other": 1.0 - target - retain, "n": n}


def located_specificity(preds: dict[str, torch.Tensor], located: list[list[int]], own_gold, target_gold, *,
                        feeding_slots: Sequence[int] = (1, 3)) -> dict:
    """Target rates at the located site versus the other feeding site, for the state
    route (``state_s``) and the following even store (``v89_{s+1}``)."""
    rows = torch.arange(len(located))
    report = {}
    for s in feeding_slots:
        mask = torch.tensor([s in slots for slots in located])
        other = [o for o in feeding_slots if o != s]
        report[str(s)] = {
            "n": int(mask.sum()),
            "state_at_site": outcome(preds[f"state_{s}"], own_gold, target_gold, mask),
            "state_elsewhere": {str(o): outcome(preds[f"state_{o}"], own_gold, target_gold, mask) for o in other},
            "store_at_site": outcome(preds[f"v89_{s + 1}"], own_gold, target_gold, mask),
            "store_elsewhere": {str(o + 1): outcome(preds[f"v89_{o + 1}"], own_gold, target_gold, mask) for o in other},
        }
    del rows
    return report
