"""The checked-in CODI/KaVa configs preserve the controlled comparison."""
from __future__ import annotations

import copy

from scripts.validate_phase2 import _controlled_differences
from src.utils.config import load_config


def test_primary_phase2_configs_match_all_controlled_fields():
    codi = load_config("configs/codi.yaml")
    kava = load_config("configs/kava.yaml")
    assert _controlled_differences(codi, kava) == []


def test_peer_check_detects_latent_budget_or_hidden_loss_confounds():
    codi = load_config("configs/codi.yaml")
    changed = copy.deepcopy(load_config("configs/kava.yaml"))
    changed["task"]["latent_steps"] = 24
    changed["task"]["distillation"]["hidden_weight"] = 10.0
    fields = {item["field"] for item in _controlled_differences(codi, changed)}
    assert "task.latent_steps" in fields
    assert "task.distillation.hidden_weight" in fields
