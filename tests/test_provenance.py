"""Experiment-manifest tests: safe resume may extend runtime, not change science."""
from __future__ import annotations

import copy

import pytest

from src.utils.config import Config
from src.utils.provenance import prepare_run_manifest


def _cfg(output_dir, data_config=None):
    value = {
        "run_name": "test",
        "seed": 0,
        "offline": True,
        "output_dir": str(output_dir),
        "train": {
            "total_steps": 10,
            "batch_size": 2,
            "lr": 1e-4,
            "ckpt_every": 5,
            "keep_last": 2,
            "max_seconds": 60,
        },
    }
    if data_config is not None:
        value["data_config"] = str(data_config)
    return Config(value)


def test_manifest_allows_runtime_extension(tmp_path):
    cfg = _cfg(tmp_path / "run")
    original = prepare_run_manifest(cfg.output_dir, cfg)
    extended = copy.deepcopy(cfg)
    extended["train"]["total_steps"] = 20
    extended["train"]["max_seconds"] = 120
    resumed = prepare_run_manifest(extended.output_dir, extended)
    assert resumed["fingerprint"] == original["fingerprint"]


def test_manifest_rejects_changed_seed_or_optimizer(tmp_path):
    cfg = _cfg(tmp_path / "run")
    prepare_run_manifest(cfg.output_dir, cfg)
    changed = copy.deepcopy(cfg)
    changed["seed"] = 1
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        prepare_run_manifest(changed.output_dir, changed)


def test_manifest_fingerprints_data_config_contents(tmp_path):
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text("prompt:\n  answer_prefix: 'Answer:'\n", encoding="utf-8")
    cfg = _cfg(tmp_path / "run", data_cfg)
    prepare_run_manifest(cfg.output_dir, cfg)
    data_cfg.write_text("prompt:\n  answer_prefix: 'Result:'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        prepare_run_manifest(cfg.output_dir, cfg)

