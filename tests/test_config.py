"""Config loading and CLI-override coercion tests."""
from __future__ import annotations

from src.utils.config import load_config


def test_lowercase_cli_boole_and_null_are_typed(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("enabled: true\nvalue: 1\n", encoding="utf-8")
    cfg = load_config(
        path,
        overrides=["enabled=false", "offline=true", "optional=null"],
    )
    assert cfg.enabled is False
    assert cfg.offline is True
    assert cfg.optional is None


def test_nested_numeric_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  lr: 0.1\n", encoding="utf-8")
    cfg = load_config(path, overrides=["train.lr=0.0001", "train.steps=10"])
    assert cfg.train.lr == 0.0001
    assert cfg.train.steps == 10
