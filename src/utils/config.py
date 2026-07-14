"""Minimal YAML config loader with attribute access and CLI dot-overrides.

Usage:
    cfg = load_config("configs/phase0.yaml", overrides=["train.lr=0.02", "seed=1"])
    cfg.train.lr          # -> 0.02
    cfg.to_dict()         # plain nested dict (for logging / checkpoint metadata)
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """dict that also supports attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict:
        def _plain(v: Any) -> Any:
            if isinstance(v, dict):
                return {k: _plain(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_plain(x) for x in v]
            return v

        return _plain(self)


def _coerce(value: str) -> Any:
    """Turn a CLI string into a bool/int/float/None/list where possible."""
    keywords = {"true": True, "false": False, "null": None, "none": None}
    lowered = value.casefold()
    if lowered in keywords:
        return keywords[lowered]
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _apply_override(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Bad override '{override}', expected key=value")
        key, val = override.split("=", 1)
        _apply_override(raw, key.strip(), _coerce(val.strip()))
    return Config(raw)
