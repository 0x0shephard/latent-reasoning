"""Atomic, rolling checkpointing with idempotent resume.

A checkpoint bundles training step, model/optimizer state, RNG state, and config
metadata so a killed Kaggle session resumes with no metric discontinuity.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import torch


class Checkpointer:
    def __init__(self, output_dir: str | Path, keep_last: int = 3):
        self.dir = Path(output_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def _path(self, step: int) -> Path:
        return self.dir / f"step_{step:08d}.pt"

    def latest_step(self) -> Optional[int]:
        ckpts = sorted(self.dir.glob("step_*.pt"))
        if not ckpts:
            return None
        return int(ckpts[-1].stem.split("_")[1])

    def save(self, step: int, payload: dict[str, Any]) -> Path:
        """Atomic save: write to a temp file, then rename into place."""
        target = self._path(step)
        tmp = target.with_suffix(".pt.tmp")
        torch.save({"step": step, **payload}, tmp)
        tmp.replace(target)  # atomic on the same filesystem
        self._prune()
        return target

    def load_latest(self, map_location: str = "cpu") -> Optional[dict[str, Any]]:
        step = self.latest_step()
        if step is None:
            return None
        return torch.load(self._path(step), map_location=map_location, weights_only=False)

    def _prune(self) -> None:
        ckpts = sorted(self.dir.glob("step_*.pt"))
        for old in ckpts[: -self.keep_last] if self.keep_last > 0 else []:
            old.unlink(missing_ok=True)


def clear_checkpoints(output_dir: str | Path) -> None:
    """Remove all checkpoints (used by tests / fresh runs)."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
