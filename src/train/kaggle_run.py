"""Session-safe training entrypoint.

Dispatches on `cfg.task` to a task builder, then hands the model + step function to the
generic `Trainer` (checkpoint / resume / wall-clock guard). The loop is identical across:
  - dummy  (Phase 0): synthetic task that validates the harness on CPU.
  - sft    (Phase 1b): real GPT-2 SFT baselines (No-CoT / CoT).
  - latent (Phase 2): shared continuous-thought model with CODI / KaVa losses.

Exit codes:
    0   reached total_steps (complete)
    42  hit the wall-clock budget; state saved, re-run to resume (incomplete)

Run:
    python -m src.train.kaggle_run --config configs/phase0.yaml
    python -m src.train.kaggle_run --config configs/sft_cot.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

from src.train.trainer import EXIT_COMPLETE, EXIT_RESUME_NEEDED, Trainer
from src.utils.config import Config, load_config
from src.utils.seeding import set_seed

__all__ = ["run", "build_task", "EXIT_COMPLETE", "EXIT_RESUME_NEEDED"]


def _set_offline(offline: bool) -> None:
    # The augmented GSM8k files are small enough for regular HTTP; disabling Xet avoids
    # long unauthenticated transfer stalls observed in local and notebook environments.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# --------------------------------------------------------------------------- #
# Phase-0 synthetic task. Batches are a deterministic function of the step
# index, so resume continuity holds regardless of RNG drift.
# --------------------------------------------------------------------------- #
class DummyTask:
    def __init__(self, cfg: Config):
        self.in_dim = cfg.dummy.in_dim
        self.noise_std = cfg.dummy.noise_std
        self.batch_size = cfg.train.batch_size
        g = torch.Generator().manual_seed(1234)
        self.w_true = torch.randn(self.in_dim, 1, generator=g)

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(step)
        x = torch.randn(self.batch_size, self.in_dim, generator=g)
        noise = torch.randn(self.batch_size, 1, generator=g) * self.noise_std
        y = x @ self.w_true + noise
        return x, y


def build_dummy_task(cfg: Config):
    task = DummyTask(cfg)
    model = nn.Linear(cfg.dummy.in_dim, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    loss_fn = nn.MSELoss()

    def step_fn(step: int) -> float:
        x, y = task.batch(step)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    return model, optimizer, step_fn


def _task_type(cfg: Config) -> str:
    task = cfg.get("task", None)
    if task is None:
        return "dummy"
    if isinstance(task, str):
        return task
    return task.get("type", "dummy")


def build_task(cfg: Config):
    """Returns (model, optimizer, step_fn) for the configured task."""
    ttype = _task_type(cfg)
    if ttype == "dummy":
        return build_dummy_task(cfg)
    if ttype == "sft":
        from src.train.sft import build_sft_task

        return build_sft_task(cfg)
    if ttype in {"latent", "codi", "kava"}:
        from src.train.latent import build_latent_task

        # Accept the concise legacy task.type names while keeping `type: latent` as the
        # canonical schema used by the checked-in configs.
        if ttype in {"codi", "kava"} and "method" not in cfg.task:
            cfg["task"]["method"] = ttype
        return build_latent_task(cfg)
    raise ValueError(f"unknown task type: {ttype!r}")


def run(cfg: Config) -> tuple[list[float], int]:
    _set_offline(bool(cfg.get("offline", False)))
    set_seed(cfg.seed)
    model, optimizer, step_fn = build_task(cfg)
    return Trainer(cfg, model, optimizer, step_fn).fit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Session-safe trainer.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--set", nargs="*", default=[], help="Dot overrides, e.g. train.lr=0.02")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    _, code = run(cfg)
    return code


if __name__ == "__main__":
    sys.exit(main())
