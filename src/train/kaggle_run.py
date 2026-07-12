"""Session-safe training entrypoint.

Phase 0: drives a tiny synthetic regression task to prove the harness end-to-end —
deterministic, checkpointing on a cadence, exiting cleanly before a wall-clock cap,
and resuming with NO metric discontinuity. Later phases swap `build_task` for the real
latent-LM model + GSM8k data behind the same loop.

Exit codes:
    0   training reached total_steps (complete)
    42  hit the wall-clock budget; state saved, re-run to resume (incomplete)

Run:
    python -m src.train.kaggle_run --config configs/phase0.yaml
    # ...kill it mid-run, then run the same command again -> it resumes.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.checkpoint import Checkpointer
from src.utils.config import Config, load_config
from src.utils.seeding import load_rng_state, rng_state, set_seed
from src.utils.time_budget import TimeBudget

EXIT_COMPLETE = 0
EXIT_RESUME_NEEDED = 42


def _set_offline(offline: bool) -> None:
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
        # Fixed ground-truth weights, seeded independently of training RNG.
        g = torch.Generator().manual_seed(1234)
        self.w_true = torch.randn(self.in_dim, 1, generator=g)

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(step)  # deterministic per step
        x = torch.randn(self.batch_size, self.in_dim, generator=g)
        noise = torch.randn(self.batch_size, 1, generator=g) * self.noise_std
        y = x @ self.w_true + noise
        return x, y


def build_task(cfg: Config):
    """Returns (model, optimizer, step_fn). Swapped out in later phases."""
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


def run(cfg: Config) -> tuple[list[float], int]:
    """Train (or resume) to completion / budget. Returns (per-step losses, exit code)."""
    _set_offline(bool(cfg.get("offline", False)))
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpointer(output_dir, keep_last=cfg.train.get("keep_last", 3))
    budget = TimeBudget(cfg.train.max_seconds)

    model, optimizer, step_fn = build_task(cfg)

    start_step = 0
    losses: list[float] = []
    resumed = ckpt.load_latest()
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        load_rng_state(resumed["rng"])
        start_step = resumed["step"]
        losses = resumed.get("losses", [])
        print(f"[resume] continuing from step {start_step}")

    total = cfg.train.total_steps
    for step in range(start_step, total):
        if budget.should_stop():
            ckpt.save(step, _payload(step, model, optimizer, losses))
            print(f"[budget] wall-clock guard hit at step {step}; state saved. Re-run to resume.")
            return losses, EXIT_RESUME_NEEDED

        losses.append(step_fn(step))
        done = step + 1
        if done % cfg.train.ckpt_every == 0 or done == total:
            ckpt.save(done, _payload(done, model, optimizer, losses))
            print(f"[step {done}/{total}] loss={losses[-1]:.6f} (checkpointed)")

    print(f"[done] training complete at step {total}; final loss={losses[-1]:.6f}")
    return losses, EXIT_COMPLETE


def _payload(step: int, model, optimizer, losses: list[float]) -> dict:
    return {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "losses": losses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Session-safe trainer (Phase 0 smoke).")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--set", nargs="*", default=[], help="Dot overrides, e.g. train.lr=0.02")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    _, code = run(cfg)
    return code


if __name__ == "__main__":
    sys.exit(main())
