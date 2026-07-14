"""Generic session-safe training loop, reused by every method.

Given a `(model, optimizer, step_fn)` it owns: resume-from-latest-checkpoint, checkpoint
cadence, the wall-clock guard, logging, and the exit-code contract. Task builders (dummy
in Phase 0, SFT in Phase 1b, CODI/KaVa in Phase 2) provide the model + a `step_fn(step) ->
loss` and never re-implement this loop.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.utils.checkpoint import Checkpointer
from src.utils.provenance import prepare_run_manifest
from src.utils.seeding import load_rng_state, rng_state
from src.utils.time_budget import TimeBudget

EXIT_COMPLETE = 0
EXIT_RESUME_NEEDED = 42


class Trainer:
    def __init__(self, cfg, model, optimizer, step_fn):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.step_fn = step_fn
        self.device = next(model.parameters()).device
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = prepare_run_manifest(self.output_dir, cfg)
        self.experiment_fingerprint = self.manifest["fingerprint"]
        self.ckpt = Checkpointer(self.output_dir, keep_last=cfg.train.get("keep_last", 3))
        self.budget = TimeBudget(cfg.train.max_seconds)

    def _payload(self, step: int, losses: list[float]) -> dict:
        return {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_state(),
            "losses": losses,
            "experiment_fingerprint": self.experiment_fingerprint,
        }

    def fit(self) -> tuple[list[float], int]:
        start_step = 0
        losses: list[float] = []
        resumed = self.ckpt.load_latest(map_location=str(self.device))
        if resumed is not None:
            stored_fingerprint = resumed.get("experiment_fingerprint")
            if stored_fingerprint is None:
                raise RuntimeError(
                    "legacy checkpoint has no experiment fingerprint; start from a new "
                    "output_dir to avoid mixing unverifiable state"
                )
            if stored_fingerprint != self.experiment_fingerprint:
                raise RuntimeError(
                    "checkpoint fingerprint does not match this experiment; use a new output_dir"
                )
            self.model.load_state_dict(resumed["model"])
            self.optimizer.load_state_dict(resumed["optimizer"])
            load_rng_state(resumed["rng"])
            start_step = resumed["step"]
            losses = resumed.get("losses", [])
            print(f"[resume] continuing from step {start_step}")

        total = self.cfg.train.total_steps
        ckpt_every = self.cfg.train.ckpt_every
        log_every = self.cfg.train.get("log_every", ckpt_every)

        for step in range(start_step, total):
            if self.budget.should_stop():
                self.ckpt.save(step, self._payload(step, losses))
                print(f"[budget] wall-clock guard hit at step {step}; state saved. Re-run to resume.")
                return losses, EXIT_RESUME_NEEDED

            losses.append(self.step_fn(step))
            done = step + 1
            if done % log_every == 0:
                print(f"[step {done}/{total}] loss={losses[-1]:.6f}")
            if done % ckpt_every == 0 or done == total:
                self.ckpt.save(done, self._payload(done, losses))

        print(f"[done] training complete at step {total}; final loss={losses[-1]:.6f}")
        return losses, EXIT_COMPLETE
