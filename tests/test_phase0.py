"""Phase 0 gate: the session-safe trainer is deterministic and resumes with no
metric discontinuity.

These tests use only the synthetic task, so they run in seconds on CPU with no model
or dataset downloads.
"""
from __future__ import annotations

import copy

from src.train.kaggle_run import EXIT_COMPLETE, run
from src.utils.checkpoint import clear_checkpoints
from src.utils.config import Config


def _base_cfg(tmp_path, total_steps: int = 20) -> Config:
    return Config(
        {
            "run_name": "test",
            "seed": 0,
            "offline": True,
            "output_dir": str(tmp_path),
            "train": {
                "total_steps": total_steps,
                "batch_size": 16,
                "lr": 0.01,
                "ckpt_every": 5,
                "keep_last": 3,
                "max_seconds": 3600,
            },
            "dummy": {"in_dim": 8, "noise_std": 0.1},
        }
    )


def test_determinism(tmp_path):
    """Two fresh runs with the same seed produce identical loss trajectories."""
    cfg_a = _base_cfg(tmp_path / "a")
    cfg_b = _base_cfg(tmp_path / "b")
    losses_a, code_a = run(cfg_a)
    losses_b, code_b = run(cfg_b)
    assert code_a == EXIT_COMPLETE and code_b == EXIT_COMPLETE
    assert losses_a == losses_b


def test_resume_has_no_discontinuity(tmp_path):
    """A run interrupted at step 10 and resumed matches a straight 20-step run exactly."""
    # Reference: uninterrupted 20 steps.
    ref_cfg = _base_cfg(tmp_path / "ref", total_steps=20)
    ref_losses, _ = run(ref_cfg)

    # Interrupted: train 10, then "resume" to 20 from the checkpoint.
    out = tmp_path / "resumed"
    first = _base_cfg(out, total_steps=10)
    run(first)
    second = copy.deepcopy(_base_cfg(out, total_steps=20))
    resumed_losses, code = run(second)  # picks up the checkpoint under the same output_dir

    assert code == EXIT_COMPLETE
    assert len(resumed_losses) == 20
    # Bit-for-bit continuity across the resume boundary.
    assert resumed_losses == ref_losses

    clear_checkpoints(out)


def test_task_actually_learns(tmp_path):
    """Sanity: the synthetic regression loss decreases (loop is wired correctly)."""
    cfg = _base_cfg(tmp_path / "learn", total_steps=50)
    losses, _ = run(cfg)
    assert losses[-1] < losses[0]
