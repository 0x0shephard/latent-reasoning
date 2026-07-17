"""Resume a GPU checkpoint whose CPU RNG tensor was mapped onto CUDA.

Older Trainer checkpoints were loaded with ``map_location='cuda'``. That is correct for
model/optimizer tensors, but it also moved the saved CPU RNG byte tensor to CUDA, while
``torch.set_rng_state`` requires a CPU tensor. This wrapper patches only RNG restoration
at runtime. It lives outside the provenance-hashed ``src/`` tree so an existing experiment
can resume under its original fingerprint.

Usage:
    python -u scripts/resume_training.py --config configs/codi.yaml
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.train.trainer as trainer_module
from src.train.kaggle_run import run
from src.utils.config import load_config


def restore_rng_state_portably(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        saved_states = state["torch_cuda"]
        active_devices = torch.cuda.device_count()
        restore_count = min(len(saved_states), active_devices)
        for device_index in range(restore_count):
            torch.cuda.set_rng_state(
                saved_states[device_index].detach().cpu(), device=device_index
            )
        if len(saved_states) != active_devices:
            print(
                "[resume] CUDA RNG topology changed: "
                f"saved={len(saved_states)} active={active_devices}; "
                f"restored devices 0..{restore_count - 1}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume training with portable RNG restore.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="Config dot-overrides")
    args = parser.parse_args()
    trainer_module.load_rng_state = restore_rng_state_portably
    cfg = load_config(args.config, overrides=args.set)
    _, code = run(cfg)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
