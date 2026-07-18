"""Evaluate causal latent-state ablations against a completed checkpoint.

This intentionally runs after training and never modifies the checkpoint. Results are
written below ``eval/step_*/ablations/<mode>_<positions>/``.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.eval.run_eval import evaluate
from src.mech.ablation import ABLATION_MODES, LatentAblation, parse_positions
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run causal latent-state ablations.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        action="append",
        choices=ABLATION_MODES,
        help="Repeat to run multiple modes; default runs all three.",
    )
    parser.add_argument(
        "--positions",
        default="all",
        help="'all' or zero-based comma-separated latent positions.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], help="Config dot-overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    if cfg.task.get("type") not in {"latent", "codi", "kava"}:
        parser.error("the selected config is not a latent model")
    positions = parse_positions(args.positions, int(cfg.task.latent_steps))
    position_tag = "all" if positions is None else "p" + "-".join(map(str, sorted(positions)))
    modes = args.mode or list(ABLATION_MODES)

    for mode in modes:
        intervention = LatentAblation(mode=mode, positions=positions, seed=args.seed)
        tag = f"{mode}_{position_tag}"
        print(f"[ablation] mode={mode} positions={args.positions} tag={tag}")
        evaluate(
            cfg,
            limit=args.limit,
            intervention=intervention,
            eval_tag=tag,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
