"""Run one isolated control/additional-seed experiment on Kaggle.

The experiment directory is kept beneath ``/kaggle/working`` so Save Version with
outputs enabled persists checkpoints.  A later version can attach that output, restore
the whole experiment directory, and invoke this same command to resume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_control_runner import (
    EXPERIMENTS,
    build_experiment_config,
    experiment_output,
)
from scripts.colab_runner import portable_manifest_preparer


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_session(args) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; enable a Kaggle GPU accelerator")
    name = args.experiment
    spec = EXPERIMENTS[name]
    output_root = Path(args.output_root).expanduser().resolve()
    output_dir = experiment_output(output_root, name)
    status_path = output_root / "status" / "controls_and_seeds" / f"{name}.json"
    cfg = build_experiment_config(
        name,
        output_dir=output_dir,
        max_seconds=args.max_seconds,
        keep_last=args.keep_last,
    )
    status = {
        "experiment": name,
        "base_method": spec.method,
        "seed": spec.seed,
        "platform": "kaggle",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "output_dir": str(output_dir),
        "started_at_utc": _utc_now(),
        "state": "training",
    }
    _atomic_json(status_path, status)

    import src.train.trainer as trainer_module
    from scripts.resume_training import restore_rng_state_portably
    from src.eval.run_eval import evaluate
    from src.train.kaggle_run import EXIT_COMPLETE, run

    trainer_module.load_rng_state = restore_rng_state_portably
    trainer_module.prepare_run_manifest = portable_manifest_preparer(
        args.allow_environment_change
    )
    try:
        _, code = run(cfg)
        if code == EXIT_COMPLETE and args.eval_limit >= 0:
            status.update(state="evaluating", training_exit_code=code, updated_at_utc=_utc_now())
            _atomic_json(status_path, status)
            evaluate(cfg, limit=None if args.eval_limit == 0 else args.eval_limit)
        checkpoints = sorted((output_dir / "checkpoints").glob("step_*.pt"))
        status.update(
            state="complete" if code == EXIT_COMPLETE else "resume_needed",
            training_exit_code=code,
            latest_checkpoint=checkpoints[-1].name if checkpoints else None,
            checkpoint_step=int(checkpoints[-1].stem.split("_")[1]) if checkpoints else None,
            updated_at_utc=_utc_now(),
        )
        _atomic_json(status_path, status)
        return code
    except Exception as exc:
        status.update(
            state="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            updated_at_utc=_utc_now(),
        )
        _atomic_json(status_path, status)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle-safe control/seed training")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=39600,
        help="11h; the trainer stops 5%% early to leave time for Kaggle output persistence",
    )
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--allow-environment-change", action="store_true")
    return run_session(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
