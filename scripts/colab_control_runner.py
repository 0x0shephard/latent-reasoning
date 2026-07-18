"""Train Phase-2 controls and additional seeds with durable Colab/Drive resume.

Unlike ``colab_runner.py``, these experiments intentionally start from the pinned GPT-2
backbone rather than an uploaded partial checkpoint.  Every experiment has an isolated
Drive directory, so it cannot restore from or overwrite the completed seed-zero CODI and
KaVa runs.  Re-running the same experiment resumes its newest durable checkpoint.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_runner import DriveMirror, portable_manifest_preparer
from src.utils.config import load_config


@dataclass(frozen=True)
class Experiment:
    config: str
    method: str
    seed: int


EXPERIMENTS = {
    "latent_nodistill_seed0": Experiment(
        "configs/latent_nodistill.yaml", "latent_nodistill", 0
    ),
    "kava_random_seed0": Experiment("configs/kava_random.yaml", "kava_random", 0),
    "kava_uniform_seed0": Experiment("configs/kava_uniform.yaml", "kava_uniform", 0),
    "codi_seed1": Experiment("configs/codi.yaml", "codi", 1),
    "kava_seed1": Experiment("configs/kava.yaml", "kava", 1),
    "codi_seed2": Experiment("configs/codi.yaml", "codi", 2),
    "kava_seed2": Experiment("configs/kava.yaml", "kava", 2),
}


def experiment_output(root: Path, name: str) -> Path:
    """Return the isolated output directory for a named experiment."""
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}")
    return root / "outputs" / "controls_and_seeds" / name


def build_experiment_config(
    name: str,
    *,
    output_dir: Path,
    max_seconds: float,
    keep_last: int,
):
    """Load a checked-in base config with only identity/session overrides."""
    spec = EXPERIMENTS[name]
    return load_config(
        REPO_ROOT / spec.config,
        overrides=[
            f"seed={spec.seed}",
            f"run_name={name}",
            f"output_dir={output_dir}",
            f"train.max_seconds={max_seconds}",
            f"train.keep_last={keep_last}",
            "eval.batch_size=8",
        ],
    )


def run_session(args) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; choose a Colab GPU runtime")

    name = args.experiment
    spec = EXPERIMENTS[name]
    drive_root = Path(args.drive_root).expanduser().resolve()
    if not drive_root.is_dir():
        raise FileNotFoundError(f"Drive root is not mounted: {drive_root}")
    local_root = Path(args.local_root).expanduser().resolve()
    drive_output = experiment_output(drive_root, name)
    local_output = experiment_output(local_root, name)
    status_path = drive_root / "status" / "controls_and_seeds" / f"{name}.json"
    drive_output.mkdir(parents=True, exist_ok=True)

    mirror = DriveMirror(
        local_output,
        drive_output,
        status_path,
        name,
        poll_seconds=args.sync_every,
        keep_last=args.keep_last,
    )
    restored = mirror.restore(require_checkpoint=False)
    mirror.update_status(
        "restored" if restored is not None else "fresh_start",
        experiment=name,
        base_method=spec.method,
        seed=spec.seed,
    )

    import src.train.trainer as trainer_module
    from scripts.resume_training import restore_rng_state_portably
    from src.eval.run_eval import evaluate
    from src.train.kaggle_run import EXIT_COMPLETE, run

    trainer_module.load_rng_state = restore_rng_state_portably
    trainer_module.prepare_run_manifest = portable_manifest_preparer(
        args.allow_environment_change
    )
    cfg = build_experiment_config(
        name,
        output_dir=local_output,
        max_seconds=args.max_seconds,
        keep_last=args.keep_last,
    )

    mirror.update_status(
        "training",
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        max_seconds=args.max_seconds,
        eval_limit=args.eval_limit,
    )
    mirror.start()
    try:
        _, code = run(cfg)
        if code == EXIT_COMPLETE and args.eval_limit >= 0:
            mirror.update_status("evaluating", training_exit_code=code)
            evaluate(cfg, limit=None if args.eval_limit == 0 else args.eval_limit)
        mirror.close()
        mirror.update_status(
            "complete" if code == EXIT_COMPLETE else "resume_needed",
            training_exit_code=code,
        )
        mirror.sync(force=True)
        return code
    except Exception as exc:
        mirror.close()
        mirror.update_status(
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        try:
            mirror.sync(force=True)
        except Exception as sync_exc:
            print(f"[drive-sync] final failure: {type(sync_exc).__name__}: {sync_exc}")
        raise
    finally:
        mirror.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Colab-safe training for controls and additional seeds"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/CODI_KAVA")
    parser.add_argument("--local-root", default="/content/codikava_runtime")
    parser.add_argument("--max-seconds", type=float, default=32400)
    parser.add_argument("--sync-every", type=float, default=60)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=200,
        help="200=quick gate, 0=full evaluation, negative=skip evaluation",
    )
    parser.add_argument(
        "--allow-environment-change",
        action="store_true",
        help="permit an audited dependency change when resuming on a later runtime",
    )
    return run_session(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
