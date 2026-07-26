"""Run one Stage 1d warm-started key-supervision arm with durable Drive resume."""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
class KeyExperiment:
    method: str
    kv_weight: float
    kv_target: str
    projection_kind: str | None = None


EXPERIMENTS = {
    "codi_continue_seed1": KeyExperiment(
        "codi_key_control", 0.0, "key"
    ),
    "key_full_seed1": KeyExperiment(
        "kava_key_full", 1.0, "key"
    ),
    "key_rank4_seed1": KeyExperiment(
        "kava_key_rank4", 1.0, "projected_key", "learned"
    ),
    "key_random_rank4_seed1": KeyExperiment(
        "kava_key_random_rank4", 1.0, "projected_key", "random"
    ),
}


def experiment_output(root: Path, name: str) -> Path:
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}")
    return root / "outputs" / "key_projection" / name


def _atomic_copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".copying")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    return target


def _prepare_inputs(
    drive_root: Path,
    local_root: Path,
    warm_start: Path | None,
    projection: Path | None,
) -> tuple[Path, Path]:
    import torch

    warm_source = warm_start or (
        drive_root
        / "outputs"
        / "controls_and_seeds"
        / "codi_seed1"
        / "checkpoints"
        / "step_00096405.pt"
    )
    projection_source = projection or (
        drive_root / "artifacts" / "stage1d_key_rank4_projectors.pt"
    )
    if not warm_source.is_file():
        raise FileNotFoundError(f"missing CODI seed-one checkpoint: {warm_source}")
    if not projection_source.is_file():
        raise FileNotFoundError(f"missing key projection artifact: {projection_source}")

    bootstrap = local_root / "bootstrap" / "stage1d"
    local_warm = bootstrap / "codi_seed1_step_00096405.pt"
    local_projection = bootstrap / "key_rank4_projectors.pt"
    if not local_warm.is_file() or local_warm.stat().st_size != warm_source.stat().st_size:
        print(f"[bootstrap] copying {warm_source.name} to local disk")
        _atomic_copy(warm_source, local_warm)
    if (
        not local_projection.is_file()
        or local_projection.stat().st_size != projection_source.stat().st_size
    ):
        print(f"[bootstrap] copying {projection_source.name} to local disk")
        _atomic_copy(projection_source, local_projection)

    checkpoint = torch.load(
        local_warm,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if int(checkpoint.get("step", -1)) != 96405:
        raise ValueError("warm-start checkpoint must be completed CODI seed one")
    manifest_path = warm_source.parent.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing warm-start run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    effective = manifest.get("effective_config", {})
    if (
        effective.get("seed") != 1
        or effective.get("task", {}).get("method") != "codi"
        or checkpoint.get("experiment_fingerprint") != manifest.get("fingerprint")
    ):
        raise ValueError("warm start is not the verified completed CODI seed-one run")
    del checkpoint
    artifact = torch.load(
        local_projection,
        map_location="cpu",
        weights_only=False,
    )
    if (
        int(artifact.get("schema_version", -1)) != 1
        or artifact.get("kind") != "key"
        or int(artifact.get("rank", -1)) != 4
        or int(artifact.get("processed_examples", -1)) != 5000
    ):
        raise ValueError("projection artifact violates the Stage 1d contract")
    print("[bootstrap] verified CODI seed-one warm start and rank-four key bases")
    return local_warm, local_projection


def build_experiment_config(
    name: str,
    *,
    output_dir: Path,
    warm_start: Path,
    projection: Path,
    max_seconds: float,
    keep_last: int,
):
    spec = EXPERIMENTS[name]
    cfg = load_config(REPO_ROOT / "configs/key_projection_warmstart.yaml")
    cfg["run_name"] = name
    cfg["output_dir"] = str(output_dir)
    cfg["task"]["method"] = spec.method
    cfg["task"]["warm_start_checkpoint"] = str(warm_start)
    distill = cfg["task"]["distillation"]
    distill["kv_weight"] = spec.kv_weight
    distill["kv_target"] = spec.kv_target
    if spec.projection_kind is not None:
        distill["key_projection_path"] = str(projection)
        distill["key_projection_kind"] = spec.projection_kind
        distill["key_projection_rank"] = 4
    cfg["train"]["max_seconds"] = max_seconds
    cfg["train"]["keep_last"] = keep_last
    return cfg


def run_session(args) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; choose a Colab GPU runtime")
    name = args.experiment
    spec = EXPERIMENTS[name]
    drive_root = Path(args.drive_root).expanduser().resolve()
    local_root = Path(args.local_root).expanduser().resolve()
    if not drive_root.is_dir():
        raise FileNotFoundError(f"Drive root is not mounted: {drive_root}")
    warm_start, projection = _prepare_inputs(
        drive_root,
        local_root,
        args.warm_start,
        args.projection,
    )
    drive_output = experiment_output(drive_root, name)
    local_output = experiment_output(local_root, name)
    status_path = (
        drive_root / "status" / "key_projection" / f"{name}.json"
    )
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
        seed=1,
        adaptation_steps=10000,
        warm_start_step=96405,
        kv_target=spec.kv_target,
        projection_kind=spec.projection_kind,
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
        warm_start=warm_start,
        projection=projection,
        max_seconds=args.max_seconds,
        keep_last=args.keep_last,
    )
    mirror.update_status(
        "training",
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        eval_limit=args.eval_limit,
    )
    mirror.start()
    try:
        _, code = run(cfg)
        if code == EXIT_COMPLETE and args.eval_limit >= 0:
            mirror.update_status("evaluating", training_exit_code=code)
            # ``evaluate`` treats an explicit zero as "no cap". Passing None
            # would fall back to cfg.eval.limit (200 in the Stage 1d config).
            evaluate(cfg, limit=args.eval_limit)
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
            print(
                f"[drive-sync] final failure: "
                f"{type(sync_exc).__name__}: {sync_exc}"
            )
        raise
    finally:
        mirror.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1d warm-started key projection experiment"
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/CODI_KAVA")
    parser.add_argument("--local-root", default="/content/codikava_runtime")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--max-seconds", type=float, default=32400)
    parser.add_argument("--sync-every", type=float, default=60)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--allow-environment-change", action="store_true")
    return run_session(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
