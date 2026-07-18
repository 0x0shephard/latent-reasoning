"""Verify and atomically import a Kaggle control/seed output into Google Drive."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_control_runner import EXPERIMENTS, experiment_output
from scripts.colab_runner import (
    validate_checkpoint_payload,
    validate_torch_checkpoint_archive,
)


def _resolve_experiment_source(source: Path, experiment: str) -> Path:
    source = source.expanduser().resolve()
    if (source / "run_manifest.json").is_file() and source.name == experiment:
        return source
    candidates = [
        path.parent
        for path in source.rglob("run_manifest.json")
        if path.parent.name == experiment
        and path.parent.parent.name == "controls_and_seeds"
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one controls_and_seeds/{experiment} under {source}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _validate_identity(source: Path, experiment: str) -> dict:
    spec = EXPERIMENTS[experiment]
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    resume = manifest.get("resume_config", {})
    task = resume.get("task", {})
    if resume.get("seed") != spec.seed:
        raise ValueError(
            f"seed mismatch for {experiment}: {resume.get('seed')} != {spec.seed}"
        )
    if task.get("method") != spec.method:
        raise ValueError(
            f"method mismatch for {experiment}: {task.get('method')} != {spec.method}"
        )
    if resume.get("run_name") != experiment:
        raise ValueError(
            f"run_name mismatch: {resume.get('run_name')!r} != {experiment!r}"
        )
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Kaggle run manifest has no valid fingerprint")
    return manifest


def _latest_verified_checkpoint(source: Path) -> tuple[Path, int]:
    checkpoints = sorted((source / "checkpoints").glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {source}")
    checkpoint = checkpoints[-1]
    step = int(checkpoint.stem.split("_")[1])
    validate_torch_checkpoint_archive(checkpoint)
    validate_checkpoint_payload(checkpoint, step)
    return checkpoint, step


def import_experiment(source: Path, drive_root: Path, experiment: str) -> dict:
    source = _resolve_experiment_source(source, experiment)
    manifest = _validate_identity(source, experiment)
    checkpoint, step = _latest_verified_checkpoint(source)
    drive_root = drive_root.expanduser().resolve()
    if not drive_root.is_dir():
        raise FileNotFoundError(f"Drive root is not mounted: {drive_root}")
    target = experiment_output(drive_root, experiment)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"refusing to merge into non-empty {target}; never train the same experiment "
            "on Kaggle and Colab"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".importing")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    _validate_identity(temporary, experiment)
    copied_checkpoint, copied_step = _latest_verified_checkpoint(temporary)
    if copied_step != step or copied_checkpoint.stat().st_size != checkpoint.stat().st_size:
        raise RuntimeError("copied checkpoint does not match the verified Kaggle source")
    if target.exists():
        target.rmdir()
    os.replace(temporary, target)

    expected_step = int(
        manifest.get("effective_config", {}).get("train", {}).get("total_steps", -1)
    )
    eval_summary = target / "eval" / f"step_{step:08d}" / "summary.json"
    complete = step == expected_step and eval_summary.is_file()
    status = {
        "experiment": experiment,
        "base_method": EXPERIMENTS[experiment].method,
        "seed": EXPERIMENTS[experiment].seed,
        "platform": "kaggle_import",
        "state": "complete" if complete else "resume_needed",
        "checkpoint_step": step,
        "latest_drive_checkpoint": checkpoint.name,
        "drive_output": str(target),
        "manifest_fingerprint": manifest["fingerprint"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    status_path = (
        drive_root / "status" / "controls_and_seeds" / f"{experiment}.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, status_path)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Import verified Kaggle output to Drive")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    args = parser.parse_args()
    status = import_experiment(args.source, args.drive_root, args.experiment)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
