"""Run one Phase-2 training session with a durable Google Drive mirror.

The Colab notebook invokes this script in a *blocking* cell.  The browser can be closed
while the cell continues on the Colab VM, but keeping the cell active avoids turning the
runtime idle.  Completed atomic checkpoints are copied from fast local storage to Drive
by a monitor thread, so a later Colab runtime can resume from the newest durable step.

This wrapper also supports an explicitly audited cross-environment resume.  The original
Kaggle manifests fingerprint package versions, which will normally differ on Colab.  The
portable path still requires identical executable source, data config, and scientific
settings; it only permits environment metadata and the environment-derived HF dataset
fingerprint to differ.  The original checkpoint fingerprint remains authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.provenance import build_manifest, prepare_run_manifest


METHODS = {
    "codi": {
        "config": "configs/codi.yaml",
        "bootstrap_step": 80000,
        "manifest": "artifacts/phase2_resume/codi_run_manifest.json",
    },
    "kava": {
        "config": "configs/kava.yaml",
        "bootstrap_step": 24000,
        "manifest": "artifacts/phase2_resume/kava_run_manifest.json",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".uploading")
    shutil.copy2(source, tmp)
    os.replace(tmp, target)


def validate_torch_checkpoint_archive(path: Path) -> None:
    """Reject extracted directories and ordinary ZIPs that are not torch.save archives."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint must be a file, not a directory: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"checkpoint is not a valid PyTorch ZIP container: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    required_suffixes = ("/data.pkl", "/byteorder", "/version")
    missing = [suffix for suffix in required_suffixes if not any(n.endswith(suffix) for n in names)]
    if missing:
        raise ValueError(f"checkpoint archive is missing {missing}: {path}")


def validate_checkpoint_payload(path: Path, expected_step: int) -> None:
    """Use the active Colab PyTorch build to verify the reconstructed payload."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    actual_step = int(state.get("step", -1))
    fingerprint = state.get("experiment_fingerprint")
    if actual_step != expected_step:
        raise ValueError(
            f"checkpoint payload says step {actual_step}, expected {expected_step}: {path}"
        )
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"checkpoint has no valid experiment fingerprint: {path}")
    del state
    print(f"[bootstrap] PyTorch verified checkpoint step {actual_step}")


def _checkpoint_tree_root(path: Path) -> Path:
    """Resolve either an extracted archive root or a folder containing that root."""
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint folder does not exist: {path}")
    if (path / "data.pkl").is_file():
        root = path
    else:
        children = [child for child in path.iterdir() if child.is_dir()]
        roots = [child for child in children if (child / "data.pkl").is_file()]
        if len(roots) != 1:
            raise ValueError(
                f"expected data.pkl at the checkpoint root or in one child folder: {path}"
            )
        root = roots[0]
    required = (root / "data.pkl", root / "byteorder", root / "version", root / "data")
    missing = [item.name for item in required if not item.exists()]
    if missing:
        raise ValueError(f"extracted checkpoint is missing {missing}: {root}")
    return root


def pack_extracted_checkpoint(source_dir: Path, target: Path) -> Path:
    """Rebuild an uploaded torch.save folder as an uncompressed `.pt` ZIP container."""
    root = _checkpoint_tree_root(source_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".packing")
    tmp.unlink(missing_ok=True)
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".DS_Store" and "__MACOSX" not in path.parts
    ]
    print(f"[bootstrap] rebuilding {target.name} from {len(files)} uploaded files")
    with zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path in files:
            archive.write(path, (Path(target.name) / path.relative_to(root)).as_posix())
    validate_torch_checkpoint_archive(tmp)
    os.replace(tmp, target)
    return target


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = sorted(directory.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _normalized_resume_config(value: dict[str, Any]) -> dict[str, Any]:
    """Remove only the HF cache fingerprint, which can vary with datasets versions."""
    result = deepcopy(value)
    task = result.get("task")
    if isinstance(task, dict):
        task.pop("train_dataset_fingerprint", None)
    return result


def portable_manifest_mismatches(
    stored: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Return scientific incompatibilities; environment changes are audited separately."""
    mismatches = []
    if stored.get("source_sha256") != current.get("source_sha256"):
        mismatches.append("executable source hash")
    if stored.get("data_config") != current.get("data_config"):
        mismatches.append("data config")
    if _normalized_resume_config(stored.get("resume_config", {})) != _normalized_resume_config(
        current.get("resume_config", {})
    ):
        mismatches.append("scientific/resume config")
    return mismatches


def portable_manifest_preparer(allow_environment_change: bool) -> Callable:
    """Create a Trainer manifest hook that preserves the checkpoint's fingerprint."""

    def prepare(output_dir, cfg):
        output = Path(output_dir)
        path = output / "run_manifest.json"
        if not path.is_file():
            return prepare_run_manifest(output, cfg)

        stored = json.loads(path.read_text(encoding="utf-8"))
        current = build_manifest(cfg)
        if stored.get("fingerprint") == current.get("fingerprint"):
            return prepare_run_manifest(output, cfg)
        if not allow_environment_change:
            raise RuntimeError(
                "resume environment differs from the stored manifest; rerun with "
                "--allow-environment-change only for an intentional Kaggle-to-Colab move"
            )

        mismatches = portable_manifest_mismatches(stored, current)
        if mismatches:
            raise RuntimeError(
                "portable resume refused due to changed " + ", ".join(mismatches)
            )

        audit = {
            "accepted_at_utc": _utc_now(),
            "reason": "intentional Kaggle-to-Colab resume",
            "stored_fingerprint": stored.get("fingerprint"),
            "stored_environment": stored.get("environment"),
            "current_environment": current.get("environment"),
            "stored_train_dataset_fingerprint": stored.get("resume_config", {})
            .get("task", {})
            .get("train_dataset_fingerprint"),
            "current_train_dataset_fingerprint": current.get("resume_config", {})
            .get("task", {})
            .get("train_dataset_fingerprint"),
            "verified_equal": [
                "source_sha256",
                "data_config",
                "scientific/resume config (excluding HF cache fingerprint)",
            ],
        }
        _atomic_json(output / "portable_resume_audit.json", audit)
        print("[portable-resume] environment changed; scientific config/source/data verified")
        return stored

    return prepare


class DriveMirror:
    def __init__(
        self,
        local_output: Path,
        drive_output: Path,
        status_path: Path,
        method: str,
        poll_seconds: float = 60.0,
        keep_last: int = 2,
    ):
        self.local_output = local_output
        self.drive_output = drive_output
        self.status_path = status_path
        self.method = method
        self.poll_seconds = poll_seconds
        self.keep_last = keep_last
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_synced: str | None = None
        self.state: dict[str, Any] = {
            "method": method,
            "pid": os.getpid(),
            "started_at_utc": _utc_now(),
            "state": "starting",
            "local_output": str(local_output),
            "drive_output": str(drive_output),
        }

    def update_status(self, state: str, **extra: Any) -> None:
        self.state.update(extra)
        self.state["state"] = state
        self.state["updated_at_utc"] = _utc_now()
        _atomic_json(self.status_path, self.state)

    def restore(self) -> Path:
        self.local_output.mkdir(parents=True, exist_ok=True)
        for name in ("run_manifest.json", "phase2_validation.json", "portable_resume_audit.json"):
            source = self.drive_output / name
            if source.is_file():
                _atomic_copy(source, self.local_output / name)

        drive_checkpoint = _latest_checkpoint(self.drive_output / "checkpoints")
        if drive_checkpoint is None:
            raise FileNotFoundError(
                f"no durable checkpoint under {self.drive_output / 'checkpoints'}"
            )
        validate_torch_checkpoint_archive(drive_checkpoint)
        local_checkpoint = self.local_output / "checkpoints" / drive_checkpoint.name
        if not local_checkpoint.is_file() or local_checkpoint.stat().st_size != drive_checkpoint.stat().st_size:
            print(f"[restore] copying {drive_checkpoint.name} from Drive to local disk")
            _atomic_copy(drive_checkpoint, local_checkpoint)
        validate_torch_checkpoint_archive(local_checkpoint)
        print(f"[restore] ready at step {_checkpoint_step(local_checkpoint)}")
        return local_checkpoint

    def _sync_tree(self, source_root: Path, target_root: Path) -> None:
        if not source_root.is_dir():
            return
        for source in source_root.rglob("*"):
            if source.is_file() and not source.name.endswith((".tmp", ".uploading")):
                _atomic_copy(source, target_root / source.relative_to(source_root))

    def sync(self, force: bool = False) -> Path | None:
        checkpoint = _latest_checkpoint(self.local_output / "checkpoints")
        if checkpoint is None:
            return None
        if force or checkpoint.name != self.last_synced:
            validate_torch_checkpoint_archive(checkpoint)
            target = self.drive_output / "checkpoints" / checkpoint.name
            if not target.is_file() or target.stat().st_size != checkpoint.stat().st_size:
                print(f"[drive-sync] copying atomic checkpoint {checkpoint.name}")
                _atomic_copy(checkpoint, target)
                validate_torch_checkpoint_archive(target)
            self.last_synced = checkpoint.name

            durable = sorted((self.drive_output / "checkpoints").glob("step_*.pt"))
            for old in durable[: -self.keep_last] if self.keep_last > 0 else []:
                old.unlink(missing_ok=True)

        for name in ("run_manifest.json", "phase2_validation.json", "portable_resume_audit.json"):
            source = self.local_output / name
            if source.is_file():
                _atomic_copy(source, self.drive_output / name)
        self._sync_tree(self.local_output / "eval", self.drive_output / "eval")
        self.update_status(
            self.state.get("state", "training"),
            latest_local_checkpoint=checkpoint.name,
            latest_drive_checkpoint=self.last_synced,
            checkpoint_step=_checkpoint_step(checkpoint),
        )
        return checkpoint

    def _loop(self) -> None:
        while not self.stop_event.wait(self.poll_seconds):
            try:
                self.sync()
            except Exception as exc:  # keep training alive; final sync will retry
                print(f"[drive-sync] warning: {type(exc).__name__}: {exc}")
                self.update_status("training", last_sync_error=f"{type(exc).__name__}: {exc}")

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="drive-checkpoint-mirror", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.poll_seconds + 5.0))


def bootstrap_drive(method: str, drive_root: Path, scratch_root: Path) -> Path:
    """Normalize an uploaded checkpoint file or extracted folder into Drive layout."""
    spec = METHODS[method]
    step = int(spec["bootstrap_step"])
    output = drive_root / "outputs" / method
    target = output / "checkpoints" / f"step_{step:08d}.pt"
    if not target.is_file():
        uploads = drive_root / "uploads"
        file_candidates = [
            uploads / f"step_{step:08d}.pt",
            uploads / f"step_{step:08d}.zip",
        ]
        source_file = next(
            (candidate for candidate in file_candidates if candidate.is_file()), None
        )
        folder_candidates = [
            uploads / f"step_{step:08d}.pt",
            uploads / f"step_{step:08d}",
        ]
        source_folder = next(
            (candidate for candidate in folder_candidates if candidate.is_dir()), None
        )
        if source_file is None and source_folder is None:
            raise FileNotFoundError(
                f"upload the extracted step_{step:08d}.pt folder (or original .zip) "
                f"to {uploads} before starting {method}"
            )
        if source_file is not None:
            validate_torch_checkpoint_archive(source_file)
            validate_checkpoint_payload(source_file, step)
            print(f"[bootstrap] installing {source_file.name} as {target.name} on Drive")
            _atomic_copy(source_file, target)
        else:
            scratch_root.mkdir(parents=True, exist_ok=True)
            packed = scratch_root / target.name
            pack_extracted_checkpoint(source_folder, packed)
            validate_checkpoint_payload(packed, step)
            print(f"[bootstrap] persisting rebuilt {target.name} to Drive")
            _atomic_copy(packed, target)
            packed.unlink(missing_ok=True)
    validate_torch_checkpoint_archive(target)

    manifest_target = output / "run_manifest.json"
    if not manifest_target.is_file():
        manifest_source = REPO_ROOT / str(spec["manifest"])
        if not manifest_source.is_file():
            raise FileNotFoundError(f"archived resume manifest is missing: {manifest_source}")
        _atomic_copy(manifest_source, manifest_target)
    return output


def _restore_rng_portably(state: dict) -> None:
    import random

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in state["torch_cuda"]])


def run_session(args) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; choose a Colab GPU runtime")

    method = args.method
    spec = METHODS[method]
    drive_root = Path(args.drive_root).expanduser().resolve()
    if not drive_root.is_dir():
        raise FileNotFoundError(f"Drive root is not mounted: {drive_root}")
    local_root = Path(args.local_root).expanduser().resolve()
    drive_output = bootstrap_drive(method, drive_root, local_root / "bootstrap")
    local_output = local_root / "outputs" / method
    status_path = drive_root / "status" / f"{method}.json"
    mirror = DriveMirror(
        local_output,
        drive_output,
        status_path,
        method,
        poll_seconds=args.sync_every,
        keep_last=args.keep_last,
    )
    mirror.restore()
    mirror.update_status("restored")

    import src.train.trainer as trainer_module
    from src.eval.run_eval import evaluate
    from src.train.kaggle_run import EXIT_COMPLETE, EXIT_RESUME_NEEDED, run

    trainer_module.load_rng_state = _restore_rng_portably
    trainer_module.prepare_run_manifest = portable_manifest_preparer(
        args.allow_environment_change
    )

    overrides = [
        f"output_dir={local_output}",
        f"train.max_seconds={args.max_seconds}",
        f"train.keep_last={args.keep_last}",
    ]
    cfg = load_config(REPO_ROOT / str(spec["config"]), overrides=overrides)
    mirror.update_status(
        "training",
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        max_seconds=args.max_seconds,
    )
    mirror.start()
    try:
        _, code = run(cfg)
        if code == EXIT_COMPLETE and args.eval_limit >= 0:
            mirror.update_status("evaluating", training_exit_code=code)
            limit = None if args.eval_limit == 0 else args.eval_limit
            evaluate(cfg, limit=limit)
        mirror.update_status(
            "complete" if code == EXIT_COMPLETE else "resume_needed",
            training_exit_code=code,
        )
        mirror.sync(force=True)
        return code
    except Exception as exc:
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
    parser = argparse.ArgumentParser(description="Colab-safe Phase-2 training with Drive persistence")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
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
        help="permit an audited Kaggle-to-Colab dependency change",
    )
    args = parser.parse_args()
    return run_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
