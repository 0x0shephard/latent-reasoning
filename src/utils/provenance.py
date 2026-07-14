"""Experiment provenance and safe-resume guards.

Research checkpoints must not silently resume under changed data, code, dependencies, or
optimization settings. Runtime controls such as a larger total-step target remain mutable
so a Kaggle run can intentionally continue across sessions.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

_RUNTIME_TOP_LEVEL = {"offline", "output_dir"}
_RUNTIME_TRAIN_KEYS = {
    "ckpt_every",
    "epochs",
    "keep_last",
    "log_every",
    "max_seconds",
    "total_steps",
}
_PACKAGES = ("torch", "transformers", "datasets", "peft", "accelerate", "PyYAML")


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plain_config(cfg) -> dict:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return deepcopy(dict(cfg))


def resume_config(cfg) -> dict:
    """Return the immutable portion of a config used for resume compatibility."""
    result = _plain_config(cfg)
    for key in _RUNTIME_TOP_LEVEL:
        result.pop(key, None)
    train = result.get("train")
    if isinstance(train, dict):
        for key in _RUNTIME_TRAIN_KEYS:
            train.pop(key, None)
    return result


def _data_config(cfg) -> dict | None:
    path_value = cfg.get("data_config")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"data_config does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _source_hash(root: Path) -> str:
    """Hash executable project sources, excluding tests/docs and unrelated configs."""
    paths = sorted((root / "src").rglob("*.py"))
    requirements = root / "requirements.txt"
    if requirements.is_file():
        paths.append(requirements)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = run("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
    }


def build_manifest(cfg, root: str | Path | None = None) -> dict[str, Any]:
    root_path = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    immutable_config = resume_config(cfg)
    data_config = _data_config(cfg)
    environment = {
        "python": platform.python_version(),
        "packages": _package_versions(),
    }
    source_sha256 = _source_hash(root_path)
    fingerprint_payload = {
        "config": immutable_config,
        "data_config": data_config,
        "environment": environment,
        "source_sha256": source_sha256,
    }
    return {
        "fingerprint": _json_hash(fingerprint_payload),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "effective_config": _plain_config(cfg),
        "resume_config": immutable_config,
        "data_config": data_config,
        "environment": environment,
        "source_sha256": source_sha256,
        "git": _git_metadata(root_path),
    }


def prepare_run_manifest(output_dir: str | Path, cfg) -> dict[str, Any]:
    """Create a manifest, or validate that an existing run is resume-compatible."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    current = build_manifest(cfg)
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != current["fingerprint"]:
            raise RuntimeError(
                f"refusing to resume {output}: experiment fingerprint changed; "
                "use a new output_dir for the new experiment"
            )
        stored["latest_effective_config"] = current["effective_config"]
        stored["last_resumed_at_utc"] = current["created_at_utc"]
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        return stored

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return current
