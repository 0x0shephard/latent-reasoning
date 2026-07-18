"""Statically validate the control and additional-seed experiment matrix.

This check is intentionally download-free.  It verifies that the control YAMLs preserve
the primary architecture/training contract and that the Colab runner changes only seed
and experiment identity for the extra CODI/KaVa runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_control_runner import EXPERIMENTS
from scripts.validate_phase2 import _controlled_differences
from src.utils.config import load_config


def validate_matrix() -> dict:
    codi = load_config(REPO_ROOT / "configs/codi.yaml")
    kava = load_config(REPO_ROOT / "configs/kava.yaml")
    nodistill = load_config(REPO_ROOT / "configs/latent_nodistill.yaml")
    random_cfg = load_config(REPO_ROOT / "configs/kava_random.yaml")
    uniform_cfg = load_config(REPO_ROOT / "configs/kava_uniform.yaml")

    failures: list[str] = []
    nodistill_differences = _controlled_differences(nodistill, codi)
    if {item["field"] for item in nodistill_differences} != {
        "task.distillation.hidden_weight"
    }:
        failures.append(
            "latent_nodistill must differ from CODI only in controlled hidden_weight"
        )
    if float(nodistill.task.distillation.hidden_weight) != 0.0:
        failures.append("latent_nodistill hidden_weight must be zero")
    if float(nodistill.task.distillation.kv_weight) != 0.0:
        failures.append("latent_nodistill kv_weight must be zero")

    compression_controls = {
        "kava_random": (random_cfg, "random"),
        "kava_uniform": (uniform_cfg, "uniform"),
    }
    compression_differences = {}
    for name, (cfg, expected_compression) in compression_controls.items():
        differences = _controlled_differences(cfg, kava)
        compression_differences[name] = differences
        if differences:
            failures.append(f"{name} changes controlled architecture/training fields")
        actual = cfg.task.distillation.get("compression")
        if actual != expected_compression:
            failures.append(
                f"{name} compression is {actual!r}, expected {expected_compression!r}"
            )
        if float(cfg.task.distillation.get("kv_weight", 0.0)) <= 0:
            failures.append(f"{name} must retain positive KV supervision")

    expected_experiments = {
        "latent_nodistill_seed0": ("latent_nodistill", 0),
        "kava_random_seed0": ("kava_random", 0),
        "kava_uniform_seed0": ("kava_uniform", 0),
        "codi_seed1": ("codi", 1),
        "kava_seed1": ("kava", 1),
        "codi_seed2": ("codi", 2),
        "kava_seed2": ("kava", 2),
    }
    actual_experiments = {
        name: (spec.method, spec.seed) for name, spec in EXPERIMENTS.items()
    }
    if actual_experiments != expected_experiments:
        failures.append("Colab experiment matrix does not match the locked control/seed plan")

    return {
        "status": "failed" if failures else "ok",
        "experiments": actual_experiments,
        "latent_nodistill_controlled_differences": nodistill_differences,
        "compression_controlled_differences": compression_differences,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate control and seed experiment matrix")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_matrix()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"[validate] wrote {args.output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
