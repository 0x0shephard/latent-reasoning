"""Evaluate completed Stage 1d checkpoints without invoking the trainer.

This path intentionally avoids manifest preparation and resume checks. It is safe after
an analysis-only source change because evaluation reconstructs the model from each
checkpoint's stored effective config and never updates model or optimizer state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_key_projection_runner import EXPERIMENTS, experiment_output
from src.utils.config import Config


def config_from_manifest(output_dir: Path, *, batch_size: int) -> Config:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    effective = manifest.get("effective_config")
    if not isinstance(effective, dict):
        raise ValueError(f"manifest has no effective config: {manifest_path}")
    cfg = Config(effective)
    cfg["output_dir"] = str(output_dir)
    data_config = Path(cfg["data_config"])
    if not data_config.is_absolute():
        cfg["data_config"] = str(REPO_ROOT / data_config)
    cfg.setdefault("eval", {})["batch_size"] = batch_size
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluation-only runner for completed Stage 1d checkpoints"
    )
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/CODI_KAVA")
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(EXPERIMENTS),
        help="repeat to select arms; omit to evaluate all four",
    )
    parser.add_argument(
        "--include-original-codi",
        action="store_true",
        help="also evaluate the unadapted CODI seed-one checkpoint",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all examples; positive values cap each dataset",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")

    from src.eval.run_eval import evaluate

    drive_root = Path(args.drive_root).expanduser().resolve()
    names = args.experiment or list(EXPERIMENTS)
    runs = [
        (name, experiment_output(drive_root, name))
        for name in names
    ]
    if args.include_original_codi:
        runs.insert(
            0,
            (
                "codi_seed1_original",
                drive_root / "outputs" / "controls_and_seeds" / "codi_seed1",
            ),
        )
    for name, output_dir in runs:
        cfg = config_from_manifest(output_dir, batch_size=args.batch_size)
        checkpoints = sorted((output_dir / "checkpoints").glob("step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"missing checkpoint for {name}: {output_dir}")
        print(f"\n[stage1d-eval] {name} from {checkpoints[-1].name}")
        evaluate(cfg, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
