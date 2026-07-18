"""Restore a completed Phase-2 checkpoint, run ablations, and persist them to Drive.

Unlike ``colab_runner.py``, this entrypoint never enters the training loop. It copies the
newest durable checkpoint to fast VM-local storage, evaluates interventions there, and
atomically mirrors the small prediction artifacts back to Google Drive.
"""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_runner import METHODS, DriveMirror, bootstrap_drive
from src.utils.config import load_config


def run_session(args) -> int:
    import torch

    from src.eval.run_eval import evaluate
    from src.mech.ablation import ABLATION_MODES, LatentAblation, parse_positions

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable; choose a Colab GPU runtime")

    spec = METHODS[args.method]
    drive_root = Path(args.drive_root).expanduser().resolve()
    if not drive_root.is_dir():
        raise FileNotFoundError(f"Drive root is not mounted: {drive_root}")
    local_root = Path(args.local_root).expanduser().resolve()
    drive_output = bootstrap_drive(args.method, drive_root, local_root / "bootstrap")
    local_output = local_root / "outputs" / args.method
    mirror = DriveMirror(
        local_output,
        drive_output,
        drive_root / "status" / f"{args.method}.json",
        args.method,
        keep_last=args.keep_last,
    )
    mirror.restore()

    cfg = load_config(
        REPO_ROOT / str(spec["config"]), overrides=[f"output_dir={local_output}"]
    )
    positions = parse_positions(args.positions, int(cfg.task.latent_steps))
    position_tag = "all" if positions is None else "p" + "-".join(map(str, sorted(positions)))
    modes = args.mode or list(ABLATION_MODES)
    completed = []
    mirror.update_status(
        "ablating",
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        ablation_modes=modes,
        ablation_positions=args.positions,
        eval_limit=args.limit,
    )
    mirror.start()
    try:
        for mode in modes:
            tag = None if mode == "baseline" else f"{mode}_{position_tag}"
            print(f"[ablation] method={args.method} mode={mode} positions={args.positions}")
            evaluate(
                cfg,
                limit=None if args.limit == 0 else args.limit,
                intervention=(
                    None
                    if mode == "baseline"
                    else LatentAblation(mode, positions, args.seed)
                ),
                eval_tag=tag,
            )
            completed.append(tag or "baseline")
            mirror.update_status("ablating", completed_ablations=completed)
            mirror.sync(force=True)
            gc.collect()
            torch.cuda.empty_cache()
        mirror.close()
        mirror.update_status("analysis_complete", completed_ablations=completed)
        mirror.sync(force=True)
        return 0
    except Exception as exc:
        mirror.close()
        mirror.update_status(
            "analysis_failed",
            completed_ablations=completed,
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
        description="Colab-safe causal ablation evaluation with Drive persistence."
    )
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/CODI_KAVA")
    parser.add_argument("--local-root", default="/content/codikava_runtime")
    parser.add_argument(
        "--mode",
        action="append",
        choices=("baseline", "zero", "batch_mean", "batch_shuffle"),
        help="Repeat to select modes; default runs the three interventions.",
    )
    parser.add_argument("--positions", default="all")
    parser.add_argument(
        "--limit", type=int, default=200, help="Examples per set; 0 means full evaluation."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-last", type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    return run_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
