"""Apply the frozen gates to a completed latent-trajectory detect export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_latent_trajectory_detect_analysis import (
    analyze_latent_trajectory_detect,
)
from src.utils.config import load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = load_config(args.config).latent_trajectory_detect
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    report = analyze_latent_trajectory_detect(summary, artifact, settings)
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
