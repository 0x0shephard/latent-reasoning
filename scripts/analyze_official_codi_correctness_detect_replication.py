"""Apply the frozen gate to the test-like correctness detector replication."""
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
from src.eval.official_codi_correctness_detect_replication_analysis import (
    DETECT_REPLICATION_CONTRACT,
    analyze_detect_replication,
)
from src.utils.config import load_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_config(args.config).endpoint_correctness_detect_replication
    payload = json.loads(args.sweep.read_text())
    if payload.get("contract") != DETECT_REPLICATION_CONTRACT:
        raise RuntimeError("detect sweep belongs to another contract")
    outcomes_path = args.outcomes or args.sweep.with_suffix(".pt")
    outcomes = torch.load(outcomes_path, map_location="cpu", weights_only=False)
    if outcomes.get("partition_sha256") != payload["splits"]["partition_sha256"]:
        raise RuntimeError("detect summary and paired outcomes use different splits")
    report = analyze_detect_replication(payload, outcomes, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, args.output)

    verdict = "PASS" if report["passed"] else "FAIL"
    print("=" * 72)
    print("CODI test-like correctness detector replication")
    print("=" * 72)
    print(
        f"splits fit={report['splits']['fit']} select={report['splits']['select']} "
        f"test={report['splits']['test']}"
    )
    print(
        f"[{verdict}] {report['primary_probe']} AUC {report['primary_auc']:.4f} "
        f"vs {report['baseline_probe']} {report['baseline_auc']:.4f}; "
        f"delta {report['delta_auc']:+.4f} "
        f"[{report['delta_ci'][0]:+.4f}, {report['delta_ci'][1]:+.4f}]"
    )
    print(f"optimizer certificate valid: {report['optimizer_valid']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
