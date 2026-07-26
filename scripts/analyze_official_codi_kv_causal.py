"""Analyze full-GSM8K official CODI causal KV-subspace interventions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.official_codi_kv_causal_analysis import (
    analyze_causal_interventions,
    render_causal_markdown,
)


CAUSAL_EVALUATION_SCHEMA_VERSION = 1


def _parse_positions(value: str) -> list[int]:
    try:
        positions = sorted(
            {int(item.strip()) for item in value.split(",") if item.strip()}
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "positions must be comma-separated integers"
        ) from exc
    if not positions:
        raise argparse.ArgumentTypeError("at least one position is required")
    return positions


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run paired causal analysis for official CODI KV interventions."
    )
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--primary-positions",
        type=_parse_positions,
        default=[4, 5],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    args = parser.parse_args()

    manifest_path = args.evaluation_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"evaluation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CAUSAL_EVALUATION_SCHEMA_VERSION:
        raise RuntimeError("causal evaluation manifest has an incompatible schema")
    if manifest.get("state") != "complete":
        raise RuntimeError(
            "causal evaluation is incomplete; rerun the evaluation cell to resume"
        )
    if set(manifest.get("completed_conditions", [])) != set(
        manifest.get("conditions", [])
    ):
        raise RuntimeError("causal evaluation condition set is incomplete")
    if not manifest.get("full_gsm8k"):
        raise RuntimeError("primary causal analysis requires full GSM8K")
    if int(manifest.get("evaluated_count", -1)) != 1319:
        raise RuntimeError("primary causal analysis requires all 1,319 GSM8K examples")
    baseline_summary_path = args.evaluation_root / "baseline" / "summary.json"
    if not baseline_summary_path.is_file():
        raise FileNotFoundError("causal baseline summary is missing")
    baseline_summary = json.loads(
        baseline_summary_path.read_text(encoding="utf-8")
    )
    if baseline_summary.get("accuracy_gate", {}).get("status") != "passed":
        raise RuntimeError("causal baseline did not pass the official accuracy gate")

    scopes = [f"p{position}" for position in manifest["positions"]]
    if manifest.get("include_all"):
        scopes.append("all")
    report = analyze_causal_interventions(
        args.evaluation_root,
        scopes=scopes,
        primary_positions=args.primary_positions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        familywise_alpha=args.familywise_alpha,
    )
    report["evaluation_manifest"] = manifest
    _atomic_write(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    markdown_path = args.output.with_suffix(".md")
    _atomic_write(markdown_path, render_causal_markdown(report))
    print(f"[analysis] gate={report['gate']['status']}")
    print(f"[analysis] wrote {args.output}")
    print(f"[analysis] wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
