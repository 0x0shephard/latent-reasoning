"""Confirm the latent-workspace reading of CODI's thoughts, on frozen gates.

CPU-only and deterministic: every quantity is arithmetic on the completed §52
trajectory export, the frozen readout, and the pinned GSM8K test solutions. The
frozen partition's fit/select rows set no thresholds here — every threshold was
frozen in configuration from the §53 fit/select observations — and the final 439
rows are read once.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from scripts.run_official_codi_latent_trajectory_detect import load_trajectory_export
from src.eval.official_codi_latent_workspace_analysis import analyze_latent_workspace
from src.mech.endpoint_correctness_geometry import readout_matrix
from src.mech.latent_workspace import (
    LATENT_WORKSPACE_CONTRACT,
    LATENT_WORKSPACE_SCHEMA_VERSION,
    WORKSPACE_STATE,
    WORKSPACE_TOP_K,
    alignment_table,
    decode_thought_numbers,
    parse_solution,
    per_thought_hits,
    recovery_fraction,
    seeded_derangement,
)
from src.utils.config import load_config


def _normalize_question(value: str) -> str:
    return " ".join(str(value).split())


def load_solutions(path: Path, *, expected_sha256: str, questions: list[str]) -> list[dict]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"solutions file hash {digest} does not match the pinned {expected_sha256}"
        )
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    by_question = {_normalize_question(row["question"]): row["answer"] for row in rows}
    solutions = []
    for question in questions:
        key = _normalize_question(question)
        if key not in by_question:
            raise RuntimeError("a cached question is missing from the pinned solutions")
        solutions.append(parse_solution(by_question[key]))
    return solutions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    settings = cfg.latent_workspace
    cache, readout_payload = load_margin_cache(args.states, args.readout)
    readout = readout_matrix(readout_payload)
    export = load_trajectory_export(args.trajectory, cache)
    trajectory = export["trajectory_states"]
    endpoint = export["endpoint_states"].double()
    live_first = export["live_first_token"].long()
    gold_first = cache["evaluation_gold_first_token"].long()
    correct = live_first == gold_first
    questions = list(cache["evaluation_questions"])
    expected = int(settings.expected_examples)
    if trajectory.shape[0] != expected or len(questions) != expected:
        raise RuntimeError("population size drifted from the frozen contract")
    solutions = load_solutions(
        args.solutions,
        expected_sha256=str(settings.solutions_sha256),
        questions=questions,
    )

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(
        str(cfg.model.base_model), revision=str(cfg.model.base_revision)
    )
    thought_numbers = decode_thought_numbers(
        trajectory,
        readout,
        tokenizer,
        state=int(settings.workspace_state),
        top_k=int(settings.top_k),
    )

    test_rows = [int(v) for v in export["indices"]["test"]]
    derangement = seeded_derangement(len(test_rows), seed=int(settings.null_seed))

    recovery = torch.zeros(len(test_rows))
    null_recovery = torch.zeros(len(test_rows))
    scored = torch.zeros(len(test_rows), dtype=torch.bool)
    thought_hits = torch.zeros(len(test_rows), trajectory.shape[1], dtype=torch.bool)
    own_in = torch.zeros(len(test_rows), dtype=torch.bool)
    gold_in = torch.zeros(len(test_rows), dtype=torch.bool)
    for position, row in enumerate(test_rows):
        numbers = thought_numbers[row]
        intermediates = solutions[row]["intermediates"]
        null_targets = solutions[test_rows[int(derangement[position])]]["intermediates"]
        if intermediates and null_targets:
            scored[position] = True
            recovery[position] = recovery_fraction(numbers, intermediates)
            null_recovery[position] = recovery_fraction(numbers, null_targets)
            thought_hits[position] = torch.tensor(
                per_thought_hits(numbers, intermediates)
            )
        union = set().union(*numbers) if numbers else set()
        own_in[position] = (
            tokenizer.decode([int(live_first[row])]).strip() in union
        )
        gold_in[position] = (
            tokenizer.decode([int(gold_first[row])]).strip() in union
        )

    alignment = alignment_table(
        thought_numbers,
        [solution["intermediates"] for solution in solutions],
        test_rows,
        value_slots=tuple(int(v) for v in settings.value_slots),
    )

    summary = {
        "schema_version": LATENT_WORKSPACE_SCHEMA_VERSION,
        "contract": LATENT_WORKSPACE_CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trajectory_request_sha256": export.get("request_sha256"),
        "source_request_sha256": export.get("source_request_sha256"),
        "partition_sha256": export["partition_sha256"],
        "solutions_sha256": str(settings.solutions_sha256),
        "workspace_state": int(settings.workspace_state),
        "top_k": int(settings.top_k),
        "null_seed": int(settings.null_seed),
        "splits": {
            "test": len(test_rows),
            "test_scored": int(scored.sum()),
            "test_correct_share": float(correct[torch.tensor(test_rows)].double().mean()),
            "partition_sha256": export["partition_sha256"],
        },
        "alignment_table": alignment,
    }
    artifact = {
        "contract": LATENT_WORKSPACE_CONTRACT,
        "partition_sha256": export["partition_sha256"],
        "test_rows": torch.tensor(test_rows, dtype=torch.long),
        "test_recovery": recovery,
        "test_null_recovery": null_recovery,
        "test_scored_mask": scored,
        "test_thought_hits": thought_hits,
        "test_correct": correct[torch.tensor(test_rows)],
        "test_own_token_in_thoughts": own_in,
        "test_gold_token_in_thoughts": gold_in,
    }
    artifact_path = args.artifact_output or args.output.with_suffix(".pt")
    _atomic_torch_save(artifact, artifact_path)
    _atomic_json(summary, args.output)
    report = analyze_latent_workspace(summary, artifact, settings)
    report_path = args.report_output or args.output.with_name(
        "latent_workspace_report.json"
    )
    _atomic_json(report, report_path)
    print(
        f"[complete] status={report['status']} gates_passed={report['gates_passed']}"
    )
    # The endpoint tensor is loaded only to prove the export is intact end to end.
    assert endpoint.shape[0] == expected
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
