"""Run one full-GSM8K generation arm for the margin-geometry experiment.

The analytic tier is exact only for state 12 and only for the first answer token.
This runner covers everything that claim does not reach: state 11, which does
enter the key/value cache; all-position interventions; and the numeric
exact-match outcome the project's earlier results are stated in.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_official_codi_endpoint_margin_sweep import (
    load_margin_cache,
    prepare_registry,
)
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set
from src.data.official_codi_training import OFFICIAL_CODI_SOURCE_REVISION
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    MARGIN_GEOMETRY_CONTRACT,
    MARGIN_GEOMETRY_SCHEMA_VERSION,
    PROPAGATING_STATE,
    MarginSubspace,
    OfficialCODIEndpointSubspaceIntervention,
    validate_margin_subspace,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def required_random_replicates(arm: str) -> int:
    """How many control replicates must be drawn before ``arm`` is reachable.

    Controls are drawn sequentially from one seeded generator, so replicate ``r`` is
    bit-identical whether the registry stops at ``r+1`` or continues to 200.
    """
    match = re.search(r"_r(\d+)$", arm)
    return int(match.group(1)) + 1 if match else 1


def resolve_arm_subspace(
    registry: dict[str, MarginSubspace], *, arm: str, state: int
) -> MarginSubspace:
    """Look an arm up by name and, if needed, move it to the requested state.

    State-11 arms reuse the state-12 basis by construction so that the only
    difference between the two sites is where the edit lands, not which
    directions are removed.  That is the whole point of the propagation contrast.
    """
    if arm not in registry:
        raise KeyError(f"unknown margin-geometry arm {arm}")
    subspace = registry[arm]
    if subspace.state == state:
        return subspace
    moved = MarginSubspace(
        name=f"{subspace.name}_at_s{state}",
        family=subspace.family,
        state=state,
        basis=subspace.basis.clone(),
        rank=subspace.rank,
        random_replicate=subspace.random_replicate,
        calibration_target_energy=subspace.calibration_target_energy,
        calibration_achieved_energy=subspace.calibration_achieved_energy,
        selected_overlap=subspace.selected_overlap,
        matched_family=subspace.matched_family,
    )
    validate_margin_subspace(moved)
    return moved


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    settings = cfg.endpoint_margin_geometry
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id),
        revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename),
        expected_sha256=str(cfg.checkpoint.sha256),
        token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model),
        base_revision=str(cfg.model.base_revision),
        dtype=dtype,
        settings=cfg.model,
        token=token,
    )
    load_report = load_official_checkpoint(
        model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
    )
    cache, readout_payload = load_margin_cache(args.states, args.readout)
    if str(cache["metadata"]["checkpoint_sha256"]) != load_report.checkpoint_sha256:
        raise RuntimeError("cached colon states belong to a different checkpoint")

    subspace = None
    student_mean = cache["student_mean"]
    if student_mean.shape != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE):
        raise RuntimeError("cached student mean has the wrong shape")
    if args.arm != "baseline":
        # Replicates are drawn in order from one seeded generator, so replicate ``r``
        # is identical whether the registry holds ``r+1`` or all 200 of them.
        # Building only as many as this arm needs turns a ~3-minute rebuild per arm
        # into a few seconds without changing any basis.
        prepared = prepare_registry(
            cache,
            readout_payload,
            settings,
            args,
            torch.device("cpu"),
            random_replicates=required_random_replicates(args.arm),
        )
        subspace = resolve_arm_subspace(
            prepared["registry"], arm=args.arm, state=args.state
        )

    request = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "phase": "margin_geometry_full_gsm8k_generation",
        "arm": args.arm,
        "state": args.state,
        "mode": args.mode,
        "semantics": args.semantics,
        "all_positions": bool(args.all_positions),
        "alpha": args.alpha,
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "states_request_sha256": cache["request_sha256"],
        "parity_gate": cache["parity_gate"],
        "eval_dataset": "gsm8k",
        "eval_limit": args.eval_limit,
        "eval_batch_size": args.eval_batch_size,
        "precision": args.precision,
        "subspace": None if subspace is None else subspace.state_dict(),
    }
    request_sha256 = _sha256_json(request)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "gsm8k.jsonl"
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("request_sha256") != request_sha256:
            raise RuntimeError("completed summary belongs to another request")
        if not predictions_path.is_file():
            raise RuntimeError("completed summary is missing paired predictions")
        print(f"[resume] already complete: {args.arm}")
        return summary

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    intervention = (
        None
        if subspace is None
        else OfficialCODIEndpointSubspaceIntervention(
            model,
            [subspace],
            student_mean=student_mean,
            mode=args.mode,
            semantics=args.semantics,
            all_positions=bool(args.all_positions),
            alpha=args.alpha,
        )
    )
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    style = PromptStyle.from_config(data_cfg.prompt)
    examples = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    if len(examples) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("GSM8K evaluation count drifted")
    if args.eval_limit:
        examples = examples[: args.eval_limit]
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    generations, endpoint = generate_official_codi(
        model,
        tokenizer,
        [example["question"] for example in examples],
        latent_iterations=int(cfg.eval.latent_iterations),
        max_new_tokens=int(cfg.eval.max_new_tokens),
        batch_size=args.eval_batch_size,
        device=device,
        answer_endpoint_intervention=intervention,
        answer_cue=style.answer_prefix,
        force_answer_cue=True,
        return_endpoint_metadata=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    official_correct = [
        bool(official_answers_match(generation, example["gold"]))
        for generation, example in zip(generations, examples)
    ]
    numeric_correct = [
        bool(answers_match(generation, example["gold"]))
        for generation, example in zip(generations, examples)
    ]
    endpoint_flags = endpoint["endpoint_reached"]
    records = [
        {
            "index": index,
            "question": example["question"],
            "gold": str(example["gold"]),
            "generation": generation,
            "correct": correct,
            "numeric_exact_match_correct": numeric,
            "answer_cue_endpoint_reached": bool(reached),
        }
        for index, (example, generation, correct, numeric, reached) in enumerate(
            zip(examples, generations, official_correct, numeric_correct, endpoint_flags)
        )
    ]
    temporary = predictions_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(predictions_path)
    summary = {
        **request,
        "request_sha256": request_sha256,
        "state_field": "complete",
        "evaluated_examples": len(examples),
        "correct": int(sum(official_correct)),
        "accuracy": float(sum(official_correct) / len(examples)),
        "numeric_exact_match_accuracy": float(sum(numeric_correct) / len(examples)),
        "evaluation_seconds": elapsed,
        "examples_per_second": len(examples) / elapsed,
        "endpoint_coverage": {
            key: value for key, value in endpoint.items() if key != "endpoint_reached"
        },
        "intervention_diagnostics": (
            None if intervention is None else intervention.diagnostics()
        ),
        "predictions_file": predictions_path.name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(summary, summary_path)
    _atomic_json(summary, output_dir / "run_manifest.json")
    print(
        f"[complete] {args.arm} s{args.state} {args.mode}/{args.semantics}: "
        f"accuracy={summary['accuracy']:.6f}, "
        f"cue_coverage={endpoint['endpoint_reached_fraction']:.3%}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--energy-basis", type=Path, required=True)
    parser.add_argument("--answer-conditioned-basis", type=Path, required=True)
    parser.add_argument("--parameter-aware-basis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--state", type=int, default=ANALYTIC_STATE, choices=[PROPAGATING_STATE, ANALYTIC_STATE]
    )
    parser.add_argument("--mode", default="remove", choices=["remove", "retain"])
    parser.add_argument("--semantics", default="mean", choices=["mean", "zero"])
    parser.add_argument("--all-positions", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
