"""Confirm one steering arm on full GSM8K with real decoding and exact match.

The analytic tier scores the first answer token, which is exact for state 12 but
is not the outcome the project's results are stated in.  A steering vector that
moves the first token can still fail to move the parsed numeric answer, so the
claim "steering the band improves accuracy" is only settled here.

The vector and its step size are read from the analytic export, which chose both
on the fit and select splits.  Nothing is re-tuned at this tier.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set
from src.data.official_codi_training import OFFICIAL_CODI_SOURCE_REVISION
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.endpoint_correctness_geometry import (
    CORRECTNESS_CONTRACT,
    CORRECTNESS_SCHEMA_VERSION,
    OfficialCODIEndpointSteerIntervention,
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


def load_steering_arm(vectors_path: Path, arm: str) -> tuple[torch.Tensor, float, str]:
    payload = torch.load(vectors_path, map_location="cpu", weights_only=False)
    if payload.get("contract") != CORRECTNESS_CONTRACT:
        raise RuntimeError("steering vectors belong to another contract")
    vectors = payload["steering_vectors"]
    if arm not in vectors:
        raise KeyError(f"unknown steering arm {arm}; have {sorted(vectors)}")
    alpha = float(payload["selected_alpha"][arm])
    return vectors[arm], alpha, str(payload.get("source_request_sha256"))


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
    cache, _ = load_margin_cache(args.states, args.readout)
    if str(cache["metadata"]["checkpoint_sha256"]) != load_report.checkpoint_sha256:
        raise RuntimeError("cached colon states belong to a different checkpoint")

    vector, alpha, source_sha = (None, 0.0, None)
    if args.arm != "baseline":
        vector, alpha, source_sha = load_steering_arm(args.vectors, args.arm)
        if source_sha != cache["request_sha256"]:
            raise RuntimeError(
                "steering vectors were fitted on a different colon-state collection"
            )
        if args.alpha is not None:
            alpha = float(args.alpha)

    request = {
        "schema_version": CORRECTNESS_SCHEMA_VERSION,
        "contract": CORRECTNESS_CONTRACT,
        "phase": "correctness_steer_full_gsm8k_generation",
        "arm": args.arm,
        "alpha": alpha,
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "states_request_sha256": cache["request_sha256"],
        "parity_gate": cache["parity_gate"],
        "eval_dataset": "gsm8k",
        "eval_limit": args.eval_limit,
        "eval_batch_size": args.eval_batch_size,
        "precision": args.precision,
        "vector_sha256": (
            None
            if vector is None
            else hashlib.sha256(vector.numpy().tobytes()).hexdigest()
        ),
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
        if vector is None
        else OfficialCODIEndpointSteerIntervention(model, vector, alpha=alpha)
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
            zip(
                examples,
                generations,
                official_correct,
                numeric_correct,
                endpoint["endpoint_reached"],
            )
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
    # Same guard as the margin runner: an unpinned precision resolving to emulated
    # bfloat16 shifts every arm together, and the reproduction gate cannot see it
    # because it reads a previously written summary instead of re-decoding.
    if args.arm == "baseline":
        drift = abs(summary["accuracy"] - float(reproduction["gsm8k_accuracy"]))
        allowed = float(settings.maximum_baseline_accuracy_drift)
        summary["baseline_accuracy_drift"] = drift
        summary["maximum_baseline_accuracy_drift"] = allowed
        summary["baseline_drift_passed"] = bool(drift <= allowed)
        if drift > allowed:
            raise RuntimeError(
                f"forced-cue baseline accuracy {summary['accuracy']:.6f} differs from "
                f"the reproduction gate {float(reproduction['gsm8k_accuracy']):.6f} by "
                f"{drift:.6f} > {allowed:.6f}; check that --precision is pinned "
                f"(resolved dtype was {dtype})"
            )
    _atomic_json(summary, summary_path)
    _atomic_json(summary, output_dir / "run_manifest.json")
    print(
        f"[complete] {args.arm} alpha={alpha:g}: accuracy={summary['accuracy']:.6f}, "
        f"cue_coverage={endpoint['endpoint_reached_fraction']:.3%}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--arm", required=True)
    # Left unset the step size comes from the analytic export, which chose it on
    # the select split. Overriding it here re-tunes on the test set.
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
