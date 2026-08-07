"""Run one frozen-checkpoint CODI answer-colon inference-ablation arm."""
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
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set
from src.data.official_codi_training import OFFICIAL_CODI_SOURCE_REVISION
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.endpoint_inference_ablation import (
    ENDPOINT_ABLATION_CONTRACT,
    ENDPOINT_ABLATION_SCHEMA_VERSION,
    OfficialCODIEndpointHiddenAblation,
    build_endpoint_ablation_specs,
    endpoint_ablation_spec_state,
)
from src.mech.endpoint_accuracy_localization import (
    ACCURACY_LOCALIZATION_CONTRACT,
    ACCURACY_LOCALIZATION_SCHEMA_VERSION,
    MATCHING_ALGORITHM,
    build_accuracy_localization_specs,
)
from src.mech.endpoint_retention import load_retention_bases, retention_bases_state
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    if (
        args.eval_limit < 0
        or args.eval_batch_size <= 0
        or args.random_replicates <= 0
        or not 0 < args.alpha <= 2
    ):
        raise ValueError(
            "evaluation limit must be non-negative, batch/replicates positive, "
            "and alpha in (0,2]"
        )
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("full endpoint inference ablation requires CUDA")
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
    bases = load_retention_bases(
        energy_path=args.energy_basis,
        answer_conditioned_path=args.answer_conditioned_basis,
        parameter_aware_path=args.parameter_aware_basis,
        checkpoint_sha256=load_report.checkpoint_sha256,
    )
    stats = torch.load(args.activation_stats, map_location="cpu", weights_only=False)
    if args.accuracy_localization:
        covariance_payload = stats.get("student_covariance_by_state")
        if not isinstance(covariance_payload, dict):
            raise RuntimeError("localization activation statistics lack full covariance")
        covariance_by_state = {
            int(state): value for state, value in covariance_payload.items()
        }
        specs = build_accuracy_localization_specs(
            bases,
            covariance_by_state,
            random_replicates=args.random_replicates,
            random_seed=args.random_seed,
        )
        contract = ACCURACY_LOCALIZATION_CONTRACT
        schema_version = ACCURACY_LOCALIZATION_SCHEMA_VERSION
        phase = "frozen_checkpoint_gsm8k_accuracy_localization"
    else:
        specs = build_endpoint_ablation_specs(
            bases,
            random_replicates=args.random_replicates,
            random_seed=args.random_seed,
        )
        contract = ENDPOINT_ABLATION_CONTRACT
        schema_version = ENDPOINT_ABLATION_SCHEMA_VERSION
        phase = "frozen_checkpoint_gsm8k_inference"
    if args.arm != "baseline" and args.arm not in specs:
        raise ValueError(
            f"unknown arm {args.arm!r}; use baseline or one of {sorted(specs)}"
        )
    if stats.get("contract") != contract:
        raise RuntimeError("activation statistics use another experiment contract")
    if stats.get("metadata", {}).get("checkpoint_sha256") != load_report.checkpoint_sha256:
        raise RuntimeError("activation mean uses a different CODI checkpoint")
    student_mean = stats.get("student_mean")
    if not isinstance(student_mean, torch.Tensor):
        raise RuntimeError("activation statistics are missing the student mean")

    source_state = retention_bases_state(bases)
    basis_sources = {
        name: {key: value for key, value in state.items() if key not in {"basis", "ranks"}}
        for name, state in source_state.items()
    }
    spec = None if args.arm == "baseline" else specs[args.arm]
    request = {
        "schema_version": schema_version,
        "contract": contract,
        "phase": phase,
        "arm": args.arm,
        "spec": None if spec is None else endpoint_ablation_spec_state(spec),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "basis_sources": basis_sources,
        "activation_stats_sha256": sha256_file(args.activation_stats),
        "activation_stats_request_sha256": stats.get("request_sha256"),
        "centering": "fresh_student_answer_colon_mean",
        "alpha": args.alpha,
        "random_replicates": args.random_replicates,
        "random_seed": args.random_seed,
        "eval_dataset": "gsm8k",
        "eval_limit": args.eval_limit,
        "eval_batch_size": args.eval_batch_size,
        "precision": args.precision,
        "weights_updated": False,
        "answer_cue_mode": "frozen EOT plus fixed answer cue, then greedy answer decoding",
        "state_module_map": {
            "11": "transformer.h[10] output",
            "12": "transformer.ln_f output after transformer.h[11]",
        },
        "intervention_timing": (
            "exact Hugging Face hidden-state entries 11/12 on the forward that "
            "consumes the fixed answer-cue colon"
        ),
    }
    if args.accuracy_localization:
        request["random_matching"] = {
            "matched_quantity": "per-state calibration E[||UU^T(h-mu)||^2]",
            "algorithm": MATCHING_ALGORITHM,
            "intervention_scaling": False,
            "selected_overlap_ceiling": 0.20,
        }
    request_sha256 = _sha256_json(request)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "gsm8k.jsonl"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("request_sha256") != request_sha256:
            raise RuntimeError("completed ablation output belongs to another request")
        if not predictions_path.is_file():
            raise RuntimeError("completed summary is missing paired predictions")
        print(f"[resume] already complete: {output_dir}")
        return summary
    _atomic_json(
        {**request, "request_sha256": request_sha256, "state": "running", "created_at_utc": datetime.now(timezone.utc).isoformat()},
        output_dir / "run_manifest.json",
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    intervention = (
        None
        if spec is None
        else OfficialCODIEndpointHiddenAblation(
            model, spec, student_mean=student_mean, alpha=args.alpha
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
        "state": "complete",
        "evaluated_examples": len(examples),
        "correct": int(sum(official_correct)),
        "accuracy": float(sum(official_correct) / len(examples)),
        "numeric_exact_match_accuracy": float(sum(numeric_correct) / len(examples)),
        "evaluation_seconds": elapsed,
        "examples_per_second": len(examples) / elapsed,
        "endpoint_coverage": {key: value for key, value in endpoint.items() if key != "endpoint_reached"},
        "intervention_diagnostics": None if intervention is None else intervention.diagnostics(),
        "predictions_file": predictions_path.name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(summary, summary_path)
    _atomic_json(summary, output_dir / "run_manifest.json")
    print(
        f"[complete] {args.arm}: accuracy={summary['accuracy']:.6f}, "
        f"cue_coverage={endpoint['endpoint_reached_fraction']:.3%}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--energy-basis", type=Path, required=True)
    parser.add_argument("--answer-conditioned-basis", type=Path, required=True)
    parser.add_argument("--parameter-aware-basis", type=Path, required=True)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--random-replicates", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260806)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--accuracy-localization",
        action="store_true",
        help="Use the v1 norm-matched hierarchical localization arm registry.",
    )
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
