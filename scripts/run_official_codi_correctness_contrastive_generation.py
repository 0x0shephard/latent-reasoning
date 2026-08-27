"""Run one paired exact-match arm from the contrastive covariance artifact."""
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
from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT
from src.mech.endpoint_correctness_contrastive_covariance import (
    CONTRASTIVE_COVARIANCE_CONTRACT,
    CONTRASTIVE_COVARIANCE_SCHEMA_VERSION,
)
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    MarginSubspace,
    OfficialCODIEndpointSubspaceIntervention,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


GENERATION_ARMS = (
    "baseline",
    "contrastive_correct_retain",
    "correct_only_pca_retain",
    "accuracy_band_pca_retain",
    "contrastive_wrong_remove",
)


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=GENERATION_ARMS, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    cache, _ = load_margin_cache(args.states, args.readout)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("contract") != CONTRASTIVE_COVARIANCE_CONTRACT:
        raise RuntimeError("basis artifact belongs to another experiment")
    if artifact.get("source_request_sha256") != cache.get("request_sha256"):
        raise RuntimeError("basis artifact and state cache do not match")

    original_indices = [int(value) for value in artifact["indices"]["test"]]
    arm_spec = None if args.arm == "baseline" else artifact["arms"].get(args.arm)
    if args.arm != "baseline" and arm_spec is None:
        raise KeyError(f"artifact has no arm {args.arm}")
    request = {
        "schema_version": CONTRASTIVE_COVARIANCE_SCHEMA_VERSION,
        "contract": CONTRASTIVE_COVARIANCE_CONTRACT,
        "phase": "paired_exact_match_generation",
        "arm": args.arm,
        "partition_sha256": artifact["partition_sha256"],
        "source_request_sha256": cache.get("request_sha256"),
        "checkpoint_sha256": cache["metadata"]["checkpoint_sha256"],
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "test_indices": original_indices,
        "precision": args.precision,
        "eval_batch_size": args.eval_batch_size,
    }
    request_sha256 = _sha256_json(request)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    predictions_path = args.output_dir / "gsm8k.jsonl"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("request_sha256") != request_sha256:
            raise RuntimeError("existing output belongs to another request")
        print(f"[resume] already complete: {args.arm}")
        return 0

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
    if load_report.checkpoint_sha256 != request["checkpoint_sha256"]:
        raise RuntimeError("cached states came from a different checkpoint")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    intervention = None
    if arm_spec is not None:
        basis = arm_spec["basis"].float()
        centre = arm_spec["centre"].float()
        student_mean = cache["student_mean"].clone().float()
        if student_mean.shape != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE):
            raise RuntimeError("cached student mean has the wrong shape")
        student_mean[ANALYTIC_STATE] = centre
        subspace = MarginSubspace(
            name=args.arm,
            family=str(arm_spec["family"]),
            state=ANALYTIC_STATE,
            basis=basis,
            rank=int(basis.shape[1]),
        )
        intervention = OfficialCODIEndpointSubspaceIntervention(
            model,
            [subspace],
            student_mean=student_mean,
            mode=str(arm_spec["mode"]),
            semantics="mean",
            all_positions=False,
            alpha=1.0,
        )

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    style = PromptStyle.from_config(data_cfg.prompt)
    full_examples = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    if len(full_examples) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("GSM8K evaluation count drifted")
    examples = [full_examples[index] for index in original_indices]
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
    correct = [
        bool(official_answers_match(generation, example["gold"]))
        for generation, example in zip(generations, examples)
    ]
    numeric = [
        bool(answers_match(generation, example["gold"]))
        for generation, example in zip(generations, examples)
    ]
    reached = endpoint["endpoint_reached"]
    temporary = predictions_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for original_index, example, generation, is_correct, is_numeric, cue in zip(
            original_indices, examples, generations, correct, numeric, reached
        ):
            handle.write(
                json.dumps(
                    {
                        "index": original_index,
                        "question": example["question"],
                        "gold": str(example["gold"]),
                        "generation": generation,
                        "correct": is_correct,
                        "numeric_exact_match_correct": is_numeric,
                        "answer_cue_endpoint_reached": bool(cue),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(predictions_path)
    summary = {
        **request,
        "request_sha256": request_sha256,
        "state_field": "complete",
        "evaluated_examples": len(examples),
        "correct": int(sum(correct)),
        "accuracy": float(sum(correct) / len(examples)),
        "numeric_exact_match_accuracy": float(sum(numeric) / len(examples)),
        "endpoint_coverage": {
            key: value for key, value in endpoint.items() if key != "endpoint_reached"
        },
        "intervention_diagnostics": (
            None if intervention is None else intervention.diagnostics()
        ),
        "evaluation_seconds": elapsed,
        "predictions_file": predictions_path.name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(summary, summary_path)
    _atomic_json(summary, args.output_dir / "run_manifest.json")
    print(f"[complete] {args.arm}: accuracy={summary['accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
