"""Collect a fresh student mean at the exact CODI answer-cue colon."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    verify_full_reproduction_gate,
)
from scripts.run_official_codi_endpoint_retention import sample_fresh_retention_training
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
)
from src.eval.official_codi import select_device
from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT
from src.mech.endpoint_inference_ablation import (
    ENDPOINT_ABLATION_CONTRACT,
    ENDPOINT_ABLATION_SCHEMA_VERSION,
)
from src.mech.endpoint_retention import load_retention_bases, retention_bases_state
from src.mech.official_codi_target_utility import OfficialCODIAnswerScorer
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision.casefold() in {"float32", "fp32"}:
        return nullcontext()
    dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(precision.casefold())
    if dtype is None:
        raise ValueError("precision must be float32, float16, or bfloat16")
    return torch.autocast(device_type="cuda", dtype=dtype)


def sample_fresh_ablation_calibration(dataset, *, examples: int, seed: int):
    """Exclude all selector data and the completed retention training partition."""
    retention_indices, retention_metadata = sample_fresh_retention_training(
        dataset, training_examples=512, seed=53
    )
    retention_questions = {
        _normalized_question(dataset[index]["question"]) for index in retention_indices
    }
    pool_indices, pool_metadata = sample_fresh_retention_training(
        dataset, training_examples=examples + 1024, seed=seed
    )
    selected = [
        index
        for index in pool_indices
        if _normalized_question(dataset[index]["question"]) not in retention_questions
    ][:examples]
    if len(selected) != examples:
        raise RuntimeError("not enough fresh calibration questions after retention exclusion")
    selected_questions = [_normalized_question(dataset[index]["question"]) for index in selected]
    if len(set(selected_questions)) != examples:
        raise RuntimeError("activation calibration contains duplicate normalized questions")
    return selected, {
        "calibration_indices_sha256": _sha256_json(selected),
        "retention_training_indices_sha256": retention_metadata[
            "training_indices_sha256"
        ],
        "pool_indices_sha256": pool_metadata["training_indices_sha256"],
        "excluded_retention_questions": len(retention_questions),
    }


def collect(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    if args.examples <= 0 or args.batch_size <= 0:
        raise ValueError("examples and batch size must be positive")
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("endpoint activation calibration requires CUDA")
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
    report = load_official_checkpoint(
        model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
    )
    bases = load_retention_bases(
        energy_path=args.energy_basis,
        answer_conditioned_path=args.answer_conditioned_basis,
        parameter_aware_path=args.parameter_aware_basis,
        checkpoint_sha256=report.checkpoint_sha256,
    )
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    indices, sampling = sample_fresh_ablation_calibration(
        dataset, examples=args.examples, seed=args.sampling_seed
    )
    sources = {
        name: {
            key: value
            for key, value in state.items()
            if key not in {"basis", "ranks"}
        }
        for name, state in retention_bases_state(bases).items()
    }
    request = {
        "schema_version": ENDPOINT_ABLATION_SCHEMA_VERSION,
        "contract": ENDPOINT_ABLATION_CONTRACT,
        "phase": "student_endpoint_mean_calibration",
        "checkpoint_sha256": report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "basis_sources": sources,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", "unavailable"),
        "examples": args.examples,
        "sampling_seed": args.sampling_seed,
        "sampling": sampling,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "states": [11, 12],
        "endpoint": "student fixed teacher-forced answer-cue colon after EOT",
    }
    request_sha256 = _sha256_json(request)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "activation_stats.pt"
    manifest_path = output_dir / "run_manifest.json"
    if stats_path.is_file():
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        if payload.get("request_sha256") != request_sha256:
            raise RuntimeError("existing activation statistics belong to another request")
        print(f"[resume] already complete: {stats_path}")
        return payload

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    scorer = OfficialCODIAnswerScorer(
        model, latent_positions=int(cfg.eval.latent_iterations)
    )
    total = torch.zeros(GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE, dtype=torch.float64)
    square_total = torch.zeros_like(total)
    count = 0
    progress = tqdm(total=len(indices), unit="examples", desc="Student colon activation stats")
    for start in range(0, len(indices), args.batch_size):
        selected = indices[start : start + args.batch_size]
        batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in selected],
            bot_token_id=model.bot_id,
        ).to(device)
        with torch.no_grad(), _amp_context(device, args.precision):
            output = scorer(batch, return_answer_endpoint_hidden=True)
        hidden = output.student_answer_endpoint_hidden
        if hidden is None or hidden.shape[1:] != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE):
            raise RuntimeError("student answer-colon hidden-state shape changed")
        values = hidden.detach().double().cpu()
        total += values.sum(dim=0)
        square_total += values.square().sum(dim=0)
        count += values.shape[0]
        progress.update(values.shape[0])
    progress.close()
    mean = total / count
    variance = (square_total / count - mean.square()).clamp_min(0)
    payload = {
        "schema_version": ENDPOINT_ABLATION_SCHEMA_VERSION,
        "contract": ENDPOINT_ABLATION_CONTRACT,
        "request_sha256": request_sha256,
        "metadata": request,
        "student_mean": mean.float(),
        "student_variance": variance.float(),
        "count": count,
        "indices": indices,
    }
    _atomic_torch_save(payload, stats_path)
    _atomic_json(
        {
            **request,
            "request_sha256": request_sha256,
            "state": "complete",
            "stats_file": stats_path.name,
            "stats_sha256": sha256_file(stats_path),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        manifest_path,
    )
    print(f"[complete] wrote {stats_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--energy-basis", type=Path, required=True)
    parser.add_argument("--answer-conditioned-basis", type=Path, required=True)
    parser.add_argument("--parameter-aware-basis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=67)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    collect(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
