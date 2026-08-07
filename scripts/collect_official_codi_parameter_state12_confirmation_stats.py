"""Fit parameter-aware state-12 matching statistics on disjoint GSM8K train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_activation_stats import _amp_context
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    verify_full_reproduction_gate,
)
from src.data.datasets import load_eval_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
    official_codi_answer_is_eligible,
)
from src.eval.official_codi import select_device
from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT
from src.mech.endpoint_retention import load_retention_bases, retention_bases_state
from src.mech.endpoint_state12_confirmation import (
    PRIMARY_STATE,
    STATE12_CONFIRMATION_CONTRACT,
    STATE12_CONFIRMATION_SCHEMA_VERSION,
)
from src.mech.official_codi_target_utility import OfficialCODIAnswerScorer
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


GSM8K_SOURCE_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_SOURCE_REVISION}/grade_school_math/data/train.jsonl"
)
GSM8K_EXPECTED_TRAIN_EXAMPLES = 7_473


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical_gsm8k_row(row: dict) -> dict | None:
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    if not question.strip() or "####" not in answer:
        return None
    cot, final = answer.rsplit("####", 1)
    final = final.strip().replace(",", "")
    canonical = {"question": question, "cot": cot.strip(), "answer": final}
    if not canonical["cot"] or not official_codi_answer_is_eligible(final):
        return None
    return canonical


def sample_gsm8k_train_calibration(
    rows,
    *,
    test_questions: set[str],
    examples: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Select unique eligible train questions and prove test-question disjointness."""
    if examples <= 0:
        raise ValueError("calibration examples must be positive")
    groups: dict[str, list[tuple[int, dict]]] = {}
    ineligible = 0
    for index, row in enumerate(rows):
        canonical = _canonical_gsm8k_row(dict(row))
        if canonical is None:
            ineligible += 1
            continue
        normalized = _normalized_question(canonical["question"])
        groups.setdefault(normalized, []).append((index, canonical))
    overlap = sorted(set(groups).intersection(test_questions))
    if overlap:
        raise RuntimeError(
            f"GSM8K train/test question overlap is nonzero ({len(overlap)})"
        )
    if len(groups) < examples:
        raise ValueError(f"need {examples} unique eligible train questions, found {len(groups)}")
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    selected_keys = keys[:examples]
    selected = []
    selected_indices = []
    for key in selected_keys:
        index, row = generator.choice(groups[key])
        selected.append(row)
        selected_indices.append(index)
    return selected, {
        "raw_train_examples": len(rows),
        "eligible_unique_train_questions": len(groups),
        "ineligible_rows": ineligible,
        "test_unique_questions": len(test_questions),
        "train_test_normalized_question_overlap": 0,
        "selected_examples": examples,
        "sampling_seed": seed,
        "selected_source_indices_sha256": _sha256_json(selected_indices),
        "selected_normalized_questions_sha256": _sha256_json(selected_keys),
        "test_normalized_questions_sha256": _sha256_json(sorted(test_questions)),
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
        raise RuntimeError("state-12 confirmation calibration requires CUDA")
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

    from datasets import load_dataset

    train = load_dataset(
        "json",
        data_files={"train": GSM8K_TRAIN_URL},
        split="train",
        verification_mode="no_checks",
    )
    if len(train) != GSM8K_EXPECTED_TRAIN_EXAMPLES:
        raise RuntimeError(
            f"GSM8K train count drifted: {len(train)} != {GSM8K_EXPECTED_TRAIN_EXAMPLES}"
        )
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    if len(test) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("GSM8K test count drifted")
    test_questions = {_normalized_question(row["question"]) for row in test}
    if len(test_questions) != len(test):
        raise RuntimeError("GSM8K test contains duplicate normalized questions")
    rows, sampling = sample_gsm8k_train_calibration(
        train,
        test_questions=test_questions,
        examples=args.examples,
        seed=args.sampling_seed,
    )
    sources = {
        name: {key: value for key, value in state.items() if key not in {"basis", "ranks"}}
        for name, state in retention_bases_state(bases).items()
    }
    request = {
        "schema_version": STATE12_CONFIRMATION_SCHEMA_VERSION,
        "contract": STATE12_CONFIRMATION_CONTRACT,
        "phase": "gsm8k_train_parameter_aware_state12_mean_and_covariance",
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "basis_sources": sources,
        "calibration_dataset": "openai/grade-school-math GSM8K train",
        "calibration_source_revision": GSM8K_SOURCE_REVISION,
        "calibration_source_url": GSM8K_TRAIN_URL,
        "calibration_dataset_fingerprint": getattr(train, "_fingerprint", "unavailable"),
        "examples": args.examples,
        "sampling_seed": args.sampling_seed,
        "sampling": sampling,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "states": [PRIMARY_STATE],
        "endpoint": "student fixed teacher-forced answer-cue colon after EOT",
        "test_labels_used_for_calibration": False,
        "test_activations_used_for_calibration": False,
    }
    request_sha256 = _sha256_json(request)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "activation_stats.pt"
    manifest_path = output_dir / "run_manifest.json"
    if stats_path.is_file():
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        if payload.get("request_sha256") != request_sha256:
            raise RuntimeError("existing confirmation statistics belong to another request")
        print(f"[resume] already complete: {stats_path}")
        return payload

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    scorer = OfficialCODIAnswerScorer(
        model, latent_positions=int(cfg.eval.latent_iterations)
    )
    total = torch.zeros(GPT2_HIDDEN_SIZE, dtype=torch.float64)
    cross_total = torch.zeros(
        GPT2_HIDDEN_SIZE, GPT2_HIDDEN_SIZE, dtype=torch.float64
    )
    count = 0
    progress = tqdm(total=len(rows), unit="examples", desc="GSM8K-train state-12 stats")
    for start in range(0, len(rows), args.batch_size):
        selected_rows = rows[start : start + args.batch_size]
        batch = collate_official_codi_kv_rows(
            tokenizer, selected_rows, bot_token_id=model.bot_id
        ).to(device)
        with torch.no_grad(), _amp_context(device, args.precision):
            output = scorer(batch, return_answer_endpoint_hidden=True)
        hidden = output.student_answer_endpoint_hidden
        if hidden is None or hidden.shape[1:] != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE):
            raise RuntimeError("student answer-colon hidden-state shape changed")
        values = hidden[:, PRIMARY_STATE, :].detach().double().cpu()
        total += values.sum(dim=0)
        cross_total += values.T @ values
        count += values.shape[0]
        progress.update(values.shape[0])
    progress.close()
    mean_state12 = total / count
    covariance = cross_total / count - torch.outer(mean_state12, mean_state12)
    covariance = 0.5 * (covariance + covariance.T)
    student_mean = torch.zeros(GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE, dtype=torch.float32)
    student_mean[PRIMARY_STATE] = mean_state12.float()
    payload = {
        "schema_version": STATE12_CONFIRMATION_SCHEMA_VERSION,
        "contract": STATE12_CONFIRMATION_CONTRACT,
        "request_sha256": request_sha256,
        "metadata": request,
        "student_mean": student_mean,
        "student_covariance_by_state": {str(PRIMARY_STATE): covariance.float()},
        "count": count,
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
    parser.add_argument("--examples", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=73)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    collect(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
