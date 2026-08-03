"""Fit per-layer TSV-C-inspired bases for official CODI endpoint residuals."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_kv_subspaces import _verify_reproduction_gate
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
    official_codi_answer_is_eligible,
)
from src.eval.official_codi import select_device
from src.mech.endpoint_tsvc import (
    ENDPOINT_TSVC_SCHEMA_VERSION,
    bases_from_state,
    bases_to_state,
    create_endpoint_moments,
    endpoint_moments_from_state,
    endpoint_moments_state,
    explained_energy,
    fit_endpoint_tsvc_bases,
    update_endpoint_moments,
    validate_endpoint_bases,
)
from src.mech.official_codi_target_utility import (
    OfficialCODIAnswerScorer,
    extract_official_teacher_endpoint_targets,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


_NORMALIZE_SPACE = re.compile(r"\s+")


def verify_full_reproduction_gate(path: Path, cfg) -> dict:
    """Require the passed gate and the complete preregistered GSM8K count."""
    gate = _verify_reproduction_gate(path, cfg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.get("evaluated_counts", {}).get("gsm8k")
    expected = int(cfg.eval.expected_counts.gsm8k)
    if int(observed or -1) != expected:
        raise RuntimeError(
            "endpoint TSV-C requires the complete official GSM8K reproduction "
            f"summary ({expected} examples), observed {observed!r}"
        )
    return {**gate, "evaluated_gsm8k_examples": int(observed)}


def _normalized_question(value: object) -> str:
    return _NORMALIZE_SPACE.sub(" ", str(value).strip().casefold())


def _sha256_json(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sample_endpoint_tsvc_partitions(
    dataset,
    *,
    calibration_examples: int,
    update_examples: int,
    validation_examples: int,
    seed: int,
) -> tuple[dict[str, list[int]], dict]:
    """Select one eligible row per normalized question into three disjoint sets."""
    counts = (calibration_examples, update_examples, validation_examples)
    if any(value <= 0 for value in counts):
        raise ValueError("all endpoint TSV-C partition sizes must be positive")
    groups: dict[str, list[int]] = {}
    for index, (question, answer) in enumerate(
        zip(dataset["question"], dataset["answer"])
    ):
        if official_codi_answer_is_eligible(answer):
            groups.setdefault(_normalized_question(question), []).append(index)
    required = sum(counts)
    if len(groups) < required:
        raise ValueError(
            f"need {required} unique eligible questions, found {len(groups)}"
        )
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    selected = [generator.choice(groups[key]) for key in keys[:required]]
    calibration_end = calibration_examples
    update_end = calibration_end + update_examples
    partitions = {
        "calibration": selected[:calibration_end],
        "update": selected[calibration_end:update_end],
        "validation": selected[update_end:],
    }
    question_sets = {
        name: {
            _normalized_question(dataset[index]["question"]) for index in indices
        }
        for name, indices in partitions.items()
    }
    if (
        question_sets["calibration"] & question_sets["update"]
        or question_sets["calibration"] & question_sets["validation"]
        or question_sets["update"] & question_sets["validation"]
    ):
        raise RuntimeError("endpoint TSV-C question groups overlap")
    return partitions, {
        "eligible_unique_question_groups": len(groups),
        "selection": "one_seeded_row_per_normalized_question_three_way_v1",
        "partition_sha256": {
            name: _sha256_json(indices) for name, indices in partitions.items()
        },
    }


def _amp_context(device: torch.device, precision: str):
    normalized = precision.casefold()
    if normalized in {"float32", "fp32"} or device.type != "cuda":
        return nullcontext()
    dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(normalized)
    if dtype is None:
        raise ValueError("precision must be float32, float16, or bfloat16")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _collection_identity(metadata: dict) -> dict:
    keys = (
        "checkpoint_revision",
        "checkpoint_sha256",
        "official_source_revision",
        "dataset_fingerprint",
        "calibration_indices_sha256",
        "calibration_examples",
        "batch_size",
        "sampling_seed",
        "rank",
        "random_basis_seed",
        "layers",
        "hidden_size",
        "precision",
        "alignment",
    )
    return {key: metadata.get(key) for key in keys}


def _save_progress(
    output_dir: Path,
    *,
    request_sha256: str,
    state: str,
    processed: int,
    requested: int,
) -> None:
    _atomic_json(
        {
            "schema_version": ENDPOINT_TSVC_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "state": state,
            "processed_calibration_examples": processed,
            "requested_calibration_examples": requested,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "progress.json",
    )


@torch.no_grad()
def collect(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    settings = cfg.endpoint_tsvc
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("endpoint TSV-C collection requires CUDA")
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else download_official_checkpoint(
            repo_id=str(cfg.checkpoint.repo_id),
            revision=str(cfg.checkpoint.revision),
            filename=str(cfg.checkpoint.filename),
            expected_sha256=str(cfg.checkpoint.sha256),
            token=token,
        )
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model),
        base_revision=str(cfg.model.base_revision),
        dtype=dtype,
        settings=cfg.model,
        token=token,
    )
    load_report = load_official_checkpoint(
        model,
        checkpoint,
        expected_sha256=str(cfg.checkpoint.sha256),
    )
    model.to(device=device, dtype=dtype).eval()
    torch.manual_seed(args.sampling_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.sampling_seed)

    data_cfg = load_config(str(settings.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    partitions, sampling = sample_endpoint_tsvc_partitions(
        dataset,
        calibration_examples=args.calibration_examples,
        update_examples=args.update_examples,
        validation_examples=args.validation_examples,
        seed=args.sampling_seed,
    )
    calibration_indices = partitions["calibration"]
    layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    if layers != 12 or hidden_size != 768:
        raise RuntimeError("official endpoint TSV-C requires GPT-2 [12,768]")
    if args.rank != 77:
        raise RuntimeError("preregistered endpoint TSV-C rank must remain 77")

    metadata = {
        "analysis": "official_codi_endpoint_tsvc_calibration",
        "scientific_scope": (
            "TSV-C-inspired activation filtering; original TSV-C decomposes "
            "weight-difference matrices"
        ),
        "reproduction_gate": reproduction,
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", "unavailable"),
        "calibration_examples": len(calibration_indices),
        "update_examples": len(partitions["update"]),
        "validation_examples": len(partitions["validation"]),
        "partitions": partitions,
        "sampling": sampling,
        "calibration_indices_sha256": sampling["partition_sha256"]["calibration"],
        "batch_size": args.batch_size,
        "sampling_seed": args.sampling_seed,
        "rank": args.rank,
        "random_basis_seed": args.random_basis_seed,
        "layers": layers,
        "hidden_size": hidden_size,
        "latent_positions": int(cfg.eval.latent_iterations),
        "precision": args.precision,
        "alignment": (
            "teacher block state at final token of 'The answer is:' paired with "
            "student block state after latent iteration six; embedding state excluded"
        ),
        "decomposition": "uncentered per-layer residual Gram eigendecomposition",
        "residual": "student_endpoint - stop_gradient(teacher_endpoint)",
    }
    request_sha256 = _sha256_json(_collection_identity(metadata))
    metadata["request_sha256"] = request_sha256
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    state_path = output_dir / "collection_state.pt"
    basis_path = output_dir / "basis.pt"

    previous_manifest = None
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text())
        if previous_manifest.get("request_sha256") != request_sha256:
            raise RuntimeError("output directory contains a different calibration request")

    if basis_path.is_file():
        payload = torch.load(basis_path, map_location="cpu", weights_only=False)
        if payload.get("request_sha256") != request_sha256:
            raise RuntimeError("existing basis identity does not match request")
        existing_bases = bases_from_state(payload["bases"])
        validate_endpoint_bases(
            existing_bases,
            layers=layers,
            hidden_size=hidden_size,
        )
        expected_basis_sha = (
            previous_manifest.get("basis_sha256") if previous_manifest else None
        )
        observed_basis_sha = sha256_file(basis_path)
        if expected_basis_sha and expected_basis_sha != observed_basis_sha:
            raise RuntimeError("existing endpoint basis SHA256 does not match manifest")
        _atomic_json(
            {
                **metadata,
                "state": "complete",
                "processed_examples": len(calibration_indices),
                "basis_file": basis_path.name,
                "basis_sha256": observed_basis_sha,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            manifest_path,
        )
        _save_progress(
            output_dir,
            request_sha256=request_sha256,
            state="complete",
            processed=len(calibration_indices),
            requested=len(calibration_indices),
        )
        print(f"[complete] verified existing basis: {basis_path}")
        return payload["metadata"]

    _atomic_json(
        {
            **metadata,
            "state": "running",
            "created_at_utc": (
                previous_manifest.get("created_at_utc")
                if previous_manifest
                else datetime.now(timezone.utc).isoformat()
            ),
        },
        manifest_path,
    )

    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("request_sha256") != request_sha256:
            raise RuntimeError("existing calibration state does not match request")
        moments = endpoint_moments_from_state(state["moments"])
        processed = int(state["processed_examples"])
        if calibration_indices[:processed] != state["processed_indices"]:
            raise RuntimeError("calibration resume prefix changed")
        print(f"[resume] continuing endpoint calibration from {processed}")
    else:
        moments = create_endpoint_moments(layers, hidden_size)
        processed = 0

    scorer = OfficialCODIAnswerScorer(
        model,
        latent_positions=int(cfg.eval.latent_iterations),
    )
    progress = tqdm(
        total=len(calibration_indices),
        initial=processed,
        unit="examples",
        desc="Official CODI endpoint TSV-C calibration",
    )
    while processed < len(calibration_indices):
        end = min(processed + args.batch_size, len(calibration_indices))
        batch_indices = calibration_indices[processed:end]
        rows = [dataset[index] for index in batch_indices]
        batch = collate_official_codi_kv_rows(
            tokenizer,
            rows,
            bot_token_id=model.bot_id,
        ).to(device)
        with _amp_context(device, args.precision):
            teacher = extract_official_teacher_endpoint_targets(model, batch).hidden
            student_output = scorer(
                batch,
                return_kv=False,
                return_endpoint_hidden=True,
            )
        student = student_output.student_endpoint_hidden
        if student is None or tuple(student.shape) != (
            len(batch_indices),
            layers,
            hidden_size,
        ):
            raise RuntimeError("student endpoint extraction violated [B,12,768]")
        if tuple(teacher.shape) != tuple(student.shape):
            raise RuntimeError("teacher and student endpoint shapes do not match")
        update_endpoint_moments(moments, student, teacher)
        if processed == 0:
            endpoint_tokens = batch.teacher_ids[
                torch.arange(len(batch_indices), device=device),
                batch.teacher_endpoint,
            ]
            _atomic_json(
                {
                    "teacher_endpoint_shape": list(teacher.shape),
                    "student_endpoint_shape": list(student.shape),
                    "teacher_endpoint_token_ids": [
                        int(value) for value in endpoint_tokens.detach().cpu()
                    ],
                    "student_latent_iterations": int(cfg.eval.latent_iterations),
                    "transformer_blocks": layers,
                    "embedding_state_excluded": True,
                    "finite_teacher": bool(torch.isfinite(teacher).all()),
                    "finite_student": bool(torch.isfinite(student).all()),
                },
                output_dir / "alignment_audit.json",
            )
        newly_processed = end - processed
        processed = end
        progress.update(newly_processed)
        if processed < len(calibration_indices) and (
            processed % args.save_every < newly_processed
        ):
            _atomic_torch_save(
                {
                    "schema_version": ENDPOINT_TSVC_SCHEMA_VERSION,
                    "request_sha256": request_sha256,
                    "processed_examples": processed,
                    "processed_indices": calibration_indices[:processed],
                    "moments": endpoint_moments_state(moments),
                },
                state_path,
            )
            _save_progress(
                output_dir,
                request_sha256=request_sha256,
                state="running",
                processed=processed,
                requested=len(calibration_indices),
            )
        del batch, teacher, student, student_output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress.close()

    bases = fit_endpoint_tsvc_bases(
        moments,
        rank=args.rank,
        random_seed=args.random_basis_seed,
    )
    basis_payload = {
        "schema_version": ENDPOINT_TSVC_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "metadata": metadata,
        "bases": bases_to_state(bases),
        "top_rank_explained_energy_by_layer": explained_energy(
            bases.eigenvalues,
            args.rank,
        ),
    }
    _atomic_torch_save(basis_payload, basis_path)
    basis_sha256 = sha256_file(basis_path)
    _atomic_json(
        {
            **metadata,
            "state": "complete",
            "processed_examples": processed,
            "basis_file": basis_path.name,
            "basis_sha256": basis_sha256,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        manifest_path,
    )
    _save_progress(
        output_dir,
        request_sha256=request_sha256,
        state="complete",
        processed=processed,
        requested=len(calibration_indices),
    )
    print(f"[complete] fitted rank-{args.rank} endpoint bases: {basis_path}")
    print(f"[complete] basis SHA256: {basis_sha256}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit official-CODI endpoint TSV-C-inspired bases."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/official_codi_gpt2.yaml"),
    )
    parser.add_argument("--reproduction-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--calibration-examples", type=int, default=5000)
    parser.add_argument("--update-examples", type=int, default=256)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--rank", type=int, default=77)
    parser.add_argument("--sampling-seed", type=int, default=11)
    parser.add_argument("--random-basis-seed", type=int, default=20260803)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.save_every <= 0:
        parser.error("batch size and save interval must be positive")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
