"""Fit corrected source-faithful CODI answer-cue endpoint TSV-C bases."""
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
    sample_endpoint_tsvc_partitions,
    verify_full_reproduction_gate,
)
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
)
from src.eval.official_codi import select_device
from src.mech.endpoint_tsvc import (
    bases_from_state,
    bases_to_state,
    create_endpoint_moments,
    endpoint_moments_from_state,
    endpoint_moments_state,
    explained_energy,
    fit_endpoint_tsvc_bases,
    update_endpoint_moments,
)
from src.mech.endpoint_tsvc_corrected import (
    CORRECTED_ENDPOINT_TSVC_SCHEMA_VERSION,
    corrected_endpoint_tsvc_loss,
    relative_gradient_error,
    source_faithful_native_endpoint_loss,
    validate_corrected_bases,
    validate_corrected_endpoint_hidden,
)
from src.mech.kv_target_utility import autograd_gradients
from src.mech.official_codi_target_utility import (
    OfficialCODIAnswerScorer,
    build_official_student_answer_io,
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


def _sha256_json(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
            "schema_version": CORRECTED_ENDPOINT_TSVC_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "state": state,
            "processed_calibration_examples": processed,
            "requested_calibration_examples": requested,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "progress.json",
    )


def _collection_identity(metadata: dict) -> dict:
    keys = (
        "contract",
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
        "hidden_states",
        "hidden_size",
        "precision",
        "alignment",
        "loss",
    )
    return {key: metadata.get(key) for key in keys}


def _run_native_parity_gate(
    *,
    scorer,
    model,
    tokenizer,
    dataset,
    indices: list[int],
    parameters: list[torch.Tensor],
    device: torch.device,
    precision: str,
    output_path: Path,
) -> dict:
    rows = [dataset[index] for index in indices]
    batch = collate_official_codi_kv_rows(
        tokenizer,
        rows,
        bot_token_id=model.bot_id,
    ).to(device)
    teacher = extract_official_teacher_endpoint_targets(model, batch).all_hidden
    with _amp_context(device, precision):
        output = scorer(
            batch,
            return_kv=False,
            return_endpoint_hidden=False,
            return_answer_endpoint_hidden=True,
        )
    student = output.student_answer_endpoint_hidden
    if student is None:
        raise RuntimeError("corrected student answer endpoint was not returned")
    validate_corrected_endpoint_hidden(student, teacher)

    reference_loss = source_faithful_native_endpoint_loss(student, teacher)
    candidate_loss = corrected_endpoint_tsvc_loss(
        student,
        teacher,
        scope="endpoint_all_states",
        mode="full",
    )
    reference_gradients = autograd_gradients(
        reference_loss,
        parameters,
        retain_graph=True,
    )
    candidate_gradients = autograd_gradients(
        candidate_loss,
        parameters,
        retain_graph=False,
    )
    relative_error, cosine = relative_gradient_error(
        candidate_gradients,
        reference_gradients,
    )
    absolute_loss_error = abs(
        float(candidate_loss.detach().float())
        - float(reference_loss.detach().float())
    )

    answer_inputs, _, _ = build_official_student_answer_io(
        batch,
        eot_token_id=model.eot_id,
        pad_token_id=model.pad_token_id,
    )
    endpoint_positions = batch.teacher_answer_start - batch.teacher_trace_end
    row = torch.arange(answer_inputs.shape[0], device=answer_inputs.device)
    student_tokens = answer_inputs[row, endpoint_positions]
    teacher_tokens = batch.teacher_ids[row, batch.teacher_endpoint]
    passed = (
        absolute_loss_error <= 1e-7
        and relative_error <= 1e-6
        and cosine >= 0.999999
        and torch.equal(student_tokens, teacher_tokens)
        and not teacher.requires_grad
    )
    report = {
        "schema_version": CORRECTED_ENDPOINT_TSVC_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "examples": len(indices),
        "indices": indices,
        "teacher_shape": list(teacher.shape),
        "student_shape": list(student.shape),
        "teacher_endpoint_token_ids": [int(value) for value in teacher_tokens.cpu()],
        "student_endpoint_token_ids": [int(value) for value in student_tokens.cpu()],
        "student_endpoint": "colon_after_six_latents_eot_and_answer_cue",
        "teacher_endpoint": "colon_after_explicit_cot_and_answer_cue",
        "hidden_state_convention": "embedding_plus_12_transformer_blocks",
        "reference_loss": float(reference_loss.detach().float()),
        "candidate_full_loss": float(candidate_loss.detach().float()),
        "absolute_loss_error": absolute_loss_error,
        "gradient_relative_l2_error": relative_error,
        "gradient_cosine": cosine,
        "teacher_detached": not teacher.requires_grad,
        "thresholds": {
            "absolute_loss_error_max": 1e-7,
            "gradient_relative_l2_error_max": 1e-6,
            "gradient_cosine_min": 0.999999,
        },
    }
    _atomic_json(report, output_path)
    if not passed:
        raise RuntimeError(
            "source-faithful CODI endpoint loss/gradient parity gate failed"
        )
    return report


def collect(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    settings = cfg.endpoint_tsvc_corrected
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("corrected endpoint TSV-C collection requires CUDA")
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
    hidden_states = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    if (hidden_states, hidden_size) != (13, 768):
        raise RuntimeError("corrected official endpoint TSV-C requires [13,768]")
    if args.rank != 77:
        raise RuntimeError("preregistered corrected endpoint rank must remain 77")

    metadata = {
        "analysis": "official_codi_endpoint_tsvc_corrected_calibration",
        "contract": "source_faithful_student_and_teacher_answer_colon_v2",
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
        "hidden_states": hidden_states,
        "hidden_size": hidden_size,
        "latent_positions": int(cfg.eval.latent_iterations),
        "precision": args.precision,
        "alignment": (
            "teacher and student hidden states at their exact answer-cue colon; "
            "student colon occurs after six latents and EOT"
        ),
        "loss": (
            "SmoothL1 per hidden-state tuple entry divided by unbiased teacher std, "
            "then mean across entries"
        ),
        "decomposition": "uncentered independent residual Gram eigendecomposition",
        "residual": "student_answer_colon - stop_gradient(teacher_answer_colon)",
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
            raise RuntimeError("output directory contains a different corrected request")

    scorer = OfficialCODIAnswerScorer(
        model,
        latent_positions=int(cfg.eval.latent_iterations),
    )
    parameters = [
        parameter for parameter in scorer.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("official CODI exposes no trainable parameters")
    parity = _run_native_parity_gate(
        scorer=scorer,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        indices=calibration_indices[: args.parity_examples],
        parameters=parameters,
        device=device,
        precision=args.precision,
        output_path=output_dir / "native_loss_gradient_parity.json",
    )
    metadata["native_parity_gate"] = parity

    if basis_path.is_file():
        payload = torch.load(basis_path, map_location="cpu", weights_only=False)
        if payload.get("request_sha256") != request_sha256:
            raise RuntimeError("existing corrected basis identity does not match")
        validate_corrected_bases(bases_from_state(payload["bases"]))
        observed_sha = sha256_file(basis_path)
        expected_sha = previous_manifest.get("basis_sha256") if previous_manifest else None
        if expected_sha and expected_sha != observed_sha:
            raise RuntimeError("existing corrected basis SHA256 differs")
        _atomic_json(
            {
                **metadata,
                "state": "complete",
                "processed_examples": len(calibration_indices),
                "basis_file": basis_path.name,
                "basis_sha256": observed_sha,
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
        print(f"[complete] verified corrected basis: {basis_path}")
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
            raise RuntimeError("existing corrected calibration state differs")
        moments = endpoint_moments_from_state(state["moments"])
        processed = int(state["processed_examples"])
        if calibration_indices[:processed] != state["processed_indices"]:
            raise RuntimeError("corrected calibration resume prefix changed")
        print(f"[resume] continuing corrected endpoint calibration from {processed}")
    else:
        moments = create_endpoint_moments(hidden_states, hidden_size)
        processed = 0

    progress = tqdm(
        total=len(calibration_indices),
        initial=processed,
        unit="examples",
        desc="Official CODI corrected colon TSV-C calibration",
    )
    while processed < len(calibration_indices):
        end = min(processed + args.batch_size, len(calibration_indices))
        batch_indices = calibration_indices[processed:end]
        batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in batch_indices],
            bot_token_id=model.bot_id,
        ).to(device)
        with torch.no_grad(), _amp_context(device, args.precision):
            teacher = extract_official_teacher_endpoint_targets(model, batch).all_hidden
            output = scorer(
                batch,
                return_kv=False,
                return_endpoint_hidden=False,
                return_answer_endpoint_hidden=True,
            )
        student = output.student_answer_endpoint_hidden
        if student is None:
            raise RuntimeError("corrected student colon state is missing")
        validate_corrected_endpoint_hidden(student, teacher)
        update_endpoint_moments(moments, student, teacher)
        newly_processed = end - processed
        processed = end
        progress.update(newly_processed)
        if processed < len(calibration_indices) and (
            processed % args.save_every < newly_processed
        ):
            _atomic_torch_save(
                {
                    "schema_version": CORRECTED_ENDPOINT_TSVC_SCHEMA_VERSION,
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
        del batch, teacher, student, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress.close()

    bases = fit_endpoint_tsvc_bases(
        moments,
        rank=args.rank,
        random_seed=args.random_basis_seed,
    )
    validate_corrected_bases(bases)
    basis_payload = {
        "schema_version": CORRECTED_ENDPOINT_TSVC_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "metadata": metadata,
        "bases": bases_to_state(bases),
        "top_rank_explained_energy_by_state": explained_energy(
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
    print(f"[complete] wrote corrected endpoint basis to {basis_path}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect source-faithful official-CODI colon endpoint TSV-C bases."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--calibration-examples", type=int, default=5000)
    parser.add_argument("--update-examples", type=int, default=256)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--sampling-seed", type=int, default=11)
    parser.add_argument("--random-basis-seed", type=int, default=20260803)
    parser.add_argument("--rank", type=int, default=77)
    parser.add_argument("--parity-examples", type=int, default=4)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.parity_examples <= 0 or args.parity_examples > args.calibration_examples:
        parser.error("parity examples must lie within the calibration partition")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
