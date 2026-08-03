"""Run the held-out equal-update-norm endpoint TSV-C utility screen."""
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
from torch.func import functional_call
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
)
from src.eval.official_codi import select_device
from src.eval.official_codi_endpoint_tsvc_analysis import (
    analyze_endpoint_scope,
)
from src.mech.endpoint_tsvc import (
    ENDPOINT_ARMS,
    ENDPOINT_SCOPES,
    ENDPOINT_TSVC_SCHEMA_VERSION,
    bases_from_state,
    endpoint_tsvc_loss,
    match_gradient_norm,
    validate_endpoint_bases,
)
from src.mech.kv_subspace import deterministic_derangement
from src.mech.kv_target_utility import (
    autograd_gradients,
    combine_gradients,
    gradient_inner_product,
    parameter_norm,
    updated_parameter_mapping,
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


ENDPOINT_UTILITY_RUN_SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _stateless_answer_losses(
    scorer,
    parameter_mapping: dict[str, torch.Tensor],
    batch,
    *,
    precision: str,
    device: torch.device,
) -> list[float]:
    with torch.no_grad(), _amp_context(device, precision):
        output = functional_call(
            scorer,
            parameter_mapping,
            (batch,),
            {"return_kv": False, "return_endpoint_hidden": False},
            strict=False,
        )
    return [
        float(value) for value in output.per_example_loss.detach().float().cpu()
    ]


def _completed_batch(
    path: Path,
    *,
    request_sha256: str,
    batch_index: int,
    update_indices: list[int],
    validation_indices: list[int],
) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("schema_version", -1))
        != ENDPOINT_UTILITY_RUN_SCHEMA_VERSION
        or payload.get("request_sha256") != request_sha256
        or int(payload.get("batch_index", -1)) != batch_index
        or payload.get("update_indices") != update_indices
        or payload.get("validation_indices") != validation_indices
    ):
        raise RuntimeError(f"completed endpoint batch identity mismatch: {path}")
    return payload


def _arm_loss(
    name: str,
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    scope: str,
    bases,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if name == "full":
        return endpoint_tsvc_loss(
            student,
            teacher,
            scope=scope,
            mode="full",
        )
    if name == "learned_top77":
        basis, target, mode = bases.top, teacher, "projected"
    elif name == "random_rank77":
        basis, target, mode = bases.random, teacher, "projected"
    elif name == "bottom_rank77":
        basis, target, mode = bases.bottom, teacher, "projected"
    elif name == "shuffled_top77":
        basis = bases.top
        target = teacher.index_select(0, permutation)
        mode = "projected"
    elif name == "complement":
        basis, target, mode = bases.top, teacher, "complement"
    else:
        raise ValueError(f"unknown endpoint arm {name!r}")
    return endpoint_tsvc_loss(
        student,
        target,
        scope=scope,
        mode=mode,
        basis=basis,
    )


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    if args.scope not in ENDPOINT_SCOPES:
        raise ValueError(f"scope must be one of {ENDPOINT_SCOPES}")
    if args.batch_size < 2:
        raise ValueError("batch size must be at least two for shuffled pairing")

    basis_path = args.basis
    if not basis_path.is_file():
        raise FileNotFoundError(f"endpoint basis artifact is missing: {basis_path}")
    basis_payload = torch.load(basis_path, map_location="cpu", weights_only=False)
    if int(basis_payload.get("schema_version", -1)) != ENDPOINT_TSVC_SCHEMA_VERSION:
        raise RuntimeError("endpoint basis artifact uses an incompatible schema")
    basis_metadata = basis_payload["metadata"]
    bases = bases_from_state(basis_payload["bases"])
    validate_endpoint_bases(bases, layers=12, hidden_size=768)
    if bases.rank != 77:
        raise RuntimeError("preregistered endpoint basis rank must be 77")
    partitions = basis_metadata.get("partitions", {})
    update_indices = [int(value) for value in partitions.get("update", [])]
    validation_indices = [int(value) for value in partitions.get("validation", [])]
    if len(update_indices) != len(validation_indices):
        raise RuntimeError("update and validation partitions must have equal size")
    if len(update_indices) % args.batch_size:
        raise RuntimeError("partition size must be divisible by batch size")
    if set(update_indices).intersection(validation_indices):
        raise RuntimeError("update and validation indices overlap")

    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("endpoint utility requires CUDA")
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
    if basis_metadata.get("checkpoint_sha256") != load_report.checkpoint_sha256:
        raise RuntimeError("basis and utility checkpoint hashes differ")
    model.to(device=device, dtype=dtype).eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    settings = cfg.endpoint_tsvc
    data_cfg = load_config(str(settings.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    observed_fingerprint = getattr(dataset, "_fingerprint", "unavailable")
    if basis_metadata.get("dataset_fingerprint") != observed_fingerprint:
        raise RuntimeError("basis and utility dataset fingerprints differ")

    scorer = OfficialCODIAnswerScorer(
        model,
        latent_positions=int(cfg.eval.latent_iterations),
    )
    trainable = [
        (name, parameter)
        for name, parameter in scorer.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("official CODI model exposes no trainable parameters")
    parameter_names = [name for name, _ in trainable]
    parameters = [parameter for _, parameter in trainable]
    trainable_parameter_norm = parameter_norm(parameters)
    update_norm = args.relative_update_norm * trainable_parameter_norm
    basis_sha256 = sha256_file(basis_path)

    request = {
        "schema_version": ENDPOINT_UTILITY_RUN_SCHEMA_VERSION,
        "analysis": "official_codi_endpoint_tsvc_equal_norm_utility",
        "scope": args.scope,
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "dataset_fingerprint": observed_fingerprint,
        "basis_path": str(basis_path),
        "basis_sha256": basis_sha256,
        "basis_request_sha256": basis_payload["request_sha256"],
        "rank": bases.rank,
        "update_indices": update_indices,
        "validation_indices": validation_indices,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "precision": args.precision,
        "arms": list(ENDPOINT_ARMS),
        "metric": "projected L1 with per-layer teacher-std normalization",
        "auxiliary_norm_matching": "each arm matched to full endpoint gradient",
        "base_objective": "official student gold-answer NLL",
        "relative_update_norm": args.relative_update_norm,
        "resolved_update_norm": update_norm,
        "trainable_parameter_norm": trainable_parameter_norm,
        "trainable_parameter_numel": sum(value.numel() for value in parameters),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    request_sha256 = _sha256_json(request)
    request["request_sha256"] = request_sha256
    output_dir = args.output_dir
    batches_dir = output_dir / "batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("request_sha256") != request_sha256:
            raise RuntimeError("output directory contains a different utility request")
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        **request,
        "state": "running",
        "created_at_utc": created_at,
        "completed_batches": [],
    }
    _atomic_json(manifest_path, manifest)

    total_batches = len(update_indices) // args.batch_size
    completed_payloads: list[dict] = []
    progress = tqdm(
        range(total_batches),
        unit="batch",
        desc=f"Official CODI endpoint utility ({args.scope})",
    )
    for batch_index in progress:
        start = batch_index * args.batch_size
        end = start + args.batch_size
        current_update_indices = update_indices[start:end]
        current_validation_indices = validation_indices[start:end]
        batch_path = batches_dir / f"batch_{batch_index:06d}.json"
        completed = _completed_batch(
            batch_path,
            request_sha256=request_sha256,
            batch_index=batch_index,
            update_indices=current_update_indices,
            validation_indices=current_validation_indices,
        )
        if completed is not None:
            completed_payloads.append(completed)
            progress.set_postfix_str("resumed")
            continue

        update_batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in current_update_indices],
            bot_token_id=model.bot_id,
        ).to(device)
        validation_batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in current_validation_indices],
            bot_token_id=model.bot_id,
        ).to(device)
        teacher = extract_official_teacher_endpoint_targets(
            model, update_batch
        ).hidden
        with _amp_context(device, args.precision):
            update_output = scorer(
                update_batch,
                return_kv=False,
                return_endpoint_hidden=True,
            )
            validation_output = scorer(
                validation_batch,
                return_kv=False,
                return_endpoint_hidden=False,
            )
        student = update_output.student_endpoint_hidden
        if student is None or tuple(student.shape[1:]) != (12, 768):
            raise RuntimeError("utility student endpoint must be [B,12,768]")
        if tuple(teacher.shape) != tuple(student.shape):
            raise RuntimeError("utility teacher/student endpoint shapes differ")

        base_gradients = autograd_gradients(
            update_output.mean_loss,
            parameters,
            retain_graph=True,
        )
        validation_gradients = autograd_gradients(
            validation_output.mean_loss,
            parameters,
            retain_graph=False,
        )
        original_validation_losses = [
            float(value)
            for value in validation_output.per_example_loss.detach().float().cpu()
        ]
        base_mapping, base_update = updated_parameter_mapping(
            parameter_names,
            parameters,
            base_gradients,
            update_norm=update_norm,
        )
        answer_only_losses = _stateless_answer_losses(
            scorer,
            base_mapping,
            validation_batch,
            precision=args.precision,
            device=device,
        )

        permutation_generator = torch.Generator(device="cpu").manual_seed(
            args.seed * 1_000_003 + batch_index * 10_007 + 202_608_03
        )
        permutation = deterministic_derangement(
            args.batch_size,
            generator=permutation_generator,
            device=device,
        )
        losses = {
            name: _arm_loss(
                name,
                student,
                teacher,
                scope=args.scope,
                bases=bases,
                permutation=permutation,
            )
            for name in ENDPOINT_ARMS
        }
        full_gradients = autograd_gradients(
            losses["full"],
            parameters,
            retain_graph=True,
        )
        arm_payloads = {}
        for arm_index, name in enumerate(ENDPOINT_ARMS):
            if name == "full":
                raw_gradients = full_gradients
            else:
                raw_gradients = autograd_gradients(
                    losses[name],
                    parameters,
                    retain_graph=arm_index + 1 < len(ENDPOINT_ARMS),
                )
            matched_gradients, auxiliary_norm = match_gradient_norm(
                raw_gradients,
                full_gradients,
            )
            alignment = gradient_inner_product(
                validation_gradients,
                raw_gradients,
            )
            total_gradients = combine_gradients(
                base_gradients,
                matched_gradients,
            )
            mapping, update = updated_parameter_mapping(
                parameter_names,
                parameters,
                total_gradients,
                update_norm=update_norm,
            )
            heldout_losses = _stateless_answer_losses(
                scorer,
                mapping,
                validation_batch,
                precision=args.precision,
                device=device,
            )
            arm_payloads[name] = {
                "train_endpoint_loss": float(losses[name].detach().float()),
                "gradient_alignment": alignment,
                "auxiliary_norm_matching": auxiliary_norm,
                "total_update": update,
                "validation_losses": heldout_losses,
            }
            del raw_gradients, matched_gradients, total_gradients, mapping

        payload = {
            "schema_version": ENDPOINT_UTILITY_RUN_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "scope": args.scope,
            "batch_index": batch_index,
            "update_indices": current_update_indices,
            "validation_indices": current_validation_indices,
            "derangement": [int(value) for value in permutation.cpu()],
            "update_answer_loss": float(update_output.mean_loss.detach().float()),
            "validation": {
                "original_losses": original_validation_losses,
                "answer_only_losses": answer_only_losses,
                "answer_only_update": base_update,
            },
            "arms": arm_payloads,
        }
        _atomic_json(batch_path, payload)
        completed_payloads.append(payload)
        completed_ids = [item["batch_index"] for item in completed_payloads]
        _atomic_json(
            output_dir / "progress.json",
            {
                "schema_version": ENDPOINT_UTILITY_RUN_SCHEMA_VERSION,
                "request_sha256": request_sha256,
                "scope": args.scope,
                "state": "running",
                "completed_batches": sorted(completed_ids),
                "total_batches": total_batches,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        del (
            update_batch,
            validation_batch,
            teacher,
            student,
            update_output,
            validation_output,
            base_gradients,
            validation_gradients,
            base_mapping,
            losses,
            full_gradients,
            arm_payloads,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = analyze_endpoint_scope(
        completed_payloads,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report["request"] = request
    _atomic_json(output_dir / "summary.json", report)
    completed_ids = sorted(item["batch_index"] for item in completed_payloads)
    manifest.update(
        {
            "state": "complete",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_batches": completed_ids,
            "gate_status": report["gate"]["status"],
        }
    )
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        output_dir / "progress.json",
        {
            "schema_version": ENDPOINT_UTILITY_RUN_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "scope": args.scope,
            "state": "complete",
            "completed_batches": completed_ids,
            "total_batches": total_batches,
            "gate_status": report["gate"]["status"],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[complete] scope={args.scope} gate={report['gate']['status']}")
    print(f"[complete] wrote {output_dir / 'summary.json'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official-CODI endpoint TSV-C held-out utility."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/official_codi_gpt2.yaml"),
    )
    parser.add_argument("--reproduction-summary", required=True, type=Path)
    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--scope", choices=ENDPOINT_SCOPES, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--relative-update-norm", type=float, default=1e-4)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.relative_update_norm <= 0 or args.bootstrap_samples <= 0:
        parser.error("update norm and bootstrap samples must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
