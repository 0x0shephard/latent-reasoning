"""Held-out equal-update-norm utility for parameter-aware CODI endpoint PCs."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_official_codi_endpoint_tsvc_corrected_utility import (
    _amp_context,
    _atomic_json,
    _sha256_json,
    _stateless_answer_losses,
)
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
)
from src.eval.official_codi import select_device
from src.eval.official_codi_endpoint_parameter_aware_analysis import (
    analyze_parameter_aware_utility,
)
from src.mech.endpoint_parameter_aware import (
    PARAMETER_AWARE_ARMS,
    PARAMETER_AWARE_SCHEMA_VERSION,
    PARAMETER_AWARE_SCOPE,
    PARAMETER_AWARE_UTILITY_SCHEMA_VERSION,
    parameter_aware_bases_from_state,
    parameter_aware_endpoint_loss,
    validate_parameter_aware_bases,
)
from src.mech.endpoint_tsvc import match_gradient_norm
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
        int(payload.get("schema_version", -1)) != PARAMETER_AWARE_UTILITY_SCHEMA_VERSION
        or payload.get("request_sha256") != request_sha256
        or int(payload.get("batch_index", -1)) != batch_index
        or payload.get("update_indices") != update_indices
        or payload.get("validation_indices") != validation_indices
    ):
        raise RuntimeError(f"completed parameter-aware batch mismatch: {path}")
    return payload


def _arm_loss(
    name: str,
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    bases,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if name == "full_blocks":
        return parameter_aware_endpoint_loss(student, teacher, mode="full_blocks")
    if name == "parameter_aware":
        basis, target, mode = bases.parameter_aware, teacher, "projected"
    elif name == "energy_rank_matched":
        basis, target, mode = bases.energy, teacher, "projected"
    elif name == "random_rank_matched":
        basis, target, mode = bases.random, teacher, "projected"
    elif name == "shuffled_answer_rank_matched":
        basis, target, mode = bases.shuffled_answer, teacher, "projected"
    elif name == "shuffled_teacher":
        basis = bases.parameter_aware
        target = teacher.index_select(0, permutation)
        mode = "projected"
    elif name == "complement":
        basis, target, mode = bases.parameter_aware, teacher, "complement"
    else:
        raise ValueError(f"unknown parameter-aware arm {name!r}")
    return parameter_aware_endpoint_loss(
        student,
        target,
        mode=mode,
        basis=basis,
        ranks=bases.ranks,
    )


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    if args.batch_size < 2:
        raise ValueError("utility batch size must be at least two")

    basis_payload = torch.load(args.basis, map_location="cpu", weights_only=False)
    if int(basis_payload.get("schema_version", -1)) != PARAMETER_AWARE_SCHEMA_VERSION:
        raise RuntimeError("basis is not a parameter-aware endpoint artifact")
    metadata = basis_payload["metadata"]
    if metadata.get("contract") != "parameter_aware_colon_final_two_blocks_v1":
        raise RuntimeError("parameter-aware endpoint contract changed")
    if metadata.get("native_parity_gate", {}).get("status") != "passed":
        raise RuntimeError("parameter-aware basis lacks native parity")
    if basis_payload.get("selection", {}).get("status") != "candidate_selected":
        raise RuntimeError("no stable parameter-aware candidate was selected")
    bases = parameter_aware_bases_from_state(basis_payload["bases"])
    validate_parameter_aware_bases(bases, require_candidate=True)

    partitions = metadata.get("partitions", {})
    update_indices = [int(value) for value in partitions.get("update", [])]
    validation_indices = [int(value) for value in partitions.get("validation", [])]
    if len(update_indices) != len(validation_indices) or not update_indices:
        raise RuntimeError("parameter-aware utility partitions are invalid")
    if len(update_indices) % args.batch_size:
        raise RuntimeError("utility partition must be divisible by batch size")
    if set(update_indices).intersection(validation_indices):
        raise RuntimeError("utility partitions overlap")

    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("parameter-aware utility requires CUDA")
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
    if metadata.get("checkpoint_sha256") != load_report.checkpoint_sha256:
        raise RuntimeError("basis and utility checkpoint hashes differ")
    model.to(device=device, dtype=dtype).eval()
    bases = replace(
        bases,
        parameter_aware=bases.parameter_aware.to(device=device, dtype=dtype),
        energy=bases.energy.to(device=device, dtype=dtype),
        random=bases.random.to(device=device, dtype=dtype),
        shuffled_answer=bases.shuffled_answer.to(device=device, dtype=dtype),
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_cfg = load_config(str(cfg.endpoint_parameter_aware.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    fingerprint = getattr(dataset, "_fingerprint", "unavailable")
    if metadata.get("dataset_fingerprint") != fingerprint:
        raise RuntimeError("basis and utility dataset fingerprints differ")
    scorer = OfficialCODIAnswerScorer(
        model, latent_positions=int(cfg.eval.latent_iterations)
    )
    trainable = [
        (name, parameter)
        for name, parameter in scorer.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("official CODI exposes no trainable parameters")
    parameter_names = [name for name, _ in trainable]
    parameters = [parameter for _, parameter in trainable]
    trainable_norm = parameter_norm(parameters)
    update_norm = args.relative_update_norm * trainable_norm
    basis_sha256 = sha256_file(args.basis)
    request = {
        "schema_version": PARAMETER_AWARE_UTILITY_SCHEMA_VERSION,
        "analysis": "official_codi_endpoint_parameter_aware_equal_norm_utility",
        "contract": metadata["contract"],
        "scope": PARAMETER_AWARE_SCOPE,
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "native_parity_gate": metadata["native_parity_gate"],
        "dataset_fingerprint": fingerprint,
        "basis_path": str(args.basis),
        "basis_sha256": basis_sha256,
        "basis_request_sha256": basis_payload["request_sha256"],
        "selection": basis_payload["selection"],
        "rank_by_state": [int(value) for value in bases.ranks.tolist()],
        "update_indices": update_indices,
        "validation_indices": validation_indices,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "precision": args.precision,
        "arms": list(PARAMETER_AWARE_ARMS),
        "auxiliary_norm_matching": "every arm matched to full block-target gradient",
        "base_objective": "official student gold-answer NLL",
        "relative_update_norm": args.relative_update_norm,
        "resolved_update_norm": update_norm,
        "trainable_parameter_norm": trainable_norm,
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
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_sha256:
            raise RuntimeError("utility output contains a different request")
    manifest = {
        **request,
        "state": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_batches": [],
    }
    _atomic_json(manifest_path, manifest)

    total_batches = len(update_indices) // args.batch_size
    completed_payloads = []
    progress = tqdm(
        range(total_batches), unit="batch", desc="Parameter-aware endpoint utility"
    )
    for batch_index in progress:
        start = batch_index * args.batch_size
        end = start + args.batch_size
        current_update = update_indices[start:end]
        current_validation = validation_indices[start:end]
        batch_path = batches_dir / f"batch_{batch_index:06d}.json"
        completed = _completed_batch(
            batch_path,
            request_sha256=request_sha256,
            batch_index=batch_index,
            update_indices=current_update,
            validation_indices=current_validation,
        )
        if completed is not None:
            completed_payloads.append(completed)
            progress.set_postfix_str("resumed")
            continue

        update_batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in current_update],
            bot_token_id=model.bot_id,
        ).to(device)
        validation_batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in current_validation],
            bot_token_id=model.bot_id,
        ).to(device)
        teacher = extract_official_teacher_endpoint_targets(model, update_batch).all_hidden
        with _amp_context(device, args.precision):
            update_output = scorer(update_batch, return_answer_endpoint_hidden=True)
            validation_output = scorer(validation_batch)
        student = update_output.student_answer_endpoint_hidden
        if student is None:
            raise RuntimeError("student endpoint is missing during utility")
        base_gradients = autograd_gradients(
            update_output.mean_loss, parameters, retain_graph=True
        )
        validation_gradients = autograd_gradients(
            validation_output.mean_loss, parameters, retain_graph=False
        )
        original_validation_losses = [
            float(value)
            for value in validation_output.per_example_loss.detach().float().cpu()
        ]
        base_mapping, base_update = updated_parameter_mapping(
            parameter_names, parameters, base_gradients, update_norm=update_norm
        )
        answer_only_losses = _stateless_answer_losses(
            scorer,
            base_mapping,
            validation_batch,
            precision=args.precision,
            device=device,
        )
        generator = torch.Generator(device="cpu").manual_seed(
            args.seed * 1_000_003 + batch_index * 10_007 + 202_608_05
        )
        permutation = deterministic_derangement(
            args.batch_size, generator=generator, device=device
        )
        losses = {
            name: _arm_loss(
                name,
                student,
                teacher,
                bases=bases,
                permutation=permutation,
            )
            for name in PARAMETER_AWARE_ARMS
        }
        full_gradients = autograd_gradients(
            losses["full_blocks"], parameters, retain_graph=True
        )
        arm_payloads = {}
        for arm_index, name in enumerate(PARAMETER_AWARE_ARMS):
            raw_gradients = (
                full_gradients
                if name == "full_blocks"
                else autograd_gradients(
                    losses[name],
                    parameters,
                    retain_graph=arm_index + 1 < len(PARAMETER_AWARE_ARMS),
                )
            )
            matched_gradients, auxiliary_norm = match_gradient_norm(
                raw_gradients, full_gradients
            )
            alignment = gradient_inner_product(validation_gradients, raw_gradients)
            total_gradients = combine_gradients(base_gradients, matched_gradients)
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
            "schema_version": PARAMETER_AWARE_UTILITY_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "scope": PARAMETER_AWARE_SCOPE,
            "batch_index": batch_index,
            "update_indices": current_update,
            "validation_indices": current_validation,
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
        completed_ids = sorted(value["batch_index"] for value in completed_payloads)
        _atomic_json(
            output_dir / "progress.json",
            {
                "schema_version": PARAMETER_AWARE_UTILITY_SCHEMA_VERSION,
                "request_sha256": request_sha256,
                "state": "running",
                "completed_batches": completed_ids,
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

    report = analyze_parameter_aware_utility(
        completed_payloads,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report["request"] = request
    _atomic_json(output_dir / "summary.json", report)
    completed_ids = sorted(value["batch_index"] for value in completed_payloads)
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
            "schema_version": PARAMETER_AWARE_UTILITY_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "state": "complete",
            "completed_batches": completed_ids,
            "total_batches": total_batches,
            "gate_status": report["gate"]["status"],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[complete] gate={report['gate']['status']}")
    print(f"[complete] wrote {output_dir}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run parameter-aware official-CODI endpoint utility."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--relative-update-norm", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.relative_update_norm <= 0 or args.bootstrap_samples <= 0:
        parser.error("update norm and bootstrap samples must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
