"""Test sparse, consistently answer-aligned KV gradients on official CODI.

The experiment has three fresh, question-disjoint splits:

1. calibration fits a frozen coordinate mask from repeated positive
   ``g_answer * g_kv`` contributions;
2. update batches produce full, sparse, random, shuffled, and complement gradients;
3. validation batches measure stateless equal-norm update effects on gold-answer loss.

The completed kind-level target-utility examples are excluded by normalized question.
No checkpoint parameters are overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from scripts.collect_official_codi_kv_subspaces import _verify_reproduction_gate
from scripts.run_official_codi_kv_target_utility import (
    _amp_context,
    _atomic_json,
    _atomic_text,
    _normalized_question,
    _sha256_json,
    _stateless_answer_losses,
)
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
    official_codi_answer_is_eligible,
)
from src.eval.official_codi import select_device
from src.eval.official_codi_kv_gradient_signal_analysis import (
    analyze_kv_gradient_signal_batches,
    render_kv_gradient_signal_markdown,
)
from src.mech.kv_gradient_signal import (
    GradientAlignmentAccumulator,
    mask_gradients,
    masks_to_device,
    random_mask_like,
    rescale_gradients_to_norm,
)
from src.mech.kv_subspace import deterministic_derangement
from src.mech.kv_target_utility import (
    autograd_gradients,
    build_target_groups,
    combine_gradients,
    gradient_inner_product,
    gradient_norm,
    kv_group_loss,
    parameter_norm,
    updated_parameter_mapping,
)
from src.mech.official_codi_target_utility import (
    OfficialCODIAnswerScorer,
    extract_official_teacher_kv_targets,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


KV_GRADIENT_SIGNAL_RUN_SCHEMA_VERSION = 1
CONDITIONS = (
    "full",
    "sparse_aligned",
    "random_sparse",
    "shuffled_sparse",
    "complement",
)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def verify_prior_target_utility(
    root: Path,
    *,
    cfg,
    dataset,
) -> dict:
    summary_path = root / "summary.json"
    manifest_path = root / "run_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "prior target-utility directory must contain summary.json and "
            "run_manifest.json"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_classifications = {
        "key_all": "neutral_or_inconclusive_target_family",
        "value_all": "neutral_or_inconclusive_target_family",
    }
    if summary.get("screen_status") != "no_helpful_target_family_at_this_granularity":
        raise RuntimeError("prior target-utility gate did not have the required outcome")
    if summary.get("classifications") != expected_classifications:
        raise RuntimeError("prior key/value classifications changed")
    if manifest.get("state") != "complete" or len(
        manifest.get("completed_batches", [])
    ) != 32:
        raise RuntimeError("prior target-utility kind screen is incomplete")
    if manifest.get("granularity") != "kind":
        raise RuntimeError("prior target-utility artifact is not kind-level")
    if manifest.get("checkpoint_revision") != str(cfg.checkpoint.revision):
        raise RuntimeError("prior target-utility checkpoint revision changed")
    if manifest.get("dataset_fingerprint") != getattr(
        dataset,
        "_fingerprint",
        "unavailable",
    ):
        raise RuntimeError("prior target-utility dataset fingerprint changed")
    excluded_indices = list(manifest["discovery_indices"]) + list(
        manifest["validation_indices"]
    )
    excluded_questions = sorted(
        {
            _normalized_question(dataset[index]["question"])
            for index in excluded_indices
        }
    )
    return {
        "root": str(root),
        "summary_sha256": _sha256_json(summary),
        "request_sha256": manifest["request_sha256"],
        "excluded_indices": excluded_indices,
        "excluded_normalized_questions": excluded_questions,
        "classifications": summary["classifications"],
        "screen_status": summary["screen_status"],
    }


def sample_three_group_disjoint_splits(
    dataset,
    *,
    examples_per_split: int,
    seed: int,
    excluded_normalized_questions: set[str],
) -> tuple[list[int], list[int], list[int], dict]:
    if examples_per_split <= 0:
        raise ValueError("examples_per_split must be positive")
    groups: dict[str, list[int]] = {}
    for index, (question, answer) in enumerate(
        zip(dataset["question"], dataset["answer"])
    ):
        normalized = _normalized_question(question)
        if (
            normalized in excluded_normalized_questions
            or not official_codi_answer_is_eligible(answer)
        ):
            continue
        groups.setdefault(normalized, []).append(index)
    required = 3 * examples_per_split
    if len(groups) < required:
        raise ValueError(
            f"need {required} fresh unique question groups, found {len(groups)}"
        )
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    selected = [generator.choice(groups[key]) for key in keys[:required]]
    calibration = selected[:examples_per_split]
    update = selected[examples_per_split : 2 * examples_per_split]
    validation = selected[2 * examples_per_split :]
    normalized_splits = [
        {_normalized_question(dataset[index]["question"]) for index in values}
        for values in (calibration, update, validation)
    ]
    if (
        normalized_splits[0].intersection(normalized_splits[1])
        or normalized_splits[0].intersection(normalized_splits[2])
        or normalized_splits[1].intersection(normalized_splits[2])
    ):
        raise RuntimeError("calibration/update/validation question groups overlap")
    return calibration, update, validation, {
        "eligible_fresh_unique_question_groups": len(groups),
        "excluded_prior_question_groups": len(excluded_normalized_questions),
        "selection": "three_fresh_seeded_question_disjoint_splits_v1",
    }


def _fit_or_load_masks(
    *,
    artifact_path: Path,
    request_sha256: str,
    model,
    scorer,
    tokenizer,
    dataset,
    calibration_indices: list[int],
    batch_size: int,
    parameters: list[torch.Tensor],
    groups: dict,
    latent_positions: int,
    importance_weight: float,
    metric: str,
    sparsity: float,
    minimum_positive_fraction: float,
    random_mask_seed: int,
    precision: str,
    device: torch.device,
) -> dict:
    if artifact_path.is_file():
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
        if artifact.get("request_sha256") != request_sha256:
            raise RuntimeError("existing gradient-mask artifact request changed")
        print(f"[mask] restored {artifact_path}")
        return artifact

    accumulators = {
        kind: GradientAlignmentAccumulator.from_parameters(parameters)
        for kind in groups
    }
    total_batches = len(calibration_indices) // batch_size
    progress = tqdm(
        range(total_batches),
        unit="batch",
        desc="Fit answer-aligned KV coordinate masks",
    )
    for batch_index in progress:
        start = batch_index * batch_size
        indices = calibration_indices[start : start + batch_size]
        batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in indices],
            bot_token_id=model.bot_id,
        ).to(device)
        teacher_targets = extract_official_teacher_kv_targets(
            model,
            batch,
            latent_positions=latent_positions,
            importance_weight=importance_weight,
        )
        with _amp_context(device, precision):
            output = scorer(batch, return_kv=True)
            losses = {
                "key": kv_group_loss(
                    output.student_keys,
                    teacher_targets.keys,
                    teacher_targets.mask,
                    groups["key"],
                    metric=metric,
                ),
                "value": kv_group_loss(
                    output.student_values,
                    teacher_targets.values,
                    teacher_targets.mask,
                    groups["value"],
                    metric=metric,
                ),
            }
        answer_gradients = autograd_gradients(
            output.mean_loss,
            parameters,
            retain_graph=True,
        )
        for index, kind in enumerate(("key", "value")):
            gradients = autograd_gradients(
                losses[kind],
                parameters,
                retain_graph=index == 0,
            )
            accumulators[kind].update(answer_gradients, gradients)
            del gradients
        del batch, teacher_targets, output, losses, answer_gradients
        if device.type == "cuda":
            torch.cuda.empty_cache()

    masks = {}
    summaries = {}
    random_masks = {}
    for kind in ("key", "value"):
        kind_masks, summary = accumulators[kind].build_mask(
            sparsity=sparsity,
            minimum_positive_fraction=minimum_positive_fraction,
        )
        masks[kind] = tuple(mask.cpu() for mask in kind_masks)
        summaries[kind] = summary
        random_masks[kind] = random_mask_like(
            masks[kind],
            selected_coordinates=summary["selected_coordinates"],
            seed=random_mask_seed + (0 if kind == "key" else 1),
        )
    artifact = {
        "schema_version": KV_GRADIENT_SIGNAL_RUN_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "state": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "masks": masks,
        "random_masks": random_masks,
        "summaries": summaries,
    }
    _atomic_torch_save(artifact_path, artifact)
    _atomic_json(
        artifact_path.with_suffix(".json"),
        {
            key: value
            for key, value in artifact.items()
            if key not in {"masks", "random_masks"}
        },
    )
    print(f"[mask] wrote {artifact_path}")
    del accumulators
    return artifact


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
        payload.get("schema_version") != KV_GRADIENT_SIGNAL_RUN_SCHEMA_VERSION
        or payload.get("request_sha256") != request_sha256
        or payload.get("batch_index") != batch_index
        or payload.get("update_indices") != update_indices
        or payload.get("validation_indices") != validation_indices
    ):
        raise RuntimeError(f"completed gradient-signal batch changed: {path}")
    return payload


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = _verify_reproduction_gate(args.reproduction_summary, cfg)
    if args.examples_per_split % args.batch_size:
        raise ValueError("examples_per_split must be divisible by batch_size")
    if args.batch_size < 2:
        raise ValueError("batch size must be at least two")
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("KV gradient signal test requires CUDA")
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None

    data_cfg = load_config(str(cfg.kv_subspace.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    prior = verify_prior_target_utility(
        args.prior_target_utility_dir,
        cfg=cfg,
        dataset=dataset,
    )
    calibration_indices, update_indices, validation_indices, sampling = (
        sample_three_group_disjoint_splits(
            dataset,
            examples_per_split=args.examples_per_split,
            seed=args.seed,
            excluded_normalized_questions=set(
                prior["excluded_normalized_questions"]
            ),
        )
    )

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
    model.codi.get_base_model().config._attn_implementation = "eager"
    model.to(device=device, dtype=dtype).eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    latent_positions = int(cfg.kv_subspace.latent_positions)
    target_groups = build_target_groups(
        granularity="kind",
        layer_count=int(model.config.num_hidden_layers),
        position_count=latent_positions,
        kinds=("key", "value"),
        positions=range(latent_positions),
    )
    groups = {group.kind: group for group in target_groups}
    scorer = OfficialCODIAnswerScorer(model, latent_positions=latent_positions)
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

    request = {
        "schema_version": KV_GRADIENT_SIGNAL_RUN_SCHEMA_VERSION,
        "analysis": "official_codi_sparse_answer_aligned_kv_gradient_signal",
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "prior_target_utility": prior,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", "unavailable"),
        "examples_per_split": args.examples_per_split,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "calibration_indices": calibration_indices,
        "update_indices": update_indices,
        "validation_indices": validation_indices,
        "sampling": sampling,
        "kinds": ["key", "value"],
        "positions": list(range(latent_positions)),
        "metric": args.metric,
        "kv_weight": args.kv_weight,
        "sparsity": args.sparsity,
        "minimum_positive_fraction": args.minimum_positive_fraction,
        "random_mask_seed": args.random_mask_seed,
        "relative_update_norm": args.relative_update_norm,
        "resolved_update_norm": update_norm,
        "auxiliary_energy_matching": "all components matched to full paired KV gradient",
        "precision": args.precision,
        "trainable_parameter_names": parameter_names,
        "trainable_parameter_shapes": [
            list(parameter.shape) for parameter in parameters
        ],
        "conditions": list(CONDITIONS),
        "primary_kind": args.primary_kind,
    }
    request_sha256 = _sha256_json(request)
    request["request_sha256"] = request_sha256
    output_dir = args.output_dir
    batches_dir = output_dir / "batches"
    artifact_path = output_dir / "mask_artifact.pt"
    manifest_path = output_dir / "run_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("request_sha256") != request_sha256:
            raise RuntimeError("output directory contains a different request")
    manifest = {
        **request,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "mask_fitting",
        "completed_batches": [],
    }
    _atomic_json(manifest_path, manifest)

    artifact = _fit_or_load_masks(
        artifact_path=artifact_path,
        request_sha256=request_sha256,
        model=model,
        scorer=scorer,
        tokenizer=tokenizer,
        dataset=dataset,
        calibration_indices=calibration_indices,
        batch_size=args.batch_size,
        parameters=parameters,
        groups=groups,
        latent_positions=latent_positions,
        importance_weight=float(cfg.kv_subspace.importance_weight),
        metric=args.metric,
        sparsity=args.sparsity,
        minimum_positive_fraction=args.minimum_positive_fraction,
        random_mask_seed=args.random_mask_seed,
        precision=args.precision,
        device=device,
    )
    learned_masks = {
        kind: masks_to_device(artifact["masks"][kind], device)
        for kind in ("key", "value")
    }
    random_masks = {
        kind: masks_to_device(artifact["random_masks"][kind], device)
        for kind in ("key", "value")
    }
    manifest["state"] = "evaluating"
    manifest["mask_summaries"] = artifact["summaries"]
    _atomic_json(manifest_path, manifest)

    total_batches = args.examples_per_split // args.batch_size
    completed_payloads = []
    progress = tqdm(
        range(total_batches),
        unit="batch",
        desc="Official CODI sparse KV gradient signal",
    )
    for batch_index in progress:
        start = batch_index * args.batch_size
        end = start + args.batch_size
        update_batch_indices = update_indices[start:end]
        validation_batch_indices = validation_indices[start:end]
        batch_path = batches_dir / f"batch_{batch_index:06d}.json"
        completed = _completed_batch(
            batch_path,
            request_sha256=request_sha256,
            batch_index=batch_index,
            update_indices=update_batch_indices,
            validation_indices=validation_batch_indices,
        )
        if completed is not None:
            completed_payloads.append(completed)
            progress.set_postfix_str("resumed")
            continue

        update_batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in update_batch_indices],
            bot_token_id=model.bot_id,
        ).to(device)
        validation_batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in validation_batch_indices],
            bot_token_id=model.bot_id,
        ).to(device)
        teacher_targets = extract_official_teacher_kv_targets(
            model,
            update_batch,
            latent_positions=latent_positions,
            importance_weight=float(cfg.kv_subspace.importance_weight),
        )
        with _amp_context(device, args.precision):
            update_output = scorer(update_batch, return_kv=True)
        with torch.no_grad(), _amp_context(device, args.precision):
            validation_output = scorer(validation_batch, return_kv=False)
        original_validation_losses = [
            float(value)
            for value in validation_output.per_example_loss.detach().float().cpu()
        ]
        answer_gradients = autograd_gradients(
            update_output.mean_loss,
            parameters,
            retain_graph=True,
        )
        base_mapping, base_update = updated_parameter_mapping(
            parameter_names,
            parameters,
            answer_gradients,
            update_norm=update_norm,
        )
        no_target_losses = _stateless_answer_losses(
            scorer,
            base_mapping,
            validation_batch,
            precision=args.precision,
            device=device,
        )

        generator = torch.Generator(device="cpu").manual_seed(
            args.seed * 1_000_003 + batch_index * 10_007 + 424_243
        )
        permutation = deterministic_derangement(
            args.batch_size,
            generator=generator,
            device=device,
        )
        loss_specs = []
        for kind in ("key", "value"):
            student = (
                update_output.student_keys
                if kind == "key"
                else update_output.student_values
            )
            teacher = (
                teacher_targets.keys
                if kind == "key"
                else teacher_targets.values
            )
            loss_specs.extend(
                [
                    (
                        kind,
                        "paired",
                        kv_group_loss(
                            student,
                            teacher,
                            teacher_targets.mask,
                            groups[kind],
                            metric=args.metric,
                        ),
                    ),
                    (
                        kind,
                        "shuffled",
                        kv_group_loss(
                            student,
                            teacher.index_select(0, permutation),
                            teacher_targets.mask.index_select(0, permutation),
                            groups[kind],
                            metric=args.metric,
                        ),
                    ),
                ]
            )
        gradient_values = {}
        train_losses = {}
        for index, (kind, pairing, loss) in enumerate(loss_specs):
            gradient_values[(kind, pairing)] = autograd_gradients(
                loss,
                parameters,
                retain_graph=index + 1 < len(loss_specs),
            )
            train_losses[(kind, pairing)] = float(loss.detach().float())

        kind_payloads = {}
        for kind in ("key", "value"):
            full_gradient = gradient_values[(kind, "paired")]
            full_norm = gradient_norm(full_gradient)
            raw_components = {
                "full": full_gradient,
                "sparse_aligned": mask_gradients(
                    full_gradient,
                    learned_masks[kind],
                ),
                "random_sparse": mask_gradients(
                    full_gradient,
                    random_masks[kind],
                ),
                "shuffled_sparse": mask_gradients(
                    gradient_values[(kind, "shuffled")],
                    learned_masks[kind],
                ),
                "complement": mask_gradients(
                    full_gradient,
                    learned_masks[kind],
                    complement=True,
                ),
            }
            conditions = {}
            for condition in CONDITIONS:
                component, energy_match = rescale_gradients_to_norm(
                    raw_components[condition],
                    target_norm=full_norm,
                )
                alignment = gradient_inner_product(
                    answer_gradients,
                    component,
                )
                combined = combine_gradients(
                    answer_gradients,
                    component,
                    auxiliary_weight=args.kv_weight,
                )
                mapping, total_update = updated_parameter_mapping(
                    parameter_names,
                    parameters,
                    combined,
                    update_norm=update_norm,
                )
                conditions[condition] = {
                    "validation_losses": _stateless_answer_losses(
                        scorer,
                        mapping,
                        validation_batch,
                        precision=args.precision,
                        device=device,
                    ),
                    "gradient_alignment": alignment,
                    "auxiliary_energy_match": energy_match,
                    "total_update": total_update,
                }
                del component, combined, mapping
            kind_payloads[kind] = {
                "candidate_train_loss": train_losses[(kind, "paired")],
                "shuffled_train_loss": train_losses[(kind, "shuffled")],
                "full_auxiliary_gradient_norm": full_norm,
                "conditions": conditions,
            }

        payload = {
            "schema_version": KV_GRADIENT_SIGNAL_RUN_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "batch_index": batch_index,
            "update_indices": update_batch_indices,
            "validation_indices": validation_batch_indices,
            "derangement": [int(value) for value in permutation.cpu()],
            "validation": {
                "original_losses": original_validation_losses,
                "no_target_losses": no_target_losses,
                "no_target_update": base_update,
            },
            "kinds": kind_payloads,
        }
        _atomic_json(batch_path, payload)
        completed_payloads.append(payload)
        del (
            update_batch,
            validation_batch,
            teacher_targets,
            update_output,
            validation_output,
            answer_gradients,
            base_mapping,
            loss_specs,
            gradient_values,
            train_losses,
            kind_payloads,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = analyze_kv_gradient_signal_batches(
        completed_payloads,
        primary_kind=args.primary_kind,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report["mask_summaries"] = artifact["summaries"]
    report["request"] = request
    _atomic_json(output_dir / "summary.json", report)
    _atomic_text(
        output_dir / "report.md",
        render_kv_gradient_signal_markdown(report),
    )
    manifest["state"] = "complete"
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_batches"] = list(range(total_batches))
    manifest["gate"] = report["gate"]
    _atomic_json(manifest_path, manifest)
    print(f"[complete] gate={report['gate']}")
    print(f"[complete] wrote {output_dir / 'report.md'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test sparse answer-aligned official-CODI KV gradients."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/official_codi_gpt2.yaml"),
    )
    parser.add_argument("--reproduction-summary", required=True, type=Path)
    parser.add_argument("--prior-target-utility-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--examples-per-split", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--metric", choices=("l1", "mse", "smooth_l1"), default="l1")
    parser.add_argument("--kv-weight", type=float, default=1.0)
    parser.add_argument("--sparsity", type=float, default=0.05)
    parser.add_argument("--minimum-positive-fraction", type=float, default=0.60)
    parser.add_argument("--random-mask-seed", type=int, default=20260729)
    parser.add_argument("--relative-update-norm", type=float, default=1e-4)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--primary-kind", choices=("key", "value"), default="key")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
