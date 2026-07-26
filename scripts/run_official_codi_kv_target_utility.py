"""Run a hierarchical marginal-utility screen for official-CODI KV targets.

Each correctly paired teacher target is compared with:

1. an answer-loss-only update, and
2. an equal-norm answer plus shuffled-target update.

All three updates are evaluated on a disjoint batch and are applied statelessly with
``torch.func.functional_call``.  The author checkpoint is never overwritten.
"""
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
from torch.func import functional_call
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_official_codi_kv_subspaces import _verify_reproduction_gate
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
    official_codi_answer_is_eligible,
)
from src.eval.official_codi import select_device
from src.eval.official_codi_kv_target_utility_analysis import (
    analyze_target_utility_batches,
    render_target_utility_markdown,
)
from src.mech.kv_subspace import deterministic_derangement
from src.mech.kv_target_utility import (
    TARGET_GRANULARITIES,
    autograd_gradients,
    build_target_groups,
    combine_gradients,
    gradient_inner_product,
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


TARGET_UTILITY_RUN_SCHEMA_VERSION = 1
_NORMALIZE_SPACE = re.compile(r"\s+")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha256_json(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value cannot be empty")
    return values


def _parse_positions(value: str) -> list[int]:
    try:
        values = sorted(
            {int(item.strip()) for item in value.split(",") if item.strip()}
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "positions must be comma-separated integers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("positions cannot be empty")
    return values


def _normalized_question(value: object) -> str:
    return _NORMALIZE_SPACE.sub(" ", str(value).strip().casefold())


def sample_group_disjoint_indices(
    dataset,
    *,
    examples_per_split: int,
    seed: int,
) -> tuple[list[int], list[int], dict]:
    """Select one eligible row per normalized question for two disjoint splits."""
    if examples_per_split <= 0:
        raise ValueError("examples_per_split must be positive")
    groups: dict[str, list[int]] = {}
    for index, (question, answer) in enumerate(
        zip(dataset["question"], dataset["answer"])
    ):
        if not official_codi_answer_is_eligible(answer):
            continue
        groups.setdefault(_normalized_question(question), []).append(index)
    required = 2 * examples_per_split
    if len(groups) < required:
        raise ValueError(
            f"need {required} unique eligible questions, found {len(groups)}"
        )
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    selected = [
        generator.choice(groups[key])
        for key in keys[:required]
    ]
    discovery = selected[:examples_per_split]
    validation = selected[examples_per_split:]
    if {
        _normalized_question(dataset[index]["question"]) for index in discovery
    }.intersection(
        _normalized_question(dataset[index]["question"]) for index in validation
    ):
        raise RuntimeError("question groups overlap across target-utility splits")
    return discovery, validation, {
        "eligible_unique_question_groups": len(groups),
        "selection": "one_seeded_row_per_normalized_question_v1",
    }


def _amp_context(device: torch.device, precision: str):
    normalized = precision.casefold()
    if normalized in {"float32", "fp32"} or device.type != "cuda":
        return nullcontext()
    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
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
            {"return_kv": False},
            strict=False,
        )
    return [
        float(value)
        for value in output.per_example_loss.detach().float().cpu()
    ]


def _completed_batch(
    path: Path,
    *,
    request_sha256: str,
    batch_index: int,
    discovery_indices: list[int],
    validation_indices: list[int],
) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(payload.get("schema_version", -1))
        != TARGET_UTILITY_RUN_SCHEMA_VERSION
        or payload.get("request_sha256") != request_sha256
        or int(payload.get("batch_index", -1)) != batch_index
        or payload.get("discovery_indices") != discovery_indices
        or payload.get("validation_indices") != validation_indices
    ):
        raise RuntimeError(f"completed batch identity mismatch: {path}")
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
        raise ValueError("batch_size must be at least two for shuffled controls")
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "KV target utility requires CUDA; --allow-cpu is for tiny tests only"
        )
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None

    data_cfg = load_config(str(cfg.kv_subspace.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    discovery_indices, validation_indices, sampling = (
        sample_group_disjoint_indices(
            dataset,
            examples_per_split=args.examples_per_split,
            seed=args.seed,
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
    official_backbone = model.codi.get_base_model()
    official_backbone.config._attn_implementation = "eager"
    model.to(device=device, dtype=dtype).eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    latent_positions = int(cfg.kv_subspace.latent_positions)
    layer_count = int(model.config.num_hidden_layers)
    groups = build_target_groups(
        granularity=args.granularity,
        layer_count=layer_count,
        position_count=latent_positions,
        kinds=args.kinds,
        positions=args.positions,
    )
    scorer = OfficialCODIAnswerScorer(
        model,
        latent_positions=latent_positions,
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
    trainable_numel = sum(parameter.numel() for parameter in parameters)
    trainable_parameter_norm = parameter_norm(parameters)
    update_norm = args.relative_update_norm * trainable_parameter_norm
    if update_norm <= 0.0:
        raise RuntimeError("resolved equal update norm is not positive")

    request = {
        "schema_version": TARGET_UTILITY_RUN_SCHEMA_VERSION,
        "analysis": "official_codi_hierarchical_kv_target_marginal_utility",
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", "unavailable"),
        "examples_per_split": args.examples_per_split,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "discovery_indices": discovery_indices,
        "validation_indices": validation_indices,
        "sampling": sampling,
        "granularity": args.granularity,
        "target_groups": [group.to_dict() for group in groups],
        "metric": args.metric,
        "kv_weight": args.kv_weight,
        "base_objective": "official_student_gold_answer_nll",
        "relative_update_norm": args.relative_update_norm,
        "resolved_update_norm": update_norm,
        "trainable_parameter_norm": trainable_parameter_norm,
        "trainable_parameter_numel": trainable_numel,
        "trainable_parameter_names": parameter_names,
        "precision": args.precision,
        "latent_positions": latent_positions,
        "importance_weight": float(cfg.kv_subspace.importance_weight),
        "null": "within-batch deterministic teacher-target derangement",
    }
    request_sha256 = _sha256_json(request)
    request["request_sha256"] = request_sha256
    output_dir = args.output_dir
    batches_dir = output_dir / "batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("request_sha256") != request_sha256:
            raise RuntimeError(
                "output directory contains a different target-utility request"
            )
    manifest = {
        **request,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "running",
        "completed_batches": [],
    }
    _atomic_json(manifest_path, manifest)

    total_batches = args.examples_per_split // args.batch_size
    progress = tqdm(
        range(total_batches),
        unit="batch",
        desc=f"Official CODI KV utility ({args.granularity})",
    )
    completed_payloads: list[dict] = []
    for batch_index in progress:
        start = batch_index * args.batch_size
        end = start + args.batch_size
        discovery_batch_indices = discovery_indices[start:end]
        validation_batch_indices = validation_indices[start:end]
        batch_path = batches_dir / f"batch_{batch_index:06d}.json"
        completed = _completed_batch(
            batch_path,
            request_sha256=request_sha256,
            batch_index=batch_index,
            discovery_indices=discovery_batch_indices,
            validation_indices=validation_batch_indices,
        )
        if completed is not None:
            completed_payloads.append(completed)
            progress.set_postfix_str("resumed")
            continue

        discovery_rows = [
            dataset[index] for index in discovery_batch_indices
        ]
        validation_rows = [
            dataset[index] for index in validation_batch_indices
        ]
        discovery_batch = collate_official_codi_kv_rows(
            tokenizer,
            discovery_rows,
            bot_token_id=model.bot_id,
        ).to(device)
        validation_batch = collate_official_codi_kv_rows(
            tokenizer,
            validation_rows,
            bot_token_id=model.bot_id,
        ).to(device)
        teacher_targets = extract_official_teacher_kv_targets(
            model,
            discovery_batch,
            latent_positions=latent_positions,
            importance_weight=float(cfg.kv_subspace.importance_weight),
        )
        with _amp_context(device, args.precision):
            discovery_output = scorer(discovery_batch, return_kv=True)
            validation_output = scorer(validation_batch, return_kv=False)
        if (
            discovery_output.student_keys is None
            or discovery_output.student_values is None
        ):
            raise RuntimeError("student KV targets were not retained")
        base_gradients = autograd_gradients(
            discovery_output.mean_loss,
            parameters,
            retain_graph=True,
        )
        validation_answer_gradients = autograd_gradients(
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
        no_target_losses = _stateless_answer_losses(
            scorer,
            base_mapping,
            validation_batch,
            precision=args.precision,
            device=device,
        )

        permutation_generator = torch.Generator(device="cpu").manual_seed(
            args.seed * 1_000_003 + batch_index * 10_007 + 91_919
        )
        permutation = deterministic_derangement(
            args.batch_size,
            generator=permutation_generator,
            device=device,
        )
        group_payloads: dict[str, dict] = {}
        for group_index, group in enumerate(groups):
            if group.kind == "key":
                student_values = discovery_output.student_keys
                teacher_values = teacher_targets.keys
            else:
                student_values = discovery_output.student_values
                teacher_values = teacher_targets.values
            candidate_loss = kv_group_loss(
                student_values,
                teacher_values,
                teacher_targets.mask,
                group,
                metric=args.metric,
            )
            shuffled_loss = kv_group_loss(
                student_values,
                teacher_values.index_select(0, permutation),
                teacher_targets.mask.index_select(0, permutation),
                group,
                metric=args.metric,
            )
            candidate_gradients = autograd_gradients(
                candidate_loss,
                parameters,
                retain_graph=True,
            )
            shuffled_gradients = autograd_gradients(
                shuffled_loss,
                parameters,
                retain_graph=group_index + 1 < len(groups),
            )
            candidate_alignment = gradient_inner_product(
                validation_answer_gradients,
                candidate_gradients,
            )
            shuffled_alignment = gradient_inner_product(
                validation_answer_gradients,
                shuffled_gradients,
            )
            candidate_total = combine_gradients(
                base_gradients,
                candidate_gradients,
                auxiliary_weight=args.kv_weight,
            )
            shuffled_total = combine_gradients(
                base_gradients,
                shuffled_gradients,
                auxiliary_weight=args.kv_weight,
            )
            candidate_mapping, candidate_update = updated_parameter_mapping(
                parameter_names,
                parameters,
                candidate_total,
                update_norm=update_norm,
            )
            shuffled_mapping, shuffled_update = updated_parameter_mapping(
                parameter_names,
                parameters,
                shuffled_total,
                update_norm=update_norm,
            )
            candidate_validation_losses = _stateless_answer_losses(
                scorer,
                candidate_mapping,
                validation_batch,
                precision=args.precision,
                device=device,
            )
            shuffled_validation_losses = _stateless_answer_losses(
                scorer,
                shuffled_mapping,
                validation_batch,
                precision=args.precision,
                device=device,
            )
            group_payloads[group.name] = {
                "definition": group.to_dict(),
                "candidate_train_loss": float(candidate_loss.detach().float()),
                "shuffled_train_loss": float(shuffled_loss.detach().float()),
                "gradient_alignment": {
                    "candidate": candidate_alignment,
                    "shuffled": shuffled_alignment,
                },
                "candidate_update": candidate_update,
                "shuffled_update": shuffled_update,
                "candidate_validation_losses": candidate_validation_losses,
                "shuffled_validation_losses": shuffled_validation_losses,
            }
            del (
                candidate_loss,
                shuffled_loss,
                candidate_gradients,
                shuffled_gradients,
                candidate_total,
                shuffled_total,
                candidate_mapping,
                shuffled_mapping,
            )

        payload = {
            "schema_version": TARGET_UTILITY_RUN_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "batch_index": batch_index,
            "discovery_indices": discovery_batch_indices,
            "validation_indices": validation_batch_indices,
            "derangement": [int(value) for value in permutation.cpu()],
            "discovery_answer_loss": float(
                discovery_output.mean_loss.detach().float()
            ),
            "validation": {
                "original_losses": original_validation_losses,
                "no_target_losses": no_target_losses,
                "no_target_update": base_update,
            },
            "groups": group_payloads,
        }
        _atomic_json(batch_path, payload)
        completed_payloads.append(payload)
        del (
            discovery_batch,
            validation_batch,
            teacher_targets,
            discovery_output,
            validation_output,
            base_gradients,
            validation_answer_gradients,
            base_mapping,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = analyze_target_utility_batches(
        completed_payloads,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    report["request"] = request
    _atomic_json(output_dir / "summary.json", report)
    _atomic_text(
        output_dir / "report.md",
        render_target_utility_markdown(report),
    )
    manifest["state"] = "complete"
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_batches"] = list(range(total_batches))
    manifest["screen_status"] = report["screen_status"]
    _atomic_json(manifest_path, manifest)
    print(f"[complete] screen={report['screen_status']}")
    print(f"[complete] wrote {output_dir / 'summary.json'}")
    print(f"[complete] wrote {output_dir / 'report.md'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Screen official-CODI KV target families by held-out answer-loss utility."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/official_codi_gpt2.yaml"),
    )
    parser.add_argument("--reproduction-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--examples-per-split", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--granularity",
        choices=TARGET_GRANULARITIES,
        default="kind",
    )
    parser.add_argument("--kinds", type=_parse_csv, default=["key", "value"])
    parser.add_argument(
        "--positions",
        type=_parse_positions,
        default=list(range(6)),
    )
    parser.add_argument(
        "--metric",
        choices=("l1", "mse", "smooth_l1"),
        default="l1",
    )
    parser.add_argument("--kv-weight", type=float, default=1.0)
    parser.add_argument("--relative-update-norm", type=float, default=1e-4)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.kv_weight < 0:
        parser.error("--kv-weight must be non-negative")
    if args.relative_update_norm <= 0:
        parser.error("--relative-update-norm must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

