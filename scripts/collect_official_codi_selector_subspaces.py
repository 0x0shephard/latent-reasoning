"""Collect matched official-CODI KV moments for teacher trace selectors.

R-KV, uniform, and several seeded-random selectors share the same teacher/student
forward pass, examples, split assignment, and shuffled-pairing permutations.  This
isolates the choice of teacher trace positions from model execution and data order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import (
    _amp_context,
    _atomic_json,
    _atomic_torch_save,
    _indices_fingerprint,
)
from scripts.collect_official_codi_kv_subspaces import (
    _extract_student_latent_kv,
    _extract_teacher_trace,
    _verify_reproduction_gate,
)
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
    official_codi_answer_is_eligible,
)
from src.eval.official_codi import select_device
from src.losses.kv_compress import (
    CompressedKV,
    boundary_rkv_compress,
    random_compress,
    rkv_compress,
    uniform_compress,
)
from src.mech.kv_cross_subspace import (
    create_cross_moment_collection,
    cross_moment_collection_from_state,
    cross_moment_collection_state,
)
from src.mech.kv_subspace import deterministic_derangement
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


SELECTOR_COLLECTION_SCHEMA_VERSION = 1


def _selector_names(
    random_seeds: list[int], *, include_boundary_rkv: bool = False
) -> list[str]:
    if len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random selector seeds must be unique")
    names = ["rkv", "uniform"]
    if include_boundary_rkv:
        names.insert(0, "boundary_rkv")
    return [*names, *[f"random_seed{seed}" for seed in random_seeds]]


def _load_exclusion_manifest(
    path: Path | None,
    *,
    checkpoint_revision: str,
) -> tuple[set[int], dict | None]:
    if path is None:
        return set(), None
    if not path.is_file():
        raise FileNotFoundError(f"exclusion manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "complete":
        raise RuntimeError("exclusion collection manifest is not complete")
    if str(payload.get("checkpoint_revision")) != str(checkpoint_revision):
        raise RuntimeError("exclusion manifest uses a different checkpoint revision")
    indices = payload.get("sample_indices")
    if not isinstance(indices, list) or not indices:
        raise RuntimeError("exclusion manifest contains no sample indices")
    parsed = {int(index) for index in indices}
    if len(parsed) != len(indices):
        raise RuntimeError("exclusion manifest contains duplicate sample indices")
    return parsed, {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "indices_sha256": _indices_fingerprint(sorted(parsed)),
        "count": len(parsed),
        "train_dataset_fingerprint": payload.get("train_dataset_fingerprint"),
    }


def _sample_eligible_indices_excluding(
    dataset,
    *,
    examples: int,
    seed: int,
    excluded: set[int],
) -> tuple[list[int], int]:
    eligible = [
        index
        for index, answer in enumerate(dataset["answer"])
        if official_codi_answer_is_eligible(answer)
    ]
    candidates = [index for index in eligible if index not in excluded]
    if examples > len(candidates):
        raise ValueError(
            f"requested {examples} examples from {len(candidates)} eligible rows "
            "remaining after exclusions"
        )
    random.Random(seed).shuffle(candidates)
    return candidates[:examples], len(eligible)


def _identity(metadata: dict) -> dict:
    keys = (
        "checkpoint_revision",
        "checkpoint_sha256",
        "official_source_revision",
        "train_dataset_fingerprint",
        "batch_size",
        "num_splits",
        "seed",
        "sampling_strategy",
        "shuffle_repeats",
        "layers",
        "heads",
        "positions",
        "head_dim",
        "importance_weight",
        "precision",
        "selectors",
        "random_selector_seeds",
        "random_score_dtype",
        "include_boundary_rkv",
        "excluded_indices_sha256",
        "excluded_indices_count",
        "alignment",
    )
    return {key: metadata.get(key) for key in keys}


def _new_collection(metadata: dict) -> dict:
    return create_cross_moment_collection(
        num_splits=int(metadata["num_splits"]),
        layers=int(metadata["layers"]),
        heads=int(metadata["heads"]),
        positions=int(metadata["positions"]),
        head_dim=int(metadata["head_dim"]),
    )


def _load_or_create(
    path: Path,
    *,
    metadata: dict,
) -> tuple[dict[str, dict], int]:
    selectors = list(metadata["selectors"])
    if not path.is_file():
        return {selector: _new_collection(metadata) for selector in selectors}, 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state.get("schema_version", -1)) != SELECTOR_COLLECTION_SCHEMA_VERSION:
        raise RuntimeError("selector collection uses an incompatible schema")
    if _identity(state.get("metadata", {})) != _identity(metadata):
        raise RuntimeError(
            "existing selector statistics do not match this checkpoint or request"
        )
    processed = int(state.get("processed_examples", 0))
    if not 0 <= processed <= int(metadata["requested_examples"]):
        raise RuntimeError("selector collection has an invalid processed count")
    old_indices = state.get("metadata", {}).get("sample_indices")
    new_indices = metadata.get("sample_indices")
    if (
        not isinstance(old_indices, list)
        or not isinstance(new_indices, list)
        or old_indices[:processed] != new_indices[:processed]
    ):
        raise RuntimeError("existing selector collection uses a different sample prefix")
    stored = state.get("selectors", {})
    if set(stored) != set(selectors):
        raise RuntimeError("existing selector collection has a different selector set")
    print(f"[resume] continuing matched selector extraction at example {processed}")
    return {
        selector: cross_moment_collection_from_state(stored[selector])
        for selector in selectors
    }, processed


def _save_state(
    path: Path,
    *,
    collections: dict[str, dict],
    metadata: dict,
    processed: int,
    complete: bool,
) -> None:
    current = {**metadata, "processed_examples": processed}
    _atomic_torch_save(
        {
            "schema_version": SELECTOR_COLLECTION_SCHEMA_VERSION,
            "complete": complete,
            "processed_examples": processed,
            "metadata": current,
            "selectors": {
                selector: cross_moment_collection_state(collection)
                for selector, collection in collections.items()
            },
        },
        path,
    )


def _random_generator(
    *,
    device: torch.device,
    selector_seed: int,
    processed: int,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(
        int(selector_seed) * 1_000_003 + int(processed) * 101 + 83_771
    )
    return generator


def _compress_all(
    *,
    teacher_keys: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_importance: torch.Tensor,
    teacher_trace_mask: torch.Tensor,
    positions: int,
    importance_weight: float,
    random_seeds: list[int],
    processed: int,
    include_boundary_rkv: bool,
) -> dict[str, CompressedKV]:
    selected = {
        "rkv": rkv_compress(
            teacher_keys,
            teacher_values,
            teacher_importance,
            teacher_trace_mask,
            positions,
            importance_weight=importance_weight,
        ),
        "uniform": uniform_compress(
            teacher_keys,
            teacher_values,
            teacher_trace_mask,
            positions,
        ),
    }
    if include_boundary_rkv:
        selected["boundary_rkv"] = boundary_rkv_compress(
            teacher_keys,
            teacher_values,
            teacher_importance,
            teacher_trace_mask,
            positions,
            importance_weight=importance_weight,
        )
    for seed in random_seeds:
        selected[f"random_seed{seed}"] = random_compress(
            teacher_keys,
            teacher_values,
            teacher_trace_mask,
            positions,
            generator=_random_generator(
                device=teacher_keys.device,
                selector_seed=seed,
                processed=processed,
            ),
            # Avoid low-precision random-score ties under bfloat16 inference.
            score_dtype=torch.float32,
        )
    return selected


def _selection_audit(
    *,
    batch,
    selected: dict[str, CompressedKV],
    student_keys: torch.Tensor,
    sample_indices: list[int],
) -> dict:
    trace_lengths = (
        batch.teacher_trace_end - batch.teacher_trace_start
    ).detach().cpu()
    selector_payload = {}
    for name, compressed in selected.items():
        mask = compressed.mask.detach().cpu()
        indices = compressed.indices.detach().cpu()
        selector_payload[name] = {
            "selected_valid_fraction": float(mask.float().mean()),
            "fraction_examples_with_all_targets": float(
                (mask[:, 0, 0].sum(dim=-1) == mask.shape[-1]).float().mean()
            ),
            "per_position_valid_fraction": [
                float(value)
                for value in mask[:, 0, 0].float().mean(dim=0)
            ],
            "representative_indices_layer0_head0": [
                [int(value) for value in row]
                for row in indices[:8, 0, 0]
            ],
            "finite_teacher_keys": bool(torch.isfinite(compressed.keys).all()),
            "finite_teacher_values": bool(torch.isfinite(compressed.values).all()),
        }

    agreement = {}
    names = list(selected)
    for left_index, left_name in enumerate(names):
        left = selected[left_name]
        for right_name in names[left_index + 1 :]:
            right = selected[right_name]
            valid = left.mask & right.mask
            denominator = int(valid.sum())
            agreement[f"{left_name}_vs_{right_name}"] = (
                float(((left.indices == right.indices) & valid).sum() / denominator)
                if denominator
                else None
            )
    return {
        "contract": {
            "teacher_padding": "right",
            "student_question_padding": "left",
            "alignment": (
                "same official checkpoint, examples, forward pass, layer, and head; "
                "selector-specific teacher trace tokens are sorted chronologically "
                "and paired with official student latent positions 0..5"
            ),
            "null": (
                "the same seeded within-batch teacher derangement is applied to every "
                "selector for each repeat"
            ),
        },
        "sample_indices": sample_indices[:8],
        "teacher_trace_tokens": {
            "minimum": int(trace_lengths.min()),
            "maximum": int(trace_lengths.max()),
            "values_first_eight": [int(value) for value in trace_lengths[:8]],
        },
        "student_latent_shape": list(student_keys.shape),
        "selectors": selector_payload,
        "selected_index_agreement": agreement,
        "finite_student_keys": bool(torch.isfinite(student_keys).all()),
    }


@torch.no_grad()
def collect(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    gate = _verify_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "matched selector collection requires CUDA; use --allow-cpu only "
            "for a tiny smoke test"
        )
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
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

    calibration = cfg.kv_subspace
    data_cfg = load_config(str(calibration.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    excluded_indices, exclusion = _load_exclusion_manifest(
        args.exclude_manifest,
        checkpoint_revision=str(cfg.checkpoint.revision),
    )
    if (
        exclusion is not None
        and exclusion["train_dataset_fingerprint"] not in (None, "unavailable")
        and exclusion["train_dataset_fingerprint"]
        != getattr(dataset, "_fingerprint", "unavailable")
    ):
        raise RuntimeError("exclusion manifest uses a different training dataset")
    indices, eligible_count = _sample_eligible_indices_excluding(
        dataset,
        examples=args.examples,
        seed=args.seed,
        excluded=excluded_indices,
    )
    if set(indices) & excluded_indices:
        raise RuntimeError("fresh calibration sample overlaps excluded indices")
    random_seeds = list(args.random_selector_seeds)
    selectors = _selector_names(
        random_seeds,
        include_boundary_rkv=args.include_boundary_rkv,
    )

    layers = int(model.config.num_hidden_layers)
    heads = int(
        getattr(model.config, "num_key_value_heads", None)
        or model.config.num_attention_heads
    )
    attention_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    if hidden_size % attention_heads:
        raise RuntimeError("official CODI hidden size is not divisible by heads")
    head_dim = hidden_size // attention_heads
    positions = int(calibration.latent_positions)
    importance_weight = float(calibration.importance_weight)

    metadata = {
        "analysis": "official_codi_teacher_trace_selector_specificity",
        "scientific_scope": (
            "observational selector comparison on an accuracy-gated CODI checkpoint; "
            "all selectors are analysis-time interventions"
        ),
        "reproduction_gate": gate,
        "official_source_repo": str(cfg.official_source.repo),
        "official_source_revision": str(cfg.official_source.revision),
        "checkpoint_repo": str(cfg.checkpoint.repo_id),
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "checkpoint_matched_numel_fraction": load_report.matched_numel_fraction,
        "base_model": str(cfg.model.base_model),
        "base_revision": str(cfg.model.base_revision),
        "attention_implementation": "eager",
        "train_dataset_fingerprint": getattr(dataset, "_fingerprint", "unavailable"),
        "eligible_train_examples": eligible_count,
        "requested_examples": len(indices),
        "processed_examples": 0,
        "batch_size": args.batch_size,
        "num_splits": args.num_splits,
        "seed": args.seed,
        "sampling_strategy": "seeded_eligible_permutation_prefix_v1",
        "sample_indices": indices,
        "indices_sha256": _indices_fingerprint(indices),
        "shuffle_repeats": args.shuffle_repeats,
        "layers": layers,
        "heads": heads,
        "positions": positions,
        "head_dim": head_dim,
        "importance_weight": importance_weight,
        "precision": args.precision,
        "selectors": selectors,
        "random_selector_seeds": random_seeds,
        "random_score_dtype": "float32",
        "include_boundary_rkv": bool(args.include_boundary_rkv),
        "boundary_selector": (
            "force first and last valid trace tokens; fill four interior slots "
            "with R-KV"
            if args.include_boundary_rkv
            else None
        ),
        "exclusion_manifest": exclusion,
        "excluded_indices_sha256": (
            None if exclusion is None else exclusion["indices_sha256"]
        ),
        "excluded_indices_count": (
            0 if exclusion is None else exclusion["count"]
        ),
        "sample_overlap_with_exclusion": 0,
        "alignment": (
            "same checkpoint, batch, teacher/student forward pass, layer, and head; "
            "selector-specific teacher trace positions sorted chronologically and "
            "paired with the six official student latent positions"
        ),
        "split_assignment": (
            "whole extraction batches alternate between splits; all selectors share "
            "the assignment and derangements remain within each split"
        ),
        "null": (
            "matched repeated seeded within-batch teacher derangements retain each "
            "selector's marginals while destroying example pairing"
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "collection_state.pt"
    final_path = output_dir / "selector_statistics.pt"
    if args.audit_only:
        collections = None
        processed = 0
    else:
        resume_path = state_path if state_path.is_file() else final_path
        collections, processed = _load_or_create(resume_path, metadata=metadata)
        if resume_path == final_path and processed == len(indices):
            print(f"[complete] verified existing statistics: {final_path}")
            return {**metadata, "processed_examples": processed}
        if processed % args.batch_size:
            raise RuntimeError("resume state is not at a full batch boundary")

    audit_path = output_dir / "selection_audit.json"
    progress = tqdm(
        total=len(indices),
        initial=processed,
        unit="examples",
        desc="Official CODI matched selector calibration",
    )
    while processed < len(indices):
        end = min(processed + args.batch_size, len(indices))
        batch_indices = indices[processed:end]
        if len(batch_indices) < 2:
            raise RuntimeError(
                "final calibration batch has one example; change examples or batch size"
            )
        rows = [dataset[index] for index in batch_indices]
        batch = collate_official_codi_kv_rows(
            tokenizer,
            rows,
            bot_token_id=model.bot_id,
        ).to(device)
        with _amp_context(device, args.precision):
            teacher_outputs = model.codi(
                input_ids=batch.teacher_ids,
                attention_mask=batch.teacher_mask,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=True,
                return_dict=True,
            )
            (
                teacher_keys,
                teacher_values,
                teacher_importance,
                teacher_trace_mask,
            ) = _extract_teacher_trace(teacher_outputs, batch)
            selected = _compress_all(
                teacher_keys=teacher_keys,
                teacher_values=teacher_values,
                teacher_importance=teacher_importance,
                teacher_trace_mask=teacher_trace_mask,
                positions=positions,
                importance_weight=importance_weight,
                random_seeds=random_seeds,
                processed=processed,
                include_boundary_rkv=args.include_boundary_rkv,
            )
            student_keys, student_values = _extract_student_latent_kv(
                model,
                batch,
                positions=positions,
            )

        expected = (len(batch_indices), layers, heads, positions, head_dim)
        if student_keys.shape != expected:
            raise RuntimeError(
                f"student KV shape {tuple(student_keys.shape)} != {expected}"
            )
        for selector, compressed in selected.items():
            if compressed.keys.shape != expected:
                raise RuntimeError(
                    f"{selector} teacher KV shape {tuple(compressed.keys.shape)} "
                    f"!= {expected}"
                )
        audit = _selection_audit(
            batch=batch,
            selected=selected,
            student_keys=student_keys,
            sample_indices=batch_indices,
        )
        if args.audit_only or not audit_path.is_file():
            _atomic_json(audit, audit_path)
            print(f"[audit] wrote {audit_path}")
        if args.audit_only:
            _atomic_json(
                {
                    "schema_version": SELECTOR_COLLECTION_SCHEMA_VERSION,
                    "state": "audit_only_complete",
                    **metadata,
                    "processed_examples": len(batch_indices),
                    "selection_audit_file": audit_path.name,
                },
                output_dir / "collection_manifest.json",
            )
            progress.update(len(batch_indices))
            progress.close()
            print("[audit] matched selector smoke test complete")
            return {**metadata, "processed_examples": len(batch_indices)}

        split = (processed // args.batch_size) % args.num_splits
        split_ids = torch.full((len(batch_indices),), split, dtype=torch.long)
        permutations = []
        for repeat in range(args.shuffle_repeats):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed * 1_000_003
                + processed * 101
                + repeat * 10_007
                + 71_117
            )
            permutations.append(
                deterministic_derangement(
                    len(batch_indices),
                    generator=generator,
                    device=device,
                )
            )

        for selector, compressed in selected.items():
            collection = collections[selector]
            collection["actual"]["key"].update(
                compressed.keys, student_keys, compressed.mask, split_ids
            )
            collection["actual"]["value"].update(
                compressed.values, student_values, compressed.mask, split_ids
            )
            for permutation in permutations:
                shuffled_mask = compressed.mask.index_select(0, permutation)
                collection["shuffled"]["key"].update(
                    compressed.keys.index_select(0, permutation),
                    student_keys,
                    shuffled_mask,
                    split_ids,
                )
                collection["shuffled"]["value"].update(
                    compressed.values.index_select(0, permutation),
                    student_values,
                    shuffled_mask,
                    split_ids,
                )

        newly_processed = end - processed
        processed = end
        progress.update(newly_processed)
        del (
            teacher_outputs,
            teacher_keys,
            teacher_values,
            teacher_importance,
            teacher_trace_mask,
            selected,
            student_keys,
            student_values,
            permutations,
            batch,
        )
        if processed < len(indices) and processed % args.save_every < newly_processed:
            _save_state(
                state_path,
                collections=collections,
                metadata=metadata,
                processed=processed,
                complete=False,
            )
            print(f"[checkpoint] saved matched selector extraction at {processed}")
    progress.close()

    _save_state(
        state_path,
        collections=collections,
        metadata=metadata,
        processed=processed,
        complete=True,
    )
    state_path.replace(final_path)
    final_metadata = {**metadata, "processed_examples": processed}
    _atomic_json(
        {
            "schema_version": SELECTOR_COLLECTION_SCHEMA_VERSION,
            "state": "complete",
            **final_metadata,
            "statistics_file": final_path.name,
            "selection_audit_file": audit_path.name,
        },
        output_dir / "collection_manifest.json",
    )
    print(f"[complete] wrote {final_path}")
    return final_metadata


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "random selector seeds must be comma-separated integers"
        ) from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one random selector seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("random selector seeds must be unique")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect matched R-KV, uniform, and random teacher-trace selector "
            "statistics from the accuracy-gated official CODI checkpoint."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--reproduction-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--examples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--shuffle-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-selector-seeds", type=_parse_seeds, default=[101, 211, 307, 401])
    parser.add_argument(
        "--include-boundary-rkv",
        action="store_true",
        help=(
            "Add a hybrid selector that forces the first and last valid trace "
            "tokens and uses R-KV for the interior slots."
        ),
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help=(
            "Complete prior collection_manifest.json whose sample indices must "
            "be excluded from this calibration sample."
        ),
    )
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="cuda",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.examples < 2:
        parser.error("examples must be at least two")
    if args.batch_size < 2:
        parser.error("batch-size must be at least two")
    if args.examples % args.batch_size == 1:
        parser.error("examples modulo batch-size cannot equal one")
    if args.num_splits < 2:
        parser.error("num-splits must be at least two")
    if args.shuffle_repeats < 1:
        parser.error("shuffle-repeats must be positive")
    if args.save_every <= 0:
        parser.error("save-every must be positive")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
