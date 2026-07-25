"""Collect paired teacher/student KV cross-moments from a trained KaVa checkpoint.

Unlike residual covariance, centered cross-covariance isolates the dependence between a
teacher target and the matching student state. Repeated within-batch derangements provide
a marginal-preserving null. All examples in an extraction batch belong to the same split,
so shuffling never leaks examples between the independent stability halves.
"""
from __future__ import annotations

import argparse
import os
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
    _sample_indices,
    _trained_task,
)
from src.data.datasets import load_train_set
from src.data.prompts import PromptStyle
from src.data.teacher_cache import collate_latent_rows, extract_teacher_targets
from src.eval.run_eval import load_eval_model
from src.losses.kv_compress import rkv_compress
from src.mech.kv_cross_subspace import (
    CROSS_STATISTICS_SCHEMA_VERSION,
    create_cross_moment_collection,
    cross_moment_collection_from_state,
    cross_moment_collection_state,
)
from src.mech.kv_subspace import deterministic_derangement
from src.models.latent_lm import LatentCausalLM
from src.utils.config import load_config


def _identity(metadata: dict) -> dict:
    keys = (
        "checkpoint_step",
        "checkpoint_fingerprint",
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
        "alignment",
    )
    return {key: metadata.get(key) for key in keys}


def _load_or_create(
    path: Path,
    *,
    metadata: dict,
) -> tuple[dict, int]:
    if not path.is_file():
        return (
            create_cross_moment_collection(
                num_splits=int(metadata["num_splits"]),
                layers=int(metadata["layers"]),
                heads=int(metadata["heads"]),
                positions=int(metadata["positions"]),
                head_dim=int(metadata["head_dim"]),
            ),
            0,
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state.get("schema_version", -1)) != CROSS_STATISTICS_SCHEMA_VERSION:
        raise RuntimeError("cross-moment state uses an incompatible schema")
    old_metadata = state.get("metadata", {})
    if _identity(old_metadata) != _identity(metadata):
        raise RuntimeError(
            "cross-moment state does not match this checkpoint or calibration request"
        )
    processed = int(state.get("processed_examples", 0))
    if not 0 <= processed <= int(metadata["requested_examples"]):
        raise RuntimeError("cross-moment state has an invalid processed count")
    old_indices = old_metadata.get("sample_indices")
    new_indices = metadata.get("sample_indices")
    if (
        not isinstance(old_indices, list)
        or not isinstance(new_indices, list)
        or old_indices[:processed] != new_indices[:processed]
    ):
        raise RuntimeError("cross-moment state has a different sample prefix")
    print(f"[resume] continuing paired KV extraction from example {processed}")
    return cross_moment_collection_from_state(state["moments"]), processed


def _save_state(
    path: Path,
    *,
    collection: dict,
    metadata: dict,
    processed: int,
    complete: bool,
) -> None:
    current = dict(metadata)
    current["processed_examples"] = processed
    _atomic_torch_save(
        {
            "schema_version": CROSS_STATISTICS_SCHEMA_VERSION,
            "complete": complete,
            "processed_examples": processed,
            "metadata": current,
            "moments": cross_moment_collection_state(collection),
        },
        path,
    )


@torch.no_grad()
def collect(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    checkpoint_root = Path(args.checkpoint_root or cfg.output_dir).expanduser()
    cfg["output_dir"] = str(checkpoint_root)
    trained_task, manifest = _trained_task(checkpoint_root)
    model, tokenizer, device, checkpoint_step = load_eval_model(cfg)
    if not isinstance(model, LatentCausalLM):
        raise TypeError("Stage 1b requires a trained latent KaVa checkpoint")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "Stage 1b normally requires CUDA. Use --allow-cpu only for a tiny smoke test."
        )
    model.eval()

    data_config_path = manifest.get("effective_config", {}).get(
        "data_config", cfg.data_config
    )
    data_cfg = load_config(data_config_path)
    trace_style = trained_task.get("trace_style", "eq_only")
    dataset = load_train_set(data_cfg, trace_style)
    indices = _sample_indices(len(dataset), args.examples, args.seed)
    style = PromptStyle.from_config(data_cfg["prompt"])
    max_length = int(trained_task.get("max_length", 256))
    importance_weight = float(
        trained_task.get("distillation", {}).get("importance_weight", 0.1)
    )

    layers = int(model.config.num_hidden_layers)
    heads = int(
        getattr(model.config, "num_key_value_heads", None)
        or model.config.num_attention_heads
    )
    hidden_size = int(model.config.hidden_size)
    attention_heads = int(model.config.num_attention_heads)
    if hidden_size % attention_heads:
        raise ValueError("hidden size is not divisible by attention-head count")
    head_dim = hidden_size // attention_heads
    positions = int(model.latent_steps)

    metadata = {
        "analysis": "whitened_teacher_student_kv_cross_covariance",
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_fingerprint": manifest.get("fingerprint"),
        "backbone": trained_task.get("backbone"),
        "resolved_backbone_revision": trained_task.get(
            "resolved_backbone_revision"
        ),
        "train_dataset_fingerprint": getattr(
            dataset,
            "_fingerprint",
            trained_task.get("train_dataset_fingerprint", "unavailable"),
        ),
        "requested_examples": len(indices),
        "processed_examples": 0,
        "batch_size": args.batch_size,
        "num_splits": args.num_splits,
        "seed": args.seed,
        "sampling_strategy": "seeded_full_permutation_prefix_v1",
        "sample_indices": indices,
        "indices_sha256": _indices_fingerprint(indices),
        "shuffle_repeats": args.shuffle_repeats,
        "layers": layers,
        "heads": heads,
        "positions": positions,
        "head_dim": head_dim,
        "trace_style": trace_style,
        "max_length": max_length,
        "importance_weight": importance_weight,
        "precision": args.precision,
        "alignment": (
            "same layer and KV head; teacher R-KV-selected trace tokens sorted "
            "chronologically and paired with student latent slots 0..M-1"
        ),
        "split_assignment": (
            "whole extraction batches alternate between splits; all derangements "
            "remain inside one split"
        ),
        "null": (
            "cross-moments pooled over repeated, seeded, within-batch teacher "
            "derangements while retaining the student order"
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "collection_state.pt"
    final_path = output_dir / "statistics.pt"
    resume_path = state_path if state_path.is_file() else final_path
    collection, processed = _load_or_create(resume_path, metadata=metadata)
    if resume_path == final_path and processed == len(indices):
        result = dict(metadata)
        result["processed_examples"] = processed
        print(f"[complete] verified existing statistics: {final_path}")
        return result
    if processed % args.batch_size:
        raise RuntimeError(
            "resume state is not at a full batch boundary; use a new output directory"
        )

    progress = tqdm(
        total=len(indices),
        initial=processed,
        unit="examples",
        desc="Paired KV calibration",
    )
    while processed < len(indices):
        end = min(processed + args.batch_size, len(indices))
        batch_indices = indices[processed:end]
        if len(batch_indices) < 2:
            raise RuntimeError(
                "the final calibration batch has one example; choose an example count "
                "whose remainder under batch-size is not one"
            )
        rows = [dataset[index] for index in batch_indices]
        batch = collate_latent_rows(
            tokenizer,
            rows,
            style,
            bot_token_id=model.bot_token_id,
            eot_token_id=model.eot_token_id,
            trace_style=trace_style,
            max_length=max_length,
            latent_steps=positions,
        ).to(device)
        with _amp_context(device, args.precision):
            teacher_outputs = model.backbone(
                input_ids=batch.teacher_ids,
                attention_mask=batch.teacher_mask,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            targets = extract_teacher_targets(teacher_outputs, batch)
            compressed = rkv_compress(
                targets.trace_keys,
                targets.trace_values,
                targets.importance,
                targets.trace_mask,
                positions,
                importance_weight=importance_weight,
            )
            _, _, _, student_keys, student_values = model.latent_context(
                batch.question_ids,
                batch.question_mask,
            )
        if compressed.keys.shape != student_keys.shape:
            raise RuntimeError(
                "teacher/student KV shapes do not align: "
                f"{tuple(compressed.keys.shape)} versus {tuple(student_keys.shape)}"
            )
        expected = (layers, heads, positions, head_dim)
        if compressed.keys.shape[1:] != expected:
            raise RuntimeError(
                f"observed KV contract {tuple(compressed.keys.shape[1:])}, "
                f"expected {expected}"
            )

        split = (processed // args.batch_size) % args.num_splits
        split_ids = torch.full(
            (len(batch_indices),),
            split,
            dtype=torch.long,
        )
        collection["actual"]["key"].update(
            compressed.keys,
            student_keys,
            compressed.mask,
            split_ids,
        )
        collection["actual"]["value"].update(
            compressed.values,
            student_values,
            compressed.mask,
            split_ids,
        )
        for repeat in range(args.shuffle_repeats):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed * 1_000_003
                + processed * 101
                + repeat * 10_007
                + 71_117
            )
            permutation = deterministic_derangement(
                len(batch_indices),
                generator=generator,
                device=device,
            )
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
            targets,
            compressed,
            student_keys,
            student_values,
            batch,
        )
        if (
            processed < len(indices)
            and processed % args.save_every < newly_processed
        ):
            _save_state(
                state_path,
                collection=collection,
                metadata=metadata,
                processed=processed,
                complete=False,
            )
            print(f"[checkpoint] saved paired extraction at {processed} examples")
    progress.close()

    _save_state(
        state_path,
        collection=collection,
        metadata=metadata,
        processed=processed,
        complete=True,
    )
    state_path.replace(final_path)
    final_metadata = dict(metadata)
    final_metadata["processed_examples"] = processed
    _atomic_json(
        {
            "schema_version": CROSS_STATISTICS_SCHEMA_VERSION,
            "state": "complete",
            **final_metadata,
            "statistics_file": final_path.name,
        },
        output_dir / "collection_manifest.json",
    )
    print(f"[complete] wrote {final_path}")
    print(f"[complete] wrote {output_dir / 'collection_manifest.json'}")
    return final_metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect paired and shuffled teacher/student KV cross-moments."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--shuffle-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--allow-cpu", action="store_true")
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
