"""Collect streaming teacher/student KV residual statistics from a trained KaVa run.

This is a calibration-only workflow.  It never updates model weights.  By default it
stores exact covariance sufficient statistics rather than raw activations, supports
atomic restart checkpoints, and uses the same R-KV alignment as KaVa training.

Example:
    python -u scripts/collect_kv_subspaces.py \
      --config configs/kava.yaml \
      --checkpoint-root /content/drive/MyDrive/CODI_KAVA/outputs/kava \
      --output-dir /content/drive/MyDrive/CODI_KAVA/outputs/stage1_kv_subspaces \
      --examples 2000 --batch-size 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datasets import load_train_set
from src.data.prompts import PromptStyle
from src.data.teacher_cache import collate_latent_rows, extract_teacher_targets
from src.eval.run_eval import load_eval_model
from src.losses.kv_compress import rkv_compress
from src.mech.kv_subspace import (
    STATISTICS_SCHEMA_VERSION,
    create_moment_collection,
    deterministic_derangement,
    energy_matched_random,
    moment_collection_from_state,
    moment_collection_state,
)
from src.models.latent_lm import LatentCausalLM
from src.utils.config import load_config


def _atomic_torch_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def _atomic_json(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _sample_indices(size: int, examples: int, seed: int) -> list[int]:
    if examples <= 0:
        raise ValueError("examples must be positive")
    if examples > size:
        raise ValueError(f"requested {examples} examples from a dataset of {size}")
    # Shuffle once, then take a prefix.  Unlike ``random.sample(..., k)``, this makes a
    # 2,000-example calibration an exact prefix of a later 5,000-example confirmation.
    order = list(range(size))
    random.Random(seed).shuffle(order)
    return order[:examples]


def _indices_fingerprint(indices: list[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    dtype = torch.float16
    if precision == "bfloat16":
        dtype = torch.bfloat16
    elif precision == "auto":
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _trained_task(checkpoint_root: Path) -> tuple[dict, dict]:
    manifest_path = checkpoint_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = manifest.get("effective_config", {}).get("task", {})
    if task.get("method") != "kava":
        raise ValueError(
            "Stage 1 must start from a KaVa checkpoint so teacher KV supervision "
            "matches the trained objective"
        )
    if float(task.get("distillation", {}).get("kv_weight", 0.0)) <= 0:
        raise ValueError("the checkpoint manifest does not enable KV distillation")
    return task, manifest


def _identity(metadata: dict) -> dict:
    keys = (
        "checkpoint_step",
        "checkpoint_fingerprint",
        "train_dataset_fingerprint",
        "batch_size",
        "num_splits",
        "seed",
        "sampling_strategy",
        "layers",
        "heads",
        "positions",
        "head_dim",
        "importance_weight",
        "precision",
        "raw_residual_shards",
        "alignment",
    )
    return {key: metadata.get(key) for key in keys}


def _load_or_create_collection(
    state_path: Path,
    *,
    metadata: dict,
) -> tuple[dict, int]:
    if not state_path.is_file():
        return (
            create_moment_collection(
                num_splits=int(metadata["num_splits"]),
                layers=int(metadata["layers"]),
                heads=int(metadata["heads"]),
                positions=int(metadata["positions"]),
                head_dim=int(metadata["head_dim"]),
            ),
            0,
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state.get("schema_version", -1)) != STATISTICS_SCHEMA_VERSION:
        raise RuntimeError("collection state uses an incompatible schema version")
    if _identity(state.get("metadata", {})) != _identity(metadata):
        raise RuntimeError(
            "collection state does not match this checkpoint/calibration request; "
            "use a new output directory"
        )
    old_metadata = state.get("metadata", {})
    processed = int(state.get("processed_examples", 0))
    if not 0 <= processed <= int(metadata["requested_examples"]):
        raise RuntimeError("collection state has an invalid processed-example count")
    old_indices = old_metadata.get("sample_indices")
    new_indices = metadata.get("sample_indices")
    if (
        not isinstance(old_indices, list)
        or not isinstance(new_indices, list)
        or old_indices[:processed] != new_indices[:processed]
    ):
        raise RuntimeError(
            "collection state does not contain the same deterministic sample prefix"
        )
    print(f"[resume] continuing KV extraction from example {processed}")
    return moment_collection_from_state(state["moments"]), processed


def _save_state(
    path: Path,
    *,
    collection: dict,
    metadata: dict,
    processed: int,
    complete: bool,
) -> None:
    current_metadata = dict(metadata)
    current_metadata["processed_examples"] = processed
    _atomic_torch_save(
        {
            "schema_version": STATISTICS_SCHEMA_VERSION,
            "complete": complete,
            "processed_examples": processed,
            "metadata": current_metadata,
            "moments": moment_collection_state(collection),
        },
        path,
    )


def _write_residual_shard(
    shard_dir: Path,
    *,
    start: int,
    end: int,
    example_indices: list[int],
    split_ids: torch.Tensor,
    key_residual: torch.Tensor,
    value_residual: torch.Tensor,
    mask: torch.Tensor,
    selected_indices: torch.Tensor,
) -> None:
    target = shard_dir / f"residuals_{start:06d}_{end:06d}.pt"
    if target.is_file():
        return
    _atomic_torch_save(
        {
            "start": start,
            "end": end,
            "example_indices": example_indices,
            "split_ids": split_ids.cpu(),
            "key_residual": key_residual.detach().to(
                device="cpu", dtype=torch.float16
            ),
            "value_residual": value_residual.detach().to(
                device="cpu", dtype=torch.float16
            ),
            "mask": mask.detach().cpu(),
            "teacher_selected_trace_indices": selected_indices.detach().cpu(),
        },
        target,
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
        raise TypeError("Stage 1 requires a latent KaVa checkpoint")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "Stage 1 extraction requires a GPU in normal use. Pass --allow-cpu only "
            "for a very small implementation smoke test."
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
    importance_weight = float(
        trained_task.get("distillation", {}).get("importance_weight", 0.1)
    )
    max_length = int(trained_task.get("max_length", 256))

    # Infer the cache contract from the model configuration. GPT-2 has one KV head per
    # attention head; models with grouped-query attention expose num_key_value_heads.
    layers = int(model.config.num_hidden_layers)
    heads = int(
        getattr(model.config, "num_key_value_heads", None)
        or model.config.num_attention_heads
    )
    hidden_size = int(model.config.hidden_size)
    if hidden_size % int(model.config.num_attention_heads):
        raise ValueError("hidden size is not divisible by attention-head count")
    head_dim = hidden_size // int(model.config.num_attention_heads)
    positions = int(model.latent_steps)

    metadata = {
        "analysis": "teacher_minus_student_kv_residual_subspaces",
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
        "nulls": {
            "shuffled": (
                "teacher targets deranged across examples within each extraction batch"
            ),
            "random": (
                "isotropic Gaussian residuals with exactly matched energy per "
                "layer, head, and aligned position"
            ),
        },
        "raw_residual_shards": bool(args.save_residual_shards),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "collection_state.pt"
    final_path = output_dir / "statistics.pt"
    resume_path = state_path if state_path.is_file() else final_path
    collection, processed = _load_or_create_collection(
        resume_path,
        metadata=metadata,
    )
    if resume_path == final_path and processed == len(indices):
        complete_metadata = dict(metadata)
        complete_metadata["processed_examples"] = processed
        print(f"[complete] verified existing statistics: {final_path}")
        return complete_metadata
    shard_dir = output_dir / "residual_shards"

    progress = tqdm(
        total=len(indices),
        initial=processed,
        unit="examples",
        desc="KV calibration",
    )
    while processed < len(indices):
        end = min(processed + args.batch_size, len(indices))
        batch_indices = indices[processed:end]
        if len(batch_indices) < 2:
            # A singleton cannot instantiate the shuffled null. Fold it into the prior
            # batch when possible; otherwise require at least two calibration examples.
            start = max(0, processed - 1)
            batch_indices = indices[start:end]
            replay_first = True
        else:
            start = processed
            replay_first = False
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
            teacher_targets = extract_teacher_targets(teacher_outputs, batch)
            compressed = rkv_compress(
                teacher_targets.trace_keys,
                teacher_targets.trace_values,
                teacher_targets.importance,
                teacher_targets.trace_mask,
                positions,
                importance_weight=importance_weight,
            )
            _, _, _, student_keys, student_values = model.latent_context(
                batch.question_ids,
                batch.question_mask,
            )

        if compressed.keys.shape != student_keys.shape:
            raise RuntimeError(
                "teacher/student KV alignment failed: "
                f"{tuple(compressed.keys.shape)} versus {tuple(student_keys.shape)}"
            )
        if compressed.keys.shape[1:] != (
            layers,
            heads,
            positions,
            head_dim,
        ):
            raise RuntimeError(
                "observed KV cache shape does not match the model-derived contract"
            )

        actual_mask = compressed.mask
        key_residual = compressed.keys.float() - student_keys.float()
        value_residual = compressed.values.float() - student_values.float()
        shuffle_generator = torch.Generator(device="cpu").manual_seed(
            args.seed * 1_000_003 + processed + 17_171
        )
        permutation = deterministic_derangement(
            len(batch_indices),
            generator=shuffle_generator,
            device=device,
        )
        shuffled_mask = compressed.mask.index_select(0, permutation)
        shuffled_key = (
            compressed.keys.index_select(0, permutation).float()
            - student_keys.float()
        )
        shuffled_value = (
            compressed.values.index_select(0, permutation).float()
            - student_values.float()
        )
        if replay_first:
            # The first row was processed in the previous batch and is present only to
            # create a valid derangement for the final singleton.
            keep = torch.tensor([False, True])
            key_residual = key_residual[keep.to(device)]
            value_residual = value_residual[keep.to(device)]
            actual_mask = actual_mask[keep.to(device)]
            shuffled_key = shuffled_key[keep.to(device)]
            shuffled_value = shuffled_value[keep.to(device)]
            shuffled_mask = shuffled_mask[keep.to(device)]
            compressed_indices = compressed.indices[keep.to(device)]
            persisted_indices = batch_indices[1:]
        else:
            compressed_indices = compressed.indices
            persisted_indices = batch_indices
        split_ids = torch.arange(
            processed, end, dtype=torch.long
        ).remainder(args.num_splits)
        random_generator = torch.Generator(device=device).manual_seed(
            args.seed * 1_000_003 + processed + 91_919
        )
        random_key = energy_matched_random(
            key_residual,
            actual_mask,
            generator=random_generator,
        )
        random_value = energy_matched_random(
            value_residual,
            actual_mask,
            generator=random_generator,
        )

        collection["actual"]["key"].update(
            key_residual, actual_mask, split_ids
        )
        collection["actual"]["value"].update(
            value_residual, actual_mask, split_ids
        )
        collection["shuffled"]["key"].update(
            shuffled_key, shuffled_mask, split_ids
        )
        collection["shuffled"]["value"].update(
            shuffled_value, shuffled_mask, split_ids
        )
        collection["random"]["key"].update(
            random_key, actual_mask, split_ids
        )
        collection["random"]["value"].update(
            random_value, actual_mask, split_ids
        )

        if args.save_residual_shards:
            _write_residual_shard(
                shard_dir,
                start=start if not replay_first else processed,
                end=end,
                example_indices=persisted_indices,
                split_ids=split_ids,
                key_residual=key_residual,
                value_residual=value_residual,
                mask=actual_mask,
                selected_indices=compressed_indices,
            )

        newly_processed = end - processed
        processed = end
        progress.update(newly_processed)
        del (
            teacher_outputs,
            teacher_targets,
            compressed,
            student_keys,
            student_values,
            key_residual,
            value_residual,
            shuffled_key,
            shuffled_value,
            random_key,
            random_value,
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
            print(f"[checkpoint] saved extraction state at {processed} examples")
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
            "schema_version": STATISTICS_SCHEMA_VERSION,
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
        description="Collect split-wise teacher/student KV residual statistics."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Override config output_dir with a completed KaVa output directory.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--save-residual-shards",
        action="store_true",
        help="Also persist float16 actual residuals and selected teacher indices.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit tiny smoke tests without CUDA; not suitable for 2,000 examples.",
    )
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("batch-size must be at least two for the shuffled baseline")
    if args.examples < 2:
        parser.error("examples must be at least two")
    if args.num_splits < 2:
        parser.error("num-splits must be at least two")
    if args.save_every <= 0:
        parser.error("save-every must be positive")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
