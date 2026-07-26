"""Collect teacher/student KV cross-moments from the released CODI GPT-2 model.

This is an observational, inference-only experiment.  It reconstructs the official
explicit-CoT teacher sequence and the official six-step latent student trajectory from
the same validated public checkpoint.  R-KV is then used only to align six teacher trace
tokens with the six student latent positions.  CODI itself was not trained with KV
distillation.

The output schema intentionally matches ``collect_kv_cross_subspaces.py`` so the existing
Stage 1b CCA/SVD and Stage 1c held-out reduced-rank analyses can be reused unchanged.
"""
from __future__ import annotations

import argparse
import json
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
)
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
    official_codi_answer_is_eligible,
)
from src.data.teacher_cache import cache_to_tensors
from src.eval.official_codi import select_device
from src.losses.kv_compress import rkv_compress
from src.mech.kv_cross_subspace import (
    CROSS_STATISTICS_SCHEMA_VERSION,
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


def _sample_eligible_indices(
    dataset,
    *,
    examples: int,
    seed: int,
) -> tuple[list[int], int]:
    import random

    eligible = [
        index
        for index, answer in enumerate(dataset["answer"])
        if official_codi_answer_is_eligible(answer)
    ]
    if examples > len(eligible):
        raise ValueError(
            f"requested {examples} examples from {len(eligible)} official-eligible rows"
        )
    random.Random(seed).shuffle(eligible)
    return eligible[:examples], len(eligible)


def _verify_reproduction_gate(path: Path, cfg) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"official CODI reproduction summary is missing: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("accuracy_gate", payload.get("gate"))
    if isinstance(gate, str):
        status = gate
    elif isinstance(gate, dict):
        status = gate.get("status")
    else:
        status = None
    if status != "passed":
        raise RuntimeError(
            "official CODI GSM8K reproduction gate has not passed; "
            f"observed status={status!r}"
        )
    revision = payload.get("checkpoint_revision")
    if revision is not None and str(revision) != str(cfg.checkpoint.revision):
        raise RuntimeError(
            "reproduction summary checkpoint revision does not match this collection"
        )
    return {
        "path": str(path),
        "status": status,
        "gsm8k_accuracy": payload.get("datasets", {}).get(
            "gsm8k",
            payload.get("official_last_number_accuracy", {}).get("gsm8k"),
        ),
    }


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
        raise RuntimeError("official cross-moment state uses an incompatible schema")
    if _identity(state.get("metadata", {})) != _identity(metadata):
        raise RuntimeError(
            "existing statistics do not match this official checkpoint or request"
        )
    processed = int(state.get("processed_examples", 0))
    if not 0 <= processed <= int(metadata["requested_examples"]):
        raise RuntimeError("official cross-moment state has an invalid processed count")
    old_indices = state.get("metadata", {}).get("sample_indices")
    new_indices = metadata.get("sample_indices")
    if (
        not isinstance(old_indices, list)
        or not isinstance(new_indices, list)
        or old_indices[:processed] != new_indices[:processed]
    ):
        raise RuntimeError("existing statistics use a different sample prefix")
    print(f"[resume] continuing official CODI KV extraction from example {processed}")
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


def _extract_teacher_trace(
    outputs,
    batch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not outputs.attentions:
        raise RuntimeError(
            "official teacher returned no attentions; use an eager-attention "
            "Transformers build"
        )
    keys, values = cache_to_tensors(outputs.past_key_values)
    attentions = torch.stack(outputs.attentions, dim=1)
    batch_size, layers, heads, _, head_dim = keys.shape
    lengths = batch.teacher_trace_end - batch.teacher_trace_start
    max_trace = max(1, int(lengths.max()))
    trace_keys = keys.new_zeros((batch_size, layers, heads, max_trace, head_dim))
    trace_values = values.new_zeros(trace_keys.shape)
    importance = keys.new_zeros((batch_size, layers, heads, max_trace))
    trace_mask = torch.zeros(
        (batch_size, max_trace),
        dtype=torch.bool,
        device=keys.device,
    )
    for index in range(batch_size):
        start = int(batch.teacher_trace_start[index])
        end = int(batch.teacher_trace_end[index])
        endpoint = int(batch.teacher_endpoint[index])
        answer_start = int(batch.teacher_answer_start[index])
        sequence_end = int(batch.teacher_mask[index].sum())
        count = end - start
        if count <= 0:
            continue
        trace_keys[index, :, :, :count] = keys[index, :, :, start:end]
        trace_values[index, :, :, :count] = values[index, :, :, start:end]
        trace_mask[index, :count] = True
        rows = attentions[index, :, :, answer_start:sequence_end, start:end]
        if rows.shape[-2] == 0:
            rows = attentions[index, :, :, endpoint : endpoint + 1, start:end]
        scores = rows.mean(dim=-2)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        importance[index, :, :, :count] = scores
    return (
        trace_keys.detach(),
        trace_values.detach(),
        importance.detach(),
        trace_mask,
    )


def _extract_student_latent_kv(
    model,
    batch,
    *,
    positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = model.codi(
        input_ids=batch.student_question_ids,
        attention_mask=batch.student_question_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    cache = encoded.past_key_values
    latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
    for _ in range(positions):
        latent_output = model.codi(
            inputs_embeds=latent,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = latent_output.past_key_values
        latent = model.prj(
            latent_output.hidden_states[-1][:, -1, :].unsqueeze(1)
        )
    keys, values = cache_to_tensors(cache)
    return keys[:, :, :, -positions:, :], values[:, :, :, -positions:, :]


def _audit_payload(
    *,
    batch,
    compressed,
    student_keys: torch.Tensor,
    sample_indices: list[int],
) -> dict:
    trace_lengths = (
        batch.teacher_trace_end - batch.teacher_trace_start
    ).detach().cpu()
    representative = compressed.indices[:, 0, 0].detach().cpu()
    return {
        "contract": {
            "teacher_padding": "right",
            "student_question_padding": "left",
            "teacher_sequence": (
                "raw question + official final-step-removed equation CoT + "
                "'The answer is: N' + EOS"
            ),
            "student_sequence": "raw question + BOT + six projected recurrent states",
            "alignment": (
                "same layer/head; R-KV-selected teacher trace tokens sorted "
                "chronologically against student latent positions 0..5"
            ),
        },
        "sample_indices": sample_indices[:8],
        "teacher_trace_tokens": {
            "minimum": int(trace_lengths.min()),
            "maximum": int(trace_lengths.max()),
            "values_first_eight": [int(value) for value in trace_lengths[:8]],
        },
        "representative_rkv_indices_layer0_head0": [
            [int(value) for value in row]
            for row in representative[:8]
        ],
        "teacher_selected_shape": list(compressed.keys.shape),
        "student_latent_shape": list(student_keys.shape),
        "selected_valid_fraction": float(compressed.mask.float().mean()),
        "finite_teacher_keys": bool(torch.isfinite(compressed.keys).all()),
        "finite_teacher_values": bool(torch.isfinite(compressed.values).all()),
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
            "official CODI KV collection requires CUDA; use --allow-cpu only "
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
    # The public training code predates SDPA and the R-KV diagnostic requires explicit
    # attention probabilities. GPT-2 selects its attention function from this config at
    # forward time, so forcing eager preserves the released path and materializes scores.
    official_backbone = model.codi.get_base_model()
    official_backbone.config._attn_implementation = "eager"
    model.to(device=device, dtype=dtype).eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    calibration = cfg.kv_subspace
    data_cfg = load_config(str(calibration.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    indices, eligible_count = _sample_eligible_indices(
        dataset,
        examples=args.examples,
        seed=args.seed,
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
        "analysis": "official_codi_teacher_student_kv_cross_covariance",
        "scientific_scope": (
            "observational diagnostic on an accuracy-gated CODI checkpoint; "
            "R-KV is an analysis-time selector and was not a CODI training target"
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
        "alignment": (
            "same official checkpoint, layer, and KV head; teacher equation-trace "
            "tokens selected by R-KV and sorted chronologically; paired with the "
            "six official student latent positions"
        ),
        "split_assignment": (
            "whole extraction batches alternate between splits; derangements "
            "remain within each split"
        ),
        "null": (
            "repeated seeded within-batch teacher derangements retain teacher and "
            "student marginal distributions while destroying example pairing"
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "collection_state.pt"
    final_path = output_dir / "statistics.pt"
    if args.audit_only:
        collection = None
        processed = 0
    else:
        resume_path = state_path if state_path.is_file() else final_path
        collection, processed = _load_or_create(resume_path, metadata=metadata)
        if resume_path == final_path and processed == len(indices):
            print(f"[complete] verified existing statistics: {final_path}")
            return {**metadata, "processed_examples": processed}
        if processed % args.batch_size:
            raise RuntimeError("resume state is not at a full batch boundary")

    audit_path = output_dir / "alignment_audit.json"
    progress = tqdm(
        total=len(indices),
        initial=processed,
        unit="examples",
        desc="Official CODI paired KV calibration",
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
            compressed = rkv_compress(
                teacher_keys,
                teacher_values,
                teacher_importance,
                teacher_trace_mask,
                positions,
                importance_weight=importance_weight,
            )
            student_keys, student_values = _extract_student_latent_kv(
                model,
                batch,
                positions=positions,
            )

        expected = (len(batch_indices), layers, heads, positions, head_dim)
        if compressed.keys.shape != expected or student_keys.shape != expected:
            raise RuntimeError(
                "official teacher/student KV contract mismatch: "
                f"teacher={tuple(compressed.keys.shape)}, "
                f"student={tuple(student_keys.shape)}, expected={expected}"
            )
        if not audit_path.is_file():
            _atomic_json(
                _audit_payload(
                    batch=batch,
                    compressed=compressed,
                    student_keys=student_keys,
                    sample_indices=batch_indices,
                ),
                audit_path,
            )
            print(f"[audit] wrote {audit_path}")
        if args.audit_only:
            _atomic_json(
                {
                    "schema_version": CROSS_STATISTICS_SCHEMA_VERSION,
                    "state": "audit_only_complete",
                    **metadata,
                    "processed_examples": len(batch_indices),
                    "alignment_audit_file": audit_path.name,
                },
                output_dir / "collection_manifest.json",
            )
            progress.update(len(batch_indices))
            progress.close()
            print("[audit] shape/alignment smoke test complete; no moment file created")
            return {**metadata, "processed_examples": len(batch_indices)}

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
            teacher_keys,
            teacher_values,
            teacher_importance,
            teacher_trace_mask,
            compressed,
            student_keys,
            student_values,
            batch,
        )
        if processed < len(indices) and processed % args.save_every < newly_processed:
            _save_state(
                state_path,
                collection=collection,
                metadata=metadata,
                processed=processed,
                complete=False,
            )
            print(f"[checkpoint] saved official extraction at {processed} examples")
    progress.close()

    _save_state(
        state_path,
        collection=collection,
        metadata=metadata,
        processed=processed,
        complete=True,
    )
    state_path.replace(final_path)
    final_metadata = {**metadata, "processed_examples": processed}
    _atomic_json(
        {
            "schema_version": CROSS_STATISTICS_SCHEMA_VERSION,
            "state": "complete",
            **final_metadata,
            "statistics_file": final_path.name,
            "alignment_audit_file": audit_path.name,
        },
        output_dir / "collection_manifest.json",
    )
    print(f"[complete] wrote {final_path}")
    print(f"[complete] wrote {output_dir / 'collection_manifest.json'}")
    return final_metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect R-KV-aligned teacher/student cross-moments from the "
            "accuracy-gated official CODI GPT-2 checkpoint."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--reproduction-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--examples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--shuffle-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run one batch of shape/alignment checks without allocating moment tensors.",
    )
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
