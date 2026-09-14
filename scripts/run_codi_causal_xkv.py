"""Compare ordinary xKV with a causally protected feature core in CODI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from src.data.datasets import load_eval_set
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.causal_xkv import (
    compress_reconstruct_cache, mapped_group_causal_basis,
    mapped_group_independent_kv_basis, mapped_group_variable_causal_basis,
)
from src.mech.layerwise_u28 import RidgeTransport, random_orthonormal_basis
from src.mech.official_codi_layerwise import gpt2_qkv_response_bases
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_causal_protected_cross_layer_kv_v1"


def groups_around_validated_run(selected: tuple[int, ...], maximum_other_group: int = 4):
    """Partition 12 layers while keeping the validated contiguous run intact."""
    if not selected or tuple(range(selected[0], selected[-1] + 1)) != selected:
        raise ValueError("selected layers must be one non-empty contiguous run")
    if selected[0] < 0 or selected[-1] >= 12:
        raise ValueError("selected layer is outside GPT-2 small")
    groups = []
    cursor = 0
    while cursor < 12:
        if cursor == selected[0]:
            groups.append(selected)
            cursor = selected[-1] + 1
            continue
        stop = min(cursor + maximum_other_group, selected[0] if cursor < selected[0] else 12)
        groups.append(tuple(range(cursor, stop)))
        cursor = stop
    return tuple(groups)


class FinalLatentCacheFactorizer:
    """Quality proxy: factorize after latent pass 5, reconstruct for stock GPT-2."""

    def __init__(self, groups, rank, protected_bases):
        self.groups = groups
        self.rank = rank
        self.protected_bases = protected_bases
        self.reports = []

    def __call__(self, cache, latent_position):
        if latent_position != 5:
            return cache
        transformed, report = compress_reconstruct_cache(
            cache, groups=self.groups, rank=self.rank, protected_bases=self.protected_bases,
        )
        self.reports.append(report)
        return transformed

    def summary(self) -> dict:
        dense = sum(value["dense_elements"] for value in self.reports)
        factor = sum(value["factor_elements"] for value in self.reports)
        records = [row for value in self.reports for row in value["records"]]
        compressed = [row for row in records if row["status"] == "compressed"]
        return {
            "dense_elements": dense, "factor_elements": factor,
            "modelled_compression_ratio": dense / max(1, factor),
            "batches": len(self.reports),
            "compressed_group_rows": len(compressed),
            "dense_short_context_group_rows": len(records) - len(compressed),
            "mean_relative_reconstruction_error": (
                sum(row["relative_error"] for row in compressed) / len(compressed)
                if compressed else None
            ),
            "runtime_representation": "dense_reconstruction_quality_proxy",
        }


def _accuracy(outputs, rows) -> float:
    return sum(official_answers_match(text, row["gold"]) for text, row in zip(outputs, rows)) / len(rows)


def run(args) -> dict:
    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    artifact = torch.load(args.layerwise_artifact, map_location="cpu", weights_only=False)
    supported_contracts = {
        "official_codi_layerwise_transport_of_final_u28_v1",
        "official_codi_independent_layer_subspaces_direct_latent_kv_v1",
        "official_codi_preanswer_gradient_variable_rank_kv_v2",
        "official_codi_task_sensitive_variable_rank_kv_v1",
        "official_codi_direct_cache_task_sensitive_independent_kv_v1",
        "official_codi_direct_cache_task_sensitive_confirmed_v1",
    }
    if artifact.get("contract") not in supported_contracts:
        raise RuntimeError("wrong layerwise artifact contract")
    if not artifact["gate"]["passed"] and not args.allow_unconfirmed:
        raise RuntimeError(
            "layerwise causal gate did not pass; xKV experiment is blocked. "
            "Use --allow-unconfirmed only for a labelled exploratory run."
        )
    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
        dtype=dtype, settings=cfg.model, token=token,
    )
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    if load_report.checkpoint_sha256 != artifact["checkpoint_sha256"]:
        raise RuntimeError("layerwise artifact belongs to another checkpoint")
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    style = PromptStyle.from_config(data_cfg.prompt)
    rows = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    if args.examples:
        rows = rows[: args.examples]
    questions = [row["question"] for row in rows]

    independent_kv_discovery = artifact["contract"] in {
        "official_codi_direct_cache_task_sensitive_independent_kv_v1",
        "official_codi_direct_cache_task_sensitive_confirmed_v1",
    }
    variable_discovery = artifact["contract"] in {
        "official_codi_preanswer_gradient_variable_rank_kv_v2",
        "official_codi_task_sensitive_variable_rank_kv_v1",
        "official_codi_direct_cache_task_sensitive_independent_kv_v1",
        "official_codi_direct_cache_task_sensitive_confirmed_v1",
    }
    direct_discovery = variable_discovery or artifact["contract"].endswith("direct_latent_kv_v1")
    selected_key = "selected_contiguous_layers" if direct_discovery else "selected_contiguous_attention_layers"
    selected_group = tuple(artifact["gate"][selected_key])
    if not selected_group:
        # Only reachable for explicitly labelled --allow-unconfirmed runs.
        selected_group = (8, 9, 10, 11)
    groups = groups_around_validated_run(selected_group)
    if variable_discovery:
        responses = {
            layer: {"key": artifact["selected_key_responses"][layer],
                    "value": artifact["selected_value_responses"][layer]}
            for layer in range(12)
        }
    elif direct_discovery:
        responses = {
            layer: {"key": artifact["aligned_key_responses"][layer],
                    "value": artifact["aligned_value_responses"][layer]}
            for layer in range(12)
        }
    else:
        attention_bases = {
            layer: RidgeTransport.from_state_dict(artifact["transports"][f"attn_ln_{layer:02d}"]).basis
            for layer in range(12)
        }
        responses = gpt2_qkv_response_bases(model, attention_bases)
    causal_bases = {}
    random_bases = {}
    generator = torch.Generator().manual_seed(args.random_seed)
    for group in (selected_group,):
        if independent_kv_discovery:
            mapper = mapped_group_independent_kv_basis
        else:
            mapper = (
                mapped_group_variable_causal_basis
                if variable_discovery else mapped_group_causal_basis
            )
        causal_bases[group] = mapper(
            [responses[layer]["key"] for layer in group],
            [responses[layer]["value"] for layer in group],
        )
        random_bases[group] = random_orthonormal_basis(
            causal_bases[group].shape[0], causal_bases[group].shape[1], generator=generator
        )

    started = time.perf_counter()
    dense_outputs = generate_official_codi(
        model, tokenizer, questions, latent_iterations=int(cfg.eval.latent_iterations),
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=device,
        answer_cue=style.answer_prefix, force_answer_cue=True,
    )
    dense_seconds = time.perf_counter() - started
    dense_accuracy = _accuracy(dense_outputs, rows)
    results = {
        "dense": {"accuracy": dense_accuracy, "seconds": dense_seconds,
                  "outputs": dense_outputs}
    }
    ranks = sorted({int(value) for value in args.ranks.split(",")})
    for rank in ranks:
        protected_name = (
            f"xkv_independent_kv_protected_r{rank}"
            if independent_kv_discovery else
            f"xkv_variable_protected_r{rank}"
            if variable_discovery else f"xkv_u28_protected_r{rank}"
        )
        arms = {
            f"per_layer_svd_r{rank}": (tuple((layer,) for layer in range(12)), None),
            f"xkv_svd_r{rank}": (groups, None),
            f"xkv_random_protected_r{rank}": (groups, random_bases),
            protected_name: (groups, causal_bases),
        }
        for name, (groups, protected) in arms.items():
            factorizer = FinalLatentCacheFactorizer(groups, rank, protected)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = generate_official_codi(
                model, tokenizer, questions, latent_iterations=int(cfg.eval.latent_iterations),
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=device,
                kv_intervention=factorizer, answer_cue=style.answer_prefix, force_answer_cue=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            accuracy = _accuracy(outputs, rows)
            results[name] = {
                "accuracy": accuracy,
                "accuracy_retained_fraction": accuracy / dense_accuracy if dense_accuracy else None,
                "exact_sequence_agreement": sum(a == b for a, b in zip(outputs, dense_outputs)) / len(rows),
                "seconds_quality_proxy_including_factorization_and_dense_reconstruction": elapsed,
                "cache": factorizer.summary(), "outputs": outputs,
            }
            print(name, {key: value for key, value in results[name].items() if key != "outputs"})

    summary = {
        "schema_version": 1, "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction,
        "layerwise_gate": artifact["gate"],
        "exploratory_despite_failed_layerwise_gate": bool(not artifact["gate"]["passed"]),
        "examples": len(rows), "max_new_tokens": args.max_new_tokens,
        "groups": [list(group) for group in groups],
        "protected_validated_group": list(selected_group), "ranks": ranks,
        "timing_warning": (
            "These seconds include SVD plus dense reconstruction and are not an xKV speed benchmark. "
            "Only factor storage is modelled; production speed needs a fused reduced-attention kernel."
        ),
        "results": {name: {key: value for key, value in record.items() if key != "outputs"}
                    for name, record in results.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save({**summary, "outputs": {name: value["outputs"] for name, value in results.items()}},
                       args.output_dir / "results.pt")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = {"index": index, "question": row["question"], "gold": str(row["gold"]),
                      **{name: value["outputs"][index] for name, value in results.items()}}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--layerwise-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--ranks", default="28,32,48")
    parser.add_argument("--random-seed", type=int, default=20260912)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-unconfirmed", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
