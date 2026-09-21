"""Route each SVAMP request among frozen rank-64 adaptive xKV profiles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_codi_adaptive_kv_allocation import (
    CONTRACT as ADAPTIVE_CONTRACT,
    _adaptive_factorizer,
    _ordinary_factorizer,
)
from scripts.run_codi_rank16_xkv_mechanism_confirmation import _correctness
from scripts.run_codi_preanswer_kv_subspace_discovery import _sha
from src.data.datasets import load_eval_set
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.mech.rank16_xkv_confirmation import paired_bootstrap_interval
from src.mech.request_adaptive_kv_router import (
    deterministic_question_split,
    fit_ridge_loss_router,
    per_example_dense_kl,
    prompt_feature_matrix,
    route_distribution,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_request_adaptive_kv_router_svamp_v1"
PROFILE_WEIGHTS = (0.0, 0.5, 1.0)
PROFILE_NAMES = ("reconstruction", "hybrid", "answer_fisher")
BASELINE_RANK = 64
FIT_EXAMPLES = 400
SCREEN_EXAMPLES = 300
FINAL_EXAMPLES = 300
SPLIT_SEED = 20_260_921
RIDGE = 1.0


def _profile_factorizer(weight, latent_positions, reconstruction, fisher):
    return _adaptive_factorizer(
        rank=BASELINE_RANK,
        answer_weight=weight,
        latent_positions=latent_positions,
        reconstruction_utilities=reconstruction,
        fisher_utilities=fisher,
    )


def _first_token_logits(
    model, tokenizer, rows, *, style, latent_positions, batch_size, device,
    factorizer=None,
):
    observed = []

    def observer(logits, active, answer_position):
        if int(answer_position) != 0 or not bool(active.all()):
            raise RuntimeError("first-token audit observed an unexpected generation state")
        observed.append(logits.float().cpu())

    generate_official_codi(
        model,
        tokenizer,
        [row["question"] for row in rows],
        latent_iterations=latent_positions,
        max_new_tokens=1,
        batch_size=batch_size,
        device=device,
        kv_intervention=factorizer,
        answer_cue=style.answer_prefix,
        force_answer_cue=True,
        answer_logit_observer=observer,
    )
    logits = torch.cat(observed)
    if len(logits) != len(rows):
        raise RuntimeError("first-token logit count mismatch")
    return logits


def _generation(model, tokenizer, rows, *, style, latent_positions, batch_size,
                max_new_tokens, device, factorizer=None):
    outputs = generate_official_codi(
        model,
        tokenizer,
        [row["question"] for row in rows],
        latent_iterations=latent_positions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        kv_intervention=factorizer,
        answer_cue=style.answer_prefix,
        force_answer_cue=True,
    )
    correct = _correctness(outputs, rows)
    return outputs, correct


def _routed_generation(
    model, tokenizer, rows, routes, *, style, latent_positions, batch_size,
    max_new_tokens, device, reconstruction, fisher,
):
    outputs = [None] * len(rows)
    profile_outputs = {}
    cache_summaries = {}
    for profile_index, (name, weight) in enumerate(zip(PROFILE_NAMES, PROFILE_WEIGHTS)):
        factorizer = _profile_factorizer(weight, latent_positions, reconstruction, fisher)
        arm_outputs, _ = _generation(
            model, tokenizer, rows, style=style,
            latent_positions=latent_positions, batch_size=batch_size,
            max_new_tokens=max_new_tokens, device=device, factorizer=factorizer,
        )
        profile_outputs[name] = arm_outputs
        for index in (routes == profile_index).nonzero().flatten().tolist():
            outputs[index] = arm_outputs[index]
        cache_summaries[name] = factorizer.summary()
    if any(value is None for value in outputs):
        raise RuntimeError("router did not assign every request")
    return outputs, _correctness(outputs, rows), cache_summaries, profile_outputs


def _routed_logits(profile_logits, routes):
    result = torch.empty_like(profile_logits[0])
    for profile_index, logits in enumerate(profile_logits):
        mask = routes == profile_index
        result[mask] = logits[mask]
    return result


def _generation_record(outputs, correct, dense_outputs, dense_correct):
    return {
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "examples": len(outputs),
        "accuracy_retained_fraction": (
            float(correct.mean() / dense_correct.mean())
            if float(dense_correct.mean()) else None
        ),
        "exact_sequence_agreement": sum(
            value == reference for value, reference in zip(outputs, dense_outputs)
        ) / len(outputs),
    }


def _evaluate_split(
    model, tokenizer, rows, router, *, style, latent_positions, batch_size,
    generation_batch_size, max_new_tokens, device, reconstruction, fisher,
    seed, bootstrap_samples, minimum_fidelity, noninferiority_margin,
):
    features = prompt_feature_matrix([row["question"] for row in rows])
    routes = router.route(features)
    dense_logits = _first_token_logits(
        model, tokenizer, rows, style=style, latent_positions=latent_positions,
        batch_size=batch_size, device=device,
    )
    profile_logits = []
    for weight in PROFILE_WEIGHTS:
        profile_logits.append(_first_token_logits(
            model, tokenizer, rows, style=style, latent_positions=latent_positions,
            batch_size=batch_size, device=device,
            factorizer=_profile_factorizer(
                weight, latent_positions, reconstruction, fisher
            ),
        ))
    routed_logits = _routed_logits(profile_logits, routes)
    router_kl = per_example_dense_kl(dense_logits, routed_logits)
    global_kl = per_example_dense_kl(dense_logits, profile_logits[-1])
    kl_gain = global_kl - router_kl
    kl_interval = paired_bootstrap_interval(
        kl_gain, seed=seed, samples=bootstrap_samples
    )
    dense_top1 = dense_logits.argmax(-1)
    fidelity = float((routed_logits.argmax(-1) == dense_top1).double().mean())

    dense_outputs, dense_correct = _generation(
        model, tokenizer, rows, style=style, latent_positions=latent_positions,
        batch_size=generation_batch_size, max_new_tokens=max_new_tokens, device=device,
    )
    ordinary_factorizer = _ordinary_factorizer(
        rank=BASELINE_RANK, latent_positions=latent_positions
    )
    ordinary_outputs, ordinary_correct = _generation(
        model, tokenizer, rows, style=style, latent_positions=latent_positions,
        batch_size=generation_batch_size, max_new_tokens=max_new_tokens, device=device,
        factorizer=ordinary_factorizer,
    )
    routed_outputs, routed_correct, routed_cache, profile_outputs = _routed_generation(
        model, tokenizer, rows, routes, style=style,
        latent_positions=latent_positions, batch_size=generation_batch_size,
        max_new_tokens=max_new_tokens, device=device,
        reconstruction=reconstruction, fisher=fisher,
    )
    global_outputs = profile_outputs["answer_fisher"]
    global_correct = _correctness(global_outputs, rows)
    dense_interval = paired_bootstrap_interval(
        routed_correct - dense_correct, seed=seed + 1, samples=bootstrap_samples
    )
    ordinary_interval = paired_bootstrap_interval(
        routed_correct - ordinary_correct, seed=seed + 2, samples=bootstrap_samples
    )
    global_interval = paired_bootstrap_interval(
        routed_correct - global_correct, seed=seed + 3, samples=bootstrap_samples
    )
    ordinary_cache = ordinary_factorizer.summary()
    storage_ratios = {
        name: summary["cache_bits"] / max(1, ordinary_cache["cache_bits"])
        for name, summary in routed_cache.items()
    }
    distribution = route_distribution(routes, PROFILE_NAMES)
    gate = {
        "positive_kl_gain_interval": kl_interval[0] > 0,
        "first_token_fidelity": fidelity >= minimum_fidelity,
        "dense_accuracy_noninferior": dense_interval[0] >= -noninferiority_margin,
        "ordinary_accuracy_noninferior": ordinary_interval[0] >= -noninferiority_margin,
        "global_profile_accuracy_noninferior": global_interval[0] >= -noninferiority_margin,
        "storage_matched": all(abs(value - 1.0) <= 1e-12 for value in storage_ratios.values()),
        "nondegenerate_routing": distribution["active_profiles"] >= 2,
    }
    gate["passed"] = all(gate.values())
    record = {
        "examples": len(rows),
        "route_distribution": distribution,
        "mean_router_kl_from_dense": float(router_kl.mean()),
        "mean_global_profile_kl_from_dense": float(global_kl.mean()),
        "mean_kl_gain_over_global_profile": float(kl_gain.mean()),
        "kl_gain_bootstrap_95ci": kl_interval,
        "first_token_fidelity": fidelity,
        "accuracy_difference_vs_dense_95ci": dense_interval,
        "accuracy_difference_vs_ordinary_95ci": ordinary_interval,
        "accuracy_difference_vs_global_profile_95ci": global_interval,
        "generation": {
            "dense": _generation_record(dense_outputs, dense_correct, dense_outputs, dense_correct),
            "ordinary_xkv": _generation_record(ordinary_outputs, ordinary_correct, dense_outputs, dense_correct),
            "global_answer_fisher": _generation_record(global_outputs, global_correct, dense_outputs, dense_correct),
            "request_router": _generation_record(routed_outputs, routed_correct, dense_outputs, dense_correct),
        },
        "routed_cache": routed_cache,
        "ordinary_cache": ordinary_cache,
        "storage_ratios": storage_ratios,
        "gate": gate,
    }
    tensors = {
        "routes": routes.cpu(), "features": features.cpu(),
        "router_kl": router_kl, "global_kl": global_kl,
        "dense_correct": dense_correct.cpu(), "ordinary_correct": ordinary_correct.cpu(),
        "global_correct": global_correct.cpu(), "routed_correct": routed_correct.cpu(),
    }
    output_map = {
        "dense": dense_outputs, "ordinary_xkv": ordinary_outputs,
        "global_answer_fisher": global_outputs, "request_router": routed_outputs,
    }
    return record, tensors, output_map


def _validate_predecessor(artifact):
    if artifact.get("contract") != ADAPTIVE_CONTRACT:
        raise RuntimeError("wrong adaptive-allocation predecessor artifact")
    selected = artifact.get("selected_candidate") or {}
    if selected.get("name") != "adaptive_kv_r64_w1":
        raise RuntimeError("predecessor did not freeze adaptive_kv_r64_w1")
    final = artifact.get("final_replication") or {}
    if final.get("gate", {}).get("passed") is not False:
        raise RuntimeError("this follow-up requires the failed locked-final gate")
    if final.get("gate", {}).get("positive_nll_interval") is not False:
        raise RuntimeError("predecessor failure was not the expected NLL uncertainty")
    for key in ("reconstruction_utilities", "answer_fisher_utilities"):
        if key not in artifact:
            raise RuntimeError(f"predecessor artifact is missing {key}")


def run(args):
    if args.fit_examples != FIT_EXAMPLES or args.screen_examples != SCREEN_EXAMPLES \
            or args.final_examples != FINAL_EXAMPLES:
        raise ValueError("the preregistered SVAMP split cannot be changed")
    if args.baseline_rank != BASELINE_RANK or args.ridge != RIDGE:
        raise ValueError("the preregistered router or storage budget cannot be changed")
    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    artifact = torch.load(args.previous_artifact, map_location="cpu", weights_only=False)
    _validate_predecessor(artifact)
    reconstruction = artifact["reconstruction_utilities"]
    fisher = artifact["answer_fisher_utilities"]

    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256),
        token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
        dtype=dtype, settings=cfg.model, token=token,
    )
    load_report = load_official_checkpoint(
        model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
    )
    if load_report.checkpoint_sha256 != artifact["checkpoint_sha256"]:
        raise RuntimeError("checkpoint does not match the predecessor artifact")
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    all_rows = load_eval_set("svamp", data_cfg.eval.svamp)
    fit_rows, screen_rows, final_rows = deterministic_question_split(
        all_rows, seed=SPLIT_SEED,
        sizes=(FIT_EXAMPLES, SCREEN_EXAMPLES, FINAL_EXAMPLES),
    )
    split_hashes = {
        name: _sha([" ".join(row["question"].strip().split()) for row in rows])
        for name, rows in (("fit", fit_rows), ("screen", screen_rows), ("final", final_rows))
    }
    style = PromptStyle.from_config(data_cfg.prompt)
    latent_positions = int(cfg.eval.latent_iterations)

    fit_features = prompt_feature_matrix([row["question"] for row in fit_rows])
    fit_dense_logits = _first_token_logits(
        model, tokenizer, fit_rows, style=style, latent_positions=latent_positions,
        batch_size=args.batch_size, device=device,
    )
    fit_losses = []
    for weight in PROFILE_WEIGHTS:
        logits = _first_token_logits(
            model, tokenizer, fit_rows, style=style, latent_positions=latent_positions,
            batch_size=args.batch_size, device=device,
            factorizer=_profile_factorizer(weight, latent_positions, reconstruction, fisher),
        )
        fit_losses.append(per_example_dense_kl(fit_dense_logits, logits))
    fit_losses = torch.stack(fit_losses, dim=1)
    router = fit_ridge_loss_router(
        fit_features, fit_losses, profile_names=PROFILE_NAMES, ridge=RIDGE
    )
    fit_routes = router.route(fit_features)
    fit_oracle = fit_losses.argmin(1)
    fit_audit = {
        "examples": len(fit_rows),
        "mean_profile_kl": {
            name: float(fit_losses[:, index].mean())
            for index, name in enumerate(PROFILE_NAMES)
        },
        "router_oracle_route_agreement": float((fit_routes == fit_oracle).double().mean()),
        "router_distribution": route_distribution(fit_routes, PROFILE_NAMES),
    }

    screen, screen_tensors, screen_outputs = _evaluate_split(
        model, tokenizer, screen_rows, router, style=style,
        latent_positions=latent_positions, batch_size=args.batch_size,
        generation_batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens, device=device,
        reconstruction=reconstruction, fisher=fisher, seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        minimum_fidelity=args.minimum_first_token_fidelity,
        noninferiority_margin=args.accuracy_noninferiority_margin,
    )
    summary = {
        "schema_version": 1,
        "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction,
        "lineage": {
            "previous_contract": ADAPTIVE_CONTRACT,
            "previous_artifact": str(args.previous_artifact),
            "previous_selected_candidate": artifact["selected_candidate"],
            "previous_final_gate": artifact["final_replication"]["gate"],
        },
        "preregistration": {
            "dataset": "svamp",
            "holdout_status": (
                "external compression-method holdout; the dense CODI benchmark was previously reported"
            ),
            "split_seed": SPLIT_SEED,
            "split_sizes": {"router_fit": FIT_EXAMPLES, "screen": SCREEN_EXAMPLES, "final": FINAL_EXAMPLES},
            "profile_names": list(PROFILE_NAMES),
            "profile_answer_weights": list(PROFILE_WEIGHTS),
            "baseline_rank": BASELINE_RANK,
            "ridge": RIDGE,
            "minimum_first_token_fidelity": args.minimum_first_token_fidelity,
            "accuracy_noninferiority_margin": args.accuracy_noninferiority_margin,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "split_hashes": split_hashes,
        "router": router.audit(),
        "router_fit": fit_audit,
        "screen": screen,
        "final": None,
        "decision": {
            "screen_passed": bool(screen["gate"]["passed"]),
            "final_passed": False,
            "claim": (
                "ADVANCE: request routing passed the external screen"
                if screen["gate"]["passed"] else
                "STOP: request routing did not beat the frozen global profile on the external screen"
            ),
        },
        "warnings": [
            "SVAMP is external to the xKV method-development chain, but dense CODI accuracy on SVAMP was previously reported.",
            "The router sees only lexical question features available before generation; no gold answer or generated token is an input.",
            "Profile targets use dense first-token KL on the 400-row fit split only.",
            "All routed profiles use the same rank-64 modeled cache budget; this remains a dense-reconstruction quality proxy, not a native latency benchmark.",
        ],
    }
    tensors = {
        "fit_features": fit_features, "fit_profile_losses": fit_losses,
        "fit_routes": fit_routes, "screen": screen_tensors,
    }
    outputs = {"screen": screen_outputs}
    predictions = [
        {
            "split": "screen", "index": index, "question": row["question"],
            "gold": str(row["gold"]),
            **{name: values[index] for name, values in screen_outputs.items()},
            "routed_profile": PROFILE_NAMES[int(screen_tensors["routes"][index])],
        }
        for index, row in enumerate(screen_rows)
    ]

    if screen["gate"]["passed"]:
        final, final_tensors, final_outputs = _evaluate_split(
            model, tokenizer, final_rows, router, style=style,
            latent_positions=latent_positions, batch_size=args.batch_size,
            generation_batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens, device=device,
            reconstruction=reconstruction, fisher=fisher, seed=args.seed + 10_000,
            bootstrap_samples=args.bootstrap_samples,
            minimum_fidelity=args.minimum_first_token_fidelity,
            noninferiority_margin=args.accuracy_noninferiority_margin,
        )
        summary["final"] = final
        tensors["final"] = final_tensors
        outputs["final"] = final_outputs
        summary["decision"] = {
            "screen_passed": True,
            "final_passed": bool(final["gate"]["passed"]),
            "claim": (
                "CONFIRMED: pre-answer request routing improves the frozen global xKV profile"
                if final["gate"]["passed"] else
                "STOP: request routing did not replicate on the locked SVAMP final split"
            ),
        }
        for index, row in enumerate(final_rows):
            predictions.append({
                "split": "final", "index": index, "question": row["question"],
                "gold": str(row["gold"]),
                **{name: values[index] for name, values in final_outputs.items()},
                "routed_profile": PROFILE_NAMES[int(final_tensors["routes"][index])],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save(
        {**summary, "router_state": router.audit(), "tensors": tensors, "outputs": outputs},
        args.output_dir / "request_adaptive_kv_router.pt",
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--previous-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--fit-examples", type=int, default=FIT_EXAMPLES)
    parser.add_argument("--screen-examples", type=int, default=SCREEN_EXAMPLES)
    parser.add_argument("--final-examples", type=int, default=FINAL_EXAMPLES)
    parser.add_argument("--baseline-rank", type=int, default=BASELINE_RANK)
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--minimum-first-token-fidelity", type=float, default=0.95)
    parser.add_argument("--accuracy-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
