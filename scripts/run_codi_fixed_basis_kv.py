"""Fit fixed per-head K/V bases once, then run CODI under exact projected attention.

The experiment asks whether CODI's cache has a fixed low-dimensional working
geometry per layer and head, in the same sense that the final state has a fixed
answer subspace (ledger 40, 65, 67).  Nothing is factorized per request: every
key and value a head writes is projected onto that head's leading ``r``
calibration directions, which is algebraically an ``r``-dimensional attention.
"""
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
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    verify_full_reproduction_gate,
)
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES,
    GSM8K_TRAIN_URL,
    sample_gsm8k_train_calibration,
)
from scripts.run_codi_preanswer_kv_subspace_discovery import _evaluate, _metrics, _sha
from src.data.datasets import load_eval_set
from src.data.official_codi_training import align_official_codi_gsm8k_eval_rows
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.fixed_basis_kv import (
    KINDS,
    ROW_TYPES,
    HeadBases,
    HeadProjectionHook,
    KVSecondMomentCollector,
    allocate_energy_ranks,
    collect_kv_second_moments,
    energy_fraction_curves,
    fit_head_bases,
    head_geometry,
    projectors_from_ranks,
    random_head_bases,
    rank_for_energy,
    storage_report,
    subspace_capture,
    uniform_ranks,
)
from src.mech.rank16_xkv_confirmation import paired_bootstrap_interval
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_fixed_basis_kv_v1"
BASIS_FIT_EXAMPLES = 1_024
SELECTION_EXAMPLES = 256
SAMPLING_SEED = 20_260_921
RANK_GRID = (8, 16, 24, 32, 40, 48)
RANDOM_BASIS_SEEDS = (20_260_921, 20_260_922)
NEIGHBOR_STEP = 8
MINIMUM_RETENTION = 0.98
MINIMUM_FIRST_TOKEN_FIDELITY = 0.95
ACCURACY_NONINFERIORITY_MARGIN = 0.02
ENERGY_TARGETS = (0.90, 0.95, 0.99)
SELECTION_FAMILIES = ("uniform", "energy")


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _with_gold(rows):
    return [{**row, "gold": str(row.get("gold", row["answer"]))} for row in rows]


def _correctness(outputs, rows) -> torch.Tensor:
    return torch.tensor(
        [official_answers_match(text, row["gold"]) for text, row in zip(outputs, rows)],
        dtype=torch.float32,
    )


def _serial(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if not key.startswith("per_")}


def arm_name(family: str, rank: int, seed: int | None = None) -> str:
    if family == "dense":
        return "dense"
    if family == "random":
        if seed is None:
            raise ValueError("random arms need a seed")
        return f"random_s{seed}_r{rank}"
    if family not in ("uniform", "energy", "key_only", "value_only"):
        raise ValueError(f"unknown arm family {family!r}")
    return f"{family}_r{rank}"


def select_operating_point(
    arms: dict, *, rank_grid=RANK_GRID, families=SELECTION_FAMILIES,
    minimum_retention=MINIMUM_RETENTION,
    minimum_fidelity=MINIMUM_FIRST_TOKEN_FIDELITY,
) -> dict:
    """Smallest rank, then family order, whose selection-split arm passes both checks."""
    for rank in sorted(rank_grid):
        for family in families:
            record = arms[arm_name(family, rank)]
            retention = record.get("accuracy_retained_fraction")
            fidelity = record.get("dense_first_token_top1_agreement")
            if retention is None or fidelity is None:
                continue
            if retention >= minimum_retention and fidelity >= minimum_fidelity:
                return {"passed": True, "family": family, "rank": int(rank)}
    return {"passed": False, "family": families[0], "rank": int(max(rank_grid))}


def final_arm_specs(
    selected_rank: int, *, rank_grid=RANK_GRID, seeds=RANDOM_BASIS_SEEDS,
    step=NEIGHBOR_STEP,
) -> list[tuple[str, int, int | None]]:
    """Locked final arms: selected budget, its controls, and grid neighbours."""
    specs: list[tuple[str, int, int | None]] = [
        ("uniform", selected_rank, None), ("energy", selected_rank, None),
    ]
    specs.extend(("random", selected_rank, seed) for seed in seeds)
    specs.extend([("key_only", selected_rank, None), ("value_only", selected_rank, None)])
    for neighbour in (selected_rank - step, selected_rank + step):
        if neighbour in rank_grid:
            specs.extend([("uniform", neighbour, None), ("energy", neighbour, None)])
    return specs


def _ranks_and_bases(family, rank, *, bases: HeadBases, geometry, seed=None):
    layers, heads, head_dim = geometry
    if family == "uniform":
        return uniform_ranks(layers=layers, heads=heads, rank=rank), bases
    if family == "energy":
        total = layers * len(KINDS) * heads * int(rank)
        return allocate_energy_ranks(bases.eigenvalues, total_rank=total), bases
    if family == "random":
        random_bases = random_head_bases(
            layers=layers, kinds=len(KINDS), heads=heads, head_dim=head_dim, seed=seed
        )
        return uniform_ranks(layers=layers, heads=heads, rank=rank), random_bases
    if family == "key_only":
        return uniform_ranks(
            layers=layers, heads=heads, rank=rank, key_rank=rank, value_rank=head_dim
        ), bases
    if family == "value_only":
        return uniform_ranks(
            layers=layers, heads=heads, rank=rank, key_rank=head_dim, value_rank=rank
        ), bases
    raise ValueError(f"unknown arm family {family!r}")


def _build_hook(family, rank, *, bases, geometry, seed=None) -> HeadProjectionHook:
    ranks, used = _ranks_and_bases(family, rank, bases=bases, geometry=geometry, seed=seed)
    return HeadProjectionHook(projectors_from_ranks(used, ranks), ranks=ranks)


def _run_arm(
    model, tokenizer, rows, *, hook, style, latent_positions, generation_batch_size,
    batch_size, max_new_tokens, device, dense=None,
):
    """Generate and teacher-force one arm; ``hook=None`` is the dense reference."""
    if hook is not None:
        hook.attach(model)
    try:
        outputs = generate_official_codi(
            model, tokenizer, [row["question"] for row in rows],
            latent_iterations=latent_positions, max_new_tokens=max_new_tokens,
            batch_size=generation_batch_size, device=device,
            answer_cue=style.answer_prefix, force_answer_cue=True,
        )
        losses, logits, targets = _evaluate(
            model, tokenizer, rows, latent_positions, batch_size, device, None,
            enforce_answer_eligibility=False,
        )
    finally:
        if hook is not None:
            hook.detach()
    correct = _correctness(outputs, rows)
    record = {
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "examples": len(rows),
    }
    teacher = {"losses": losses, "logits": logits, "targets": targets}
    if dense is None:
        record.update(_serial(_metrics(losses, logits, targets, losses, logits)))
        record["accuracy_retained_fraction"] = 1.0
        record["exact_sequence_agreement"] = 1.0
    else:
        record.update(_serial(_metrics(
            losses, logits, targets, dense["teacher"]["losses"], dense["teacher"]["logits"]
        )))
        dense_accuracy = float(dense["correct"].mean())
        record["accuracy_retained_fraction"] = (
            float(correct.mean() / dense_accuracy) if dense_accuracy else None
        )
        record["exact_sequence_agreement"] = sum(
            current == reference for current, reference in zip(outputs, dense["outputs"])
        ) / len(outputs)
        record["storage"] = hook.summary()
    return record, {"correct": correct, "outputs": outputs, "teacher": teacher}


def _interval(values: torch.Tensor, *, seed: int, samples: int) -> list[float]:
    return paired_bootstrap_interval(values, seed=seed, samples=samples)


def _component_table(values: torch.Tensor) -> dict:
    """Mean over heads of a ``[layers, kinds, heads]`` tensor, keyed by layer/kind."""
    result = {}
    for layer in range(values.shape[0]):
        for kind_index, kind in enumerate(KINDS):
            result[f"layer_{layer:02d}_{kind}"] = float(values[layer, kind_index].double().mean())
    return result


def _calibration_diagnostics(collector, bases: HeadBases, geometry, rank_grid):
    layers, heads, head_dim = geometry
    per_type_bases = {
        name: fit_head_bases(collector.row_type_moments(name)) for name in ROW_TYPES
        if int(collector.counts[ROW_TYPES.index(name)]) > 0
    }
    energy_ranks = {
        f"rank_for_{int(target * 100)}pct_energy": _component_table(
            rank_for_energy(bases.eigenvalues, target).double()
        ) for target in ENERGY_TARGETS
    }
    pooled_curves = energy_fraction_curves(bases.eigenvalues)
    retained_at_rank = {
        str(rank): _component_table(pooled_curves[..., rank - 1]) for rank in rank_grid
    }
    capture = {}
    for other in ("latent", "cue", "answer"):
        if other not in per_type_bases or "question" not in per_type_bases:
            continue
        capture[f"{other}_vs_question"] = {
            str(rank): {
                **_component_table(subspace_capture(
                    per_type_bases["question"].vectors, per_type_bases[other].vectors, rank
                )),
                "isotropic_expectation": rank / head_dim,
            }
            for rank in rank_grid
        }
    allocation = {}
    for rank in rank_grid:
        ranks = allocate_energy_ranks(
            bases.eigenvalues, total_rank=layers * len(KINDS) * heads * rank
        )
        allocation[str(rank)] = {
            "mean_rank_by_component": _component_table(ranks.double()),
            "minimum_rank": int(ranks.min()), "maximum_rank": int(ranks.max()),
            "storage": storage_report(ranks, head_dim=head_dim),
        }
    return {
        "row_counts": collector.count_summary(),
        "energy_ranks": energy_ranks,
        "retained_energy_at_rank": retained_at_rank,
        "row_type_capture": capture,
        "energy_allocation_by_budget": allocation,
        "basis_note": (
            "uncentered per-head second moments pooled over question, latent, cue and "
            "answer rows; eigenvectors ordered by descending eigenvalue"
        ),
    }, per_type_bases


def _save(args, summary, tensors, outputs, predictions):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save(
        {**summary, **tensors, "outputs": outputs}, args.output_dir / "fixed_basis_kv.pt"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    if _parse_ints(args.rank_grid) != RANK_GRID:
        raise ValueError("the preregistered rank grid cannot be changed")
    if _parse_ints(args.random_basis_seeds) != RANDOM_BASIS_SEEDS:
        raise ValueError("the preregistered random-basis seeds cannot be changed")
    if (
        args.basis_fit_examples != BASIS_FIT_EXAMPLES
        or args.selection_examples != SELECTION_EXAMPLES
        or args.sampling_seed != SAMPLING_SEED
    ):
        raise ValueError("the preregistered calibration protocol cannot be changed")
    if (
        args.minimum_retention != MINIMUM_RETENTION
        or args.minimum_first_token_fidelity != MINIMUM_FIRST_TOKEN_FIDELITY
        or args.accuracy_noninferiority_margin != ACCURACY_NONINFERIORITY_MARGIN
    ):
        raise ValueError("the preregistered gate thresholds cannot be changed")

    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
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
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    geometry = head_geometry(model)
    layers, heads, head_dim = geometry
    if max(RANK_GRID) >= head_dim:
        raise RuntimeError("the rank grid must stay strictly below the head width")

    from datasets import load_dataset

    train = load_dataset(
        "json", data_files={"train": GSM8K_TRAIN_URL}, split="train",
        verification_mode="no_checks",
    )
    if len(train) != GSM8K_EXPECTED_TRAIN_EXAMPLES:
        raise RuntimeError("GSM8K train count drifted")
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    test_questions = {_normalized_question(row["question"]) for row in test}
    sampled, sampling = sample_gsm8k_train_calibration(
        train, test_questions=test_questions,
        examples=BASIS_FIT_EXAMPLES + SELECTION_EXAMPLES, seed=SAMPLING_SEED,
    )
    fit_rows = _with_gold(sampled[:BASIS_FIT_EXAMPLES])
    selection_rows = _with_gold(sampled[BASIS_FIT_EXAMPLES:])
    spec = data_cfg.eval.gsm8k
    raw_test = load_dataset(
        str(spec.hf_id), data_files={str(spec.get("split", "test")): str(spec.data_file)},
        split=str(spec.get("split", "test")), verification_mode="no_checks",
    )
    test_rows = align_official_codi_gsm8k_eval_rows(
        raw_test, test, enforce_answer_eligibility=False,
    )
    if len(test_rows) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("the complete GSM8K test set could not be aligned")
    split_hashes = {
        "basis_fit": _sha([_normalized_question(row["question"]) for row in fit_rows]),
        "selection": _sha([_normalized_question(row["question"]) for row in selection_rows]),
        "test": _sha([_normalized_question(row["question"]) for row in test_rows]),
    }
    style = PromptStyle.from_config(data_cfg.prompt)
    latent_positions = int(cfg.eval.latent_iterations)

    collector = KVSecondMomentCollector(layers=layers, heads=heads, head_dim=head_dim)
    collector.attach(model)
    try:
        calibration = collect_kv_second_moments(
            model, tokenizer, [row["question"] for row in fit_rows],
            collector=collector, latent_iterations=latent_positions,
            max_new_tokens=args.max_new_tokens, batch_size=args.generation_batch_size,
            device=device, answer_cue=style.answer_prefix,
        )
    finally:
        collector.detach()
    calibration_accuracy = float(_correctness(calibration["outputs"], fit_rows).mean())
    bases = fit_head_bases(collector.pooled_moments())
    diagnostics, per_type_bases = _calibration_diagnostics(
        collector, bases, geometry, RANK_GRID
    )
    diagnostics["calibration_generation_accuracy"] = calibration_accuracy
    print("calibration rows", diagnostics["row_counts"], "accuracy", calibration_accuracy)

    common = dict(
        style=style, latent_positions=latent_positions,
        generation_batch_size=args.generation_batch_size, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, device=device,
    )
    selection_dense, selection_dense_state = _run_arm(
        model, tokenizer, selection_rows, hook=None, **common
    )
    selection_arms, selection_state = {}, {}
    selection_specs: list[tuple[str, int, int | None]] = []
    for rank in RANK_GRID:
        selection_specs.append(("uniform", rank, None))
        selection_specs.append(("energy", rank, None))
        selection_specs.extend(("random", rank, seed) for seed in RANDOM_BASIS_SEEDS)
    for family, rank, seed in selection_specs:
        name = arm_name(family, rank, seed)
        record, state = _run_arm(
            model, tokenizer, selection_rows,
            hook=_build_hook(family, rank, bases=bases, geometry=geometry, seed=seed),
            dense=selection_dense_state, **common,
        )
        record["family"], record["rank"], record["seed"] = family, int(rank), seed
        selection_arms[name], selection_state[name] = record, state
        print("selection", name, {
            "accuracy": record["accuracy"], "retained": record["accuracy_retained_fraction"],
            "fidelity": record["dense_first_token_top1_agreement"],
        })
    for rank in RANK_GRID:
        for family in SELECTION_FAMILIES:
            record = selection_arms[arm_name(family, rank)]
            record["random_control_accuracies"] = [
                selection_arms[arm_name("random", rank, seed)]["accuracy"]
                for seed in RANDOM_BASIS_SEEDS
            ]
            record["accuracy_difference_vs_dense_95ci"] = _interval(
                selection_state[arm_name(family, rank)]["correct"]
                - selection_dense_state["correct"],
                seed=args.seed + 1_000 + rank, samples=args.bootstrap_samples,
            )
    selected = select_operating_point(selection_arms)
    selected_rank = int(selected["rank"])
    selected_name = arm_name(selected["family"], selected_rank)
    print("selected", selected)

    summary = {
        "schema_version": 1,
        "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction,
        "geometry": {"layers": layers, "heads": heads, "head_dim": head_dim},
        "preregistration": {
            "basis_fit_examples": BASIS_FIT_EXAMPLES,
            "selection_examples": SELECTION_EXAMPLES,
            "sampling_seed": SAMPLING_SEED,
            "rank_grid": list(RANK_GRID),
            "random_basis_seeds": list(RANDOM_BASIS_SEEDS),
            "selection_rule": (
                "smallest rank, uniform before energy-allocated, whose selection-split "
                "arm retains at least 98% of dense exact match and at least 95% dense "
                "first-token agreement; if no rank passes, the test set is not read"
            ),
            "final_arms": [
                arm_name(family, rank, seed)
                for family, rank, seed in final_arm_specs(selected_rank)
            ],
            "final_gate": [
                "paired exact-match difference versus dense has 95% lower bound above -2 points",
                "selected arm beats every seeded random basis at equal rank with a positive lower bound",
                "dense first-token top-1 agreement at least 95%",
                "point retention at least 98% of dense exact match",
            ],
            "bootstrap_samples": args.bootstrap_samples,
            "minimum_retention": MINIMUM_RETENTION,
            "minimum_first_token_fidelity": MINIMUM_FIRST_TOKEN_FIDELITY,
            "accuracy_noninferiority_margin": ACCURACY_NONINFERIORITY_MARGIN,
        },
        "sampling": sampling,
        "split_hashes": split_hashes,
        "calibration": diagnostics,
        "selection": {"dense": selection_dense, "arms": selection_arms},
        "selected": {**selected, "name": selected_name},
        "final": None,
        "decision": {
            "selection_passed": bool(selected["passed"]),
            "final_passed": False,
            "claim": (
                "pending final evaluation" if selected["passed"] else
                "STOP: no rank in the frozen grid preserved dense accuracy on the "
                "selection split; the test set was not read"
            ),
        },
        "warnings": [
            "Bases are fitted on GSM8K-train questions disjoint from the test set; no test question enters fitting or rank selection.",
            "The complete GSM8K test set was opened by earlier method families (ledger 75-79); for this method it is read once, after the operating point is frozen on the selection split.",
            "The projection hook is the exact unfused reference for r-dimensional per-head attention; wall-clock and memory are not measured here.",
            "Compression ratios are head_dim / rank per component; fixed bases are model metadata counted separately as basis_parameters.",
        ],
    }
    tensors = {
        "second_moments_by_row_type": collector.moments,
        "row_type_counts": collector.counts,
        "pooled_basis_vectors": bases.vectors,
        "pooled_eigenvalues": bases.eigenvalues,
        "row_type_eigenvalues": {
            name: value.eigenvalues for name, value in per_type_bases.items()
        },
        "selection_correct": {
            "dense": selection_dense_state["correct"],
            **{name: state["correct"] for name, state in selection_state.items()},
        },
    }
    outputs = {
        "calibration": calibration["outputs"],
        "selection": {
            "dense": selection_dense_state["outputs"],
            **{name: state["outputs"] for name, state in selection_state.items()},
        },
    }
    predictions: list[dict] = []
    _save(args, summary, tensors, outputs, predictions)
    if not selected["passed"]:
        # Sibling experiments leave the locked split unopened when the screen fails,
        # and a descriptive read here could not support a claim either way.
        print(json.dumps(summary["decision"], indent=2))
        return summary

    final_dense, final_dense_state = _run_arm(model, tokenizer, test_rows, hook=None, **common)
    final_arms, final_state = {}, {}
    for family, rank, seed in final_arm_specs(selected_rank):
        name = arm_name(family, rank, seed)
        record, state = _run_arm(
            model, tokenizer, test_rows,
            hook=_build_hook(family, rank, bases=bases, geometry=geometry, seed=seed),
            dense=final_dense_state, **common,
        )
        record["family"], record["rank"], record["seed"] = family, int(rank), seed
        record["accuracy_difference_vs_dense_95ci"] = _interval(
            state["correct"] - final_dense_state["correct"],
            seed=args.seed + 50_000 + len(final_arms), samples=args.bootstrap_samples,
        )
        final_arms[name], final_state[name] = record, state
        print("final", name, {
            "accuracy": record["accuracy"], "retained": record["accuracy_retained_fraction"],
            "fidelity": record["dense_first_token_top1_agreement"],
        })
    selected_record = final_arms[selected_name]
    selected_correct = final_state[selected_name]["correct"]
    comparisons = {
        "selected_vs_random": {
            arm_name("random", selected_rank, seed): _interval(
                selected_correct - final_state[arm_name("random", selected_rank, seed)]["correct"],
                seed=args.seed + 60_000 + index, samples=args.bootstrap_samples,
            ) for index, seed in enumerate(RANDOM_BASIS_SEEDS)
        },
        "energy_minus_uniform_at_selected_rank": _interval(
            final_state[arm_name("energy", selected_rank)]["correct"]
            - final_state[arm_name("uniform", selected_rank)]["correct"],
            seed=args.seed + 61_000, samples=args.bootstrap_samples,
        ),
        "key_only_minus_value_only_at_selected_rank": _interval(
            final_state[arm_name("key_only", selected_rank)]["correct"]
            - final_state[arm_name("value_only", selected_rank)]["correct"],
            seed=args.seed + 62_000, samples=args.bootstrap_samples,
        ),
    }
    gate = {
        "dense_accuracy_noninferior": (
            selected_record["accuracy_difference_vs_dense_95ci"][0]
            > -ACCURACY_NONINFERIORITY_MARGIN
        ),
        "beats_every_random_basis": all(
            interval[0] > 0 for interval in comparisons["selected_vs_random"].values()
        ),
        "first_token_fidelity": (
            selected_record["dense_first_token_top1_agreement"] >= MINIMUM_FIRST_TOKEN_FIDELITY
        ),
        "point_retention": (
            (selected_record["accuracy_retained_fraction"] or 0.0) >= MINIMUM_RETENTION
        ),
    }
    gate["passed"] = all(gate.values())
    summary["final"] = {
        "dense": final_dense,
        "arms": final_arms,
        "selected_arm": selected_name,
        "comparisons": comparisons,
        "gate": gate,
    }
    ratio = selected_record["storage"]["compression_ratio"]
    summary["decision"] = {
        "selection_passed": bool(selected["passed"]),
        "final_passed": bool(gate["passed"]),
        "claim": (
            f"CONFIRMED: fixed per-head K/V bases at rank {selected_rank} "
            f"({ratio:.1f}x per-token cache reduction) retain dense CODI accuracy "
            "and beat random bases"
            if gate["passed"] else
            f"STOP: fixed per-head K/V bases at rank {selected_rank} did not pass the "
            "locked final gate"
        ),
    }
    tensors["final_correct"] = {
        "dense": final_dense_state["correct"],
        **{name: state["correct"] for name, state in final_state.items()},
    }
    outputs["final"] = {
        "dense": final_dense_state["outputs"],
        **{name: state["outputs"] for name, state in final_state.items()},
    }
    for index, row in enumerate(test_rows):
        predictions.append({
            "split": "test", "dataset_index": index,
            "question": row["question"], "gold": str(row["gold"]),
            "dense": final_dense_state["outputs"][index],
            **{name: state["outputs"][index] for name, state in final_state.items()},
        })
    _save(args, summary, tensors, outputs, predictions)
    print(json.dumps(summary["decision"], indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--basis-fit-examples", type=int, default=BASIS_FIT_EXAMPLES)
    parser.add_argument("--selection-examples", type=int, default=SELECTION_EXAMPLES)
    parser.add_argument("--sampling-seed", type=int, default=SAMPLING_SEED)
    parser.add_argument("--rank-grid", default=",".join(map(str, RANK_GRID)))
    parser.add_argument("--random-basis-seeds", default=",".join(map(str, RANDOM_BASIS_SEEDS)))
    parser.add_argument("--minimum-retention", type=float, default=MINIMUM_RETENTION)
    parser.add_argument(
        "--minimum-first-token-fidelity", type=float, default=MINIMUM_FIRST_TOKEN_FIDELITY
    )
    parser.add_argument(
        "--accuracy-noninferiority-margin", type=float, default=ACCURACY_NONINFERIORITY_MARGIN
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20_260_921)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
