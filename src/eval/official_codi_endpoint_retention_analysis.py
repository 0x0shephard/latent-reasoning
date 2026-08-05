"""Paired accuracy analysis for the CODI endpoint-retention experiment."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from src.mech.endpoint_retention import RETENTION_METHODS, RETENTION_TRAINING_ARMS


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _paired_hierarchical_delta(
    left: np.ndarray,
    right: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Resample training seeds, then paired GSM8K questions within each seed."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("paired correctness arrays must have shape [seeds, examples]")
    rng = np.random.default_rng(seed)
    seed_count, example_count = left.shape
    observed = float((left - right).mean())
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for draw in range(bootstrap_samples):
        seed_ids = rng.integers(0, seed_count, size=seed_count)
        total = 0.0
        for seed_id in seed_ids:
            example_ids = rng.integers(0, example_count, size=example_count)
            total += float(
                (left[seed_id, example_ids] - right[seed_id, example_ids]).mean()
            )
        samples[draw] = total / seed_count
    return {
        "delta_accuracy": observed,
        "delta_percentage_points": 100.0 * observed,
        "bootstrap_95_ci": _interval(samples),
        "bootstrap_95_ci_percentage_points": [
            100.0 * value for value in _interval(samples)
        ],
    }


def analyze_endpoint_retention(
    runs: Sequence[dict],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    noninferiority_margin: float = 0.01,
) -> dict:
    """Aggregate all rank-matched arms without treating seeds as independent rows."""
    if bootstrap_samples <= 0 or not 0 < noninferiority_margin < 1:
        raise ValueError("bootstrap count and noninferiority margin must be positive")
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for run in runs:
        arm = str(run["arm"])
        seed = int(run["training_seed"])
        if arm not in RETENTION_TRAINING_ARMS:
            raise ValueError(f"unknown retention arm {arm!r}")
        if seed in grouped[arm]:
            raise ValueError(f"duplicate retention run for {arm}/seed{seed}")
        grouped[arm][seed] = run
    if set(grouped) != set(RETENTION_TRAINING_ARMS):
        raise ValueError("analysis requires every registered retention arm")
    seed_sets = {arm: tuple(sorted(values)) for arm, values in grouped.items()}
    if len(set(seed_sets.values())) != 1:
        raise ValueError("every arm must use the same training seeds")
    seeds = next(iter(seed_sets.values()))
    if not seeds:
        raise ValueError("no completed training seeds")

    arrays: dict[str, np.ndarray] = {}
    summaries = {}
    expected_questions = None
    for arm in RETENTION_TRAINING_ARMS:
        rows = []
        throughputs = []
        for seed in seeds:
            run = grouped[arm][seed]
            correctness = np.asarray(run["correctness"], dtype=np.float64)
            if correctness.ndim != 1 or correctness.size == 0:
                raise ValueError("correctness must be a nonempty vector")
            if expected_questions is None:
                expected_questions = correctness.size
            if correctness.size != expected_questions:
                raise ValueError("all runs must evaluate the same questions")
            rows.append(correctness)
            throughputs.append(float(run["examples_per_second"]))
        matrix = np.stack(rows)
        arrays[arm] = matrix
        per_seed = matrix.mean(axis=1)
        summaries[arm] = {
            "mean_accuracy": float(matrix.mean()),
            "mean_accuracy_percent": 100.0 * float(matrix.mean()),
            "per_seed_accuracy": {
                str(seed): float(value) for seed, value in zip(seeds, per_seed)
            },
            "mean_examples_per_second": float(np.mean(throughputs)),
        }

    versus_full = {}
    versus_answer_only = {}
    selected_versus_complement = {}
    for index, method in enumerate(RETENTION_METHODS):
        selected = f"{method}_selected"
        complement = f"{method}_complement"
        comparison = _paired_hierarchical_delta(
            arrays[selected],
            arrays["full_common"],
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + index * 2,
        )
        lower = comparison["bootstrap_95_ci"][0]
        comparison["accuracy_loss_from_full"] = -comparison["delta_accuracy"]
        comparison["accuracy_loss_from_full_percentage_points"] = (
            -comparison["delta_percentage_points"]
        )
        comparison["noninferiority_margin"] = noninferiority_margin
        comparison["noninferior_to_full_common"] = bool(
            lower > -noninferiority_margin
        )
        versus_full[method] = comparison
        selected_versus_complement[method] = _paired_hierarchical_delta(
            arrays[selected],
            arrays[complement],
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + index * 2 + 1,
        )
        versus_answer_only[method] = _paired_hierarchical_delta(
            arrays[selected],
            arrays["answer_only"],
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 100 + index,
        )

    selected_accuracies = {
        method: summaries[f"{method}_selected"]["mean_accuracy"]
        for method in RETENTION_METHODS
    }
    winner = max(selected_accuracies, key=selected_accuracies.get)
    throughput_values = [
        summaries[arm]["mean_examples_per_second"]
        for arm in RETENTION_TRAINING_ARMS
    ]
    return {
        "analysis": "official_codi_endpoint_rank_matched_retention",
        "training_seeds": list(seeds),
        "evaluated_examples_per_run": int(expected_questions),
        "bootstrap_samples": bootstrap_samples,
        "noninferiority_margin": noninferiority_margin,
        "arms": summaries,
        "selected_vs_full_common": versus_full,
        "selected_vs_answer_only": versus_answer_only,
        "full_common_vs_answer_only": _paired_hierarchical_delta(
            arrays["full_common"],
            arrays["answer_only"],
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + 200,
        ),
        "selected_vs_own_complement": selected_versus_complement,
        "highest_selected_accuracy": {
            "method": winner,
            "accuracy": selected_accuracies[winner],
        },
        "inference_speed_interpretation": {
            "same_architecture_in_all_arms": True,
            "expected_speedup_from_retention": False,
            "observed_throughput_range": [
                float(min(throughput_values)), float(max(throughput_values))
            ],
            "reason": (
                "The frozen bases affect only the training loss; generation still runs "
                "all 12 GPT-2 blocks at width 768."
            ),
        },
    }
