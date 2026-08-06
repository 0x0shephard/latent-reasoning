"""Paired causal analysis for frozen CODI answer-colon hidden-state ablations."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _bootstrap_loss(
    baseline: np.ndarray,
    ablated: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    difference = baseline.astype(np.float64) - ablated.astype(np.float64)
    indices = rng.integers(0, difference.size, size=(samples, difference.size))
    draws = difference[indices].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _mcnemar_one_sided(losses: int, gains: int) -> float:
    discordant = losses + gains
    if discordant == 0 or losses <= gains:
        return 1.0
    numerator = sum(math.comb(discordant, value) for value in range(losses, discordant + 1))
    return float(numerator / (2**discordant))


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    total = len(ordered)
    adjusted = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, (total - index) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def _comparison(
    baseline: np.ndarray,
    ablated: np.ndarray,
    reached: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if baseline.shape != ablated.shape or baseline.shape != reached.shape:
        raise ValueError("paired ablation arrays have inconsistent shapes")
    losses = int(np.logical_and(baseline, np.logical_not(ablated)).sum())
    gains = int(np.logical_and(np.logical_not(baseline), ablated).sum())
    delta = float((baseline.astype(float) - ablated.astype(float)).mean())
    reached_delta = float(
        (baseline[reached].astype(float) - ablated[reached].astype(float)).mean()
    ) if reached.any() else 0.0
    halves = []
    for mask in (np.arange(baseline.size) % 2 == 0, np.arange(baseline.size) % 2 == 1):
        halves.append(float((baseline[mask].astype(float) - ablated[mask].astype(float)).mean()))
    return {
        "baseline_accuracy": float(baseline.mean()),
        "ablated_accuracy": float(ablated.mean()),
        "accuracy_loss": delta,
        "accuracy_loss_percentage_points": 100.0 * delta,
        "bootstrap_95_ci": _bootstrap_loss(
            baseline, ablated, samples=bootstrap_samples, seed=seed
        ),
        "baseline_correct_to_ablated_wrong": losses,
        "baseline_wrong_to_ablated_correct": gains,
        "mcnemar_one_sided_p": _mcnemar_one_sided(losses, gains),
        "accuracy_loss_on_cue_reached_examples": reached_delta,
        "deterministic_half_losses": halves,
        "positive_in_both_halves": bool(min(halves) > 0),
    }


def analyze_endpoint_inference_ablation(
    runs: Sequence[dict],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    familywise_alpha: float = 0.05,
    minimum_endpoint_coverage: float = 0.95,
) -> dict:
    if (
        bootstrap_samples <= 0
        or not 0 < familywise_alpha < 1
        or not 0 < minimum_endpoint_coverage <= 1
    ):
        raise ValueError("bootstrap samples must be positive and alpha in (0,1)")
    by_arm = {}
    for run in runs:
        arm = str(run["arm"])
        if arm in by_arm:
            raise ValueError(f"duplicate ablation arm {arm}")
        by_arm[arm] = run
    if "baseline" not in by_arm:
        raise ValueError("frozen-checkpoint baseline is required")
    baseline = np.asarray(by_arm["baseline"]["correctness"], dtype=bool)
    reached = np.asarray(by_arm["baseline"]["endpoint_reached"], dtype=bool)
    native_accuracy_value = by_arm["baseline"].get("native_reproduction_accuracy")
    native_accuracy = (
        None if native_accuracy_value is None else float(native_accuracy_value)
    )
    if baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("baseline correctness must be a nonempty vector")
    if float(reached.mean()) < minimum_endpoint_coverage:
        raise RuntimeError(
            "exact answer-cue coverage is below the preregistered causal gate"
        )
    comparisons = {}
    candidate_names = []
    joint_random = []
    single_random: dict[int, list[float]] = {11: [], 12: []}
    for index, (arm, run) in enumerate(sorted(by_arm.items())):
        if arm == "baseline":
            continue
        values = np.asarray(run["correctness"], dtype=bool)
        arm_reached = np.asarray(run["endpoint_reached"], dtype=bool)
        if values.shape != baseline.shape or not np.array_equal(arm_reached, reached):
            raise ValueError(f"{arm} is not paired to the same endpoint-reach vector")
        family = (run.get("spec") or {}).get("family")
        comparison = _comparison(
            baseline,
            values,
            reached,
            bootstrap_samples=(
                bootstrap_samples
                if family in {"selected_joint", "selected_single"}
                else min(1_000, bootstrap_samples)
            ),
            seed=bootstrap_seed + index,
        )
        comparison["spec"] = run.get("spec")
        comparison["intervention_diagnostics"] = run.get(
            "intervention_diagnostics"
        )
        comparisons[arm] = comparison
        if family in {"selected_joint", "selected_single"}:
            candidate_names.append(arm)
        elif family == "random_joint":
            joint_random.append(comparison["accuracy_loss"])
        elif family == "random_single":
            state = int(run["spec"]["state"])
            single_random[state].append(comparison["accuracy_loss"])

    joint_candidates = [
        name for name in candidate_names
        if comparisons[name]["spec"]["family"] == "selected_joint"
    ]
    single_candidates = [
        name for name in candidate_names
        if comparisons[name]["spec"]["family"] == "selected_single"
    ]
    adjusted = {}
    for family in (joint_candidates, single_candidates):
        adjusted.update(
            _holm({name: comparisons[name]["mcnemar_one_sided_p"] for name in family})
        )
    critical = []
    for name in candidate_names:
        comparison = comparisons[name]
        spec = comparison["spec"]
        null = joint_random if spec["family"] == "selected_joint" else single_random[int(spec["state"])]
        if not null:
            raise ValueError(f"no matched random-null arms for {name}")
        loss = comparison["accuracy_loss"]
        empirical_p = (1 + sum(value >= loss for value in null)) / (len(null) + 1)
        comparison["holm_adjusted_mcnemar_p"] = adjusted[name]
        comparison["random_null_replicates"] = len(null)
        comparison["random_null_mean_accuracy_loss"] = float(np.mean(null))
        comparison["random_null_95_percentile_accuracy_loss"] = float(np.quantile(null, 0.95))
        comparison["empirical_random_null_p"] = float(empirical_p)
        comparison["accuracy_critical"] = bool(
            loss > 0
            and comparison["positive_in_both_halves"]
            and adjusted[name] <= familywise_alpha
            and empirical_p <= familywise_alpha
        )
        if comparison["accuracy_critical"]:
            critical.append(name)

    def random_family(values: list[float]) -> dict:
        return {
            "replicates": len(values),
            "mean_accuracy_loss": (
                None if not values else float(np.mean(values))
            ),
            "losses": values,
        }

    random_summary = {
        "joint": random_family(joint_random),
        "single_state_11": random_family(single_random[11]),
        "single_state_12": random_family(single_random[12]),
    }
    return {
        "analysis": "official_codi_frozen_forced_answer_colon_inference_ablation",
        "evaluated_examples": int(baseline.size),
        "baseline_accuracy": float(baseline.mean()),
        "forced_cue_baseline_accuracy": float(baseline.mean()),
        "native_reproduction_accuracy": native_accuracy,
        "forced_cue_minus_native_accuracy": (
            None if native_accuracy is None else float(baseline.mean()) - native_accuracy
        ),
        "answer_cue_endpoint_coverage": float(reached.mean()),
        "bootstrap_samples": bootstrap_samples,
        "familywise_alpha": familywise_alpha,
        "minimum_endpoint_coverage": minimum_endpoint_coverage,
        "accuracy_critical_criterion": (
            "positive loss in both deterministic halves, Holm-adjusted one-sided "
            "McNemar p<=alpha, and empirical matched-random p<=alpha"
        ),
        "accuracy_critical_directions_or_groups": critical,
        "comparisons": comparisons,
        "random_null": random_summary,
        "interpretation": (
            "Only listed critical arms have confirmatory evidence that their removal "
            "causally lowers frozen-checkpoint GSM8K accuracy conditional on the fixed "
            "answer-cue colon used by the residual collectors."
        ),
    }
