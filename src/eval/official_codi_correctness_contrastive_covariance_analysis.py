"""Preregistered gates for the contrastive correctness-covariance experiment."""
from __future__ import annotations

import numpy as np

from src.mech.endpoint_correctness_contrastive_covariance import (
    CONTRASTIVE_COVARIANCE_CONTRACT,
)


def paired_bootstrap_interval(
    left,
    right,
    *,
    samples: int,
    seed: int,
    alpha: float,
) -> tuple[float, float]:
    """Paired interval for ``mean(left - right)``."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired vectors must share a non-empty 1-D shape")
    delta = left - right
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=float)
    batch = 256
    for start in range(0, int(samples), batch):
        stop = min(start + batch, int(samples))
        indices = generator.integers(0, delta.size, size=(stop - start, delta.size))
        means[start:stop] = delta[indices].mean(1)
    return tuple(float(value) for value in np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def analyze_contrastive_covariance(
    summary: dict,
    artifact: dict,
    *,
    minimum_advantage_points: float = 1.0,
    minimum_wrong_removal_gain_points: float = 1.0,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260827,
    alpha: float = 0.05,
) -> dict:
    if summary.get("contract") != CONTRASTIVE_COVARIANCE_CONTRACT:
        raise ValueError("summary belongs to another contract")
    if artifact.get("contract") != CONTRASTIVE_COVARIANCE_CONTRACT:
        raise ValueError("artifact belongs to another contract")
    if summary["splits"]["partition_sha256"] != artifact.get("partition_sha256"):
        raise RuntimeError("summary and paired artifact use different partitions")
    outcomes = artifact["outcomes"]

    required = (
        "baseline",
        "contrastive_correct_retain",
        "correct_only_pca_retain",
        "accuracy_band_pca_retain",
        "contrastive_wrong_remove",
    )
    missing = [name for name in required if name not in outcomes]
    if missing:
        raise ValueError(f"missing required arms: {missing}")

    def correctness(name):
        return np.asarray(outcomes[name]["correct"], dtype=bool)

    primary = correctness("contrastive_correct_retain")
    baseline = correctness("baseline")

    def accuracy_contrast(left_name, right_name, seed_offset):
        left, right = correctness(left_name), correctness(right_name)
        if left.shape != baseline.shape or right.shape != baseline.shape:
            raise RuntimeError("all arms must be paired on the same questions")
        low, high = paired_bootstrap_interval(
            left.astype(float),
            right.astype(float),
            samples=bootstrap_samples,
            seed=bootstrap_seed + seed_offset,
            alpha=alpha,
        )
        return {
            "left": left_name,
            "right": right_name,
            "difference_points": float((left.mean() - right.mean()) * 100),
            "bootstrap_ci_points": [low * 100, high * 100],
            "passed": bool(
                (left.mean() - right.mean()) * 100 >= minimum_advantage_points
                and low > 0
            ),
        }

    comparisons = {
        "versus_correct_only_pca": accuracy_contrast(
            "contrastive_correct_retain", "correct_only_pca_retain", 1
        ),
        "versus_accuracy_band_pca": accuracy_contrast(
            "contrastive_correct_retain", "accuracy_band_pca_retain", 2
        ),
    }
    random_retain = sorted(
        name for name in outcomes if name.startswith("random_correct_energy_retain_")
    )
    if not random_retain:
        raise ValueError("matched-random retention arms are required")
    best_random_retain = max(random_retain, key=lambda name: correctness(name).mean())
    comparisons["versus_best_matched_random"] = accuracy_contrast(
        "contrastive_correct_retain", best_random_retain, 3
    )

    wrong = correctness("contrastive_wrong_remove")
    low, high = paired_bootstrap_interval(
        wrong.astype(float),
        baseline.astype(float),
        samples=bootstrap_samples,
        seed=bootstrap_seed + 4,
        alpha=alpha,
    )
    random_remove = sorted(
        name for name in outcomes if name.startswith("random_wrong_energy_remove_")
    )
    best_random_remove = (
        max(random_remove, key=lambda name: correctness(name).mean())
        if random_remove
        else None
    )
    wrong_removal = {
        "accuracy": float(wrong.mean()),
        "gain_over_baseline_points": float((wrong.mean() - baseline.mean()) * 100),
        "gain_bootstrap_ci_points": [low * 100, high * 100],
        "best_matched_random_arm": best_random_remove,
        "best_matched_random_accuracy": (
            None if best_random_remove is None else float(correctness(best_random_remove).mean())
        ),
    }
    wrong_removal["passed"] = bool(
        wrong_removal["gain_over_baseline_points"]
        >= minimum_wrong_removal_gain_points
        and low > 0
        and (
            best_random_remove is None
            or wrong.mean() > correctness(best_random_remove).mean()
        )
    )

    correct_gate = bool(all(item["passed"] for item in comparisons.values()))
    overall = bool(correct_gate and wrong_removal["passed"])
    return {
        "analysis": "official_codi_correctness_contrastive_covariance",
        "contract": CONTRASTIVE_COVARIANCE_CONTRACT,
        "status": "confirmed" if overall else "not_confirmed",
        "contrastive_covariance_confirmed": overall,
        "correct_specific_retention_confirmed": correct_gate,
        "wrong_specific_removal_confirmed": bool(wrong_removal["passed"]),
        "rank": int(summary["rank"]),
        "selected_shrinkage": float(summary["selected_shrinkage"]),
        "evaluated_examples": int(primary.size),
        "baseline_accuracy": float(baseline.mean()),
        "primary_accuracy": float(primary.mean()),
        "minimum_advantage_points": float(minimum_advantage_points),
        "minimum_wrong_removal_gain_points": float(minimum_wrong_removal_gain_points),
        "comparisons": comparisons,
        "wrong_specific_removal": wrong_removal,
        "descriptive_metrics": summary["arms"],
        "interpretation": (
            "The 28-D covariance ratio isolates correctness-specific answer-carrying "
            "geometry beyond class-conditional or class-blind PCA."
            if overall
            else "The 28-D covariance ratio did not clear all preregistered controls; "
            "correct/wrong covariance is descriptive, not a demonstrated correction channel."
        ),
    }
