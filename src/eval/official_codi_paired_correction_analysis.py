"""Frozen gates for same-question, conditioned answer-cue correction."""
from __future__ import annotations

import numpy as np

from src.mech.endpoint_paired_correction import PAIRED_CORRECTION_CONTRACT


def paired_interval(left, right, *, samples: int, seed: int, alpha: float):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired outcome vectors must share a non-empty 1-D shape")
    delta = left - right
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=float)
    batch = 256
    for start in range(0, int(samples), batch):
        stop = min(int(samples), start + batch)
        index = generator.integers(0, delta.size, size=(stop - start, delta.size))
        means[start:stop] = delta[index].mean(1)
    return [
        float(value)
        for value in np.quantile(means, [float(alpha) / 2, 1 - float(alpha) / 2])
    ]


def analyze_paired_correction(
    summary: dict,
    artifact: dict,
    *,
    minimum_gain_points: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    alpha: float,
) -> dict:
    if summary.get("contract") != PAIRED_CORRECTION_CONTRACT:
        raise ValueError("summary belongs to another contract")
    if artifact.get("contract") != PAIRED_CORRECTION_CONTRACT:
        raise ValueError("artifact belongs to another contract")
    if summary["partition_sha256"] != artifact.get("partition_sha256"):
        raise RuntimeError("summary and artifact partitions differ")
    outcomes = artifact["outcomes"]
    required = ("baseline", "conditioned", "global_mean", "shuffled_target")
    if any(name not in outcomes for name in required):
        raise ValueError("baseline, conditioned, global and shuffled arms are required")

    def values(name, field="correct"):
        return np.asarray(outcomes[name][field])

    def contrast(left, right, offset):
        left_values = values(left).astype(float)
        right_values = values(right).astype(float)
        interval = paired_interval(
            left_values,
            right_values,
            samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
            alpha=alpha,
        )
        return {
            "left": left,
            "right": right,
            "difference_points": float((left_values.mean() - right_values.mean()) * 100),
            "bootstrap_ci_points": [100 * interval[0], 100 * interval[1]],
        }

    comparisons = {
        "versus_baseline": contrast("conditioned", "baseline", 1),
        "versus_global_mean": contrast("conditioned", "global_mean", 2),
        "versus_shuffled_target": contrast("conditioned", "shuffled_target", 3),
    }
    gain = comparisons["versus_baseline"]
    primary_pass = bool(
        gain["difference_points"] >= float(minimum_gain_points)
        and gain["bootstrap_ci_points"][0] > 0
    )
    specificity_pass = bool(
        comparisons["versus_global_mean"]["bootstrap_ci_points"][0] > 0
        and comparisons["versus_shuffled_target"]["bootstrap_ci_points"][0] > 0
    )
    selected = summary["selected_interventions"]["conditioned"]
    nontrivial = bool(
        float(selected["alpha"]) > 0
        and float(selected["edited_fraction"]) > 0
        and float(summary["test_arms"]["conditioned"]["edited_fraction"]) > 0
    )
    confirmed = bool(primary_pass and specificity_pass and nontrivial)
    return {
        "analysis": "official_codi_same_question_paired_correction",
        "contract": PAIRED_CORRECTION_CONTRACT,
        "status": "confirmed" if confirmed else "not_confirmed",
        "paired_correction_confirmed": confirmed,
        "primary_gain_passed": primary_pass,
        "conditioned_specificity_passed": specificity_pass,
        "nontrivial_intervention_selected": nontrivial,
        "minimum_gain_points": float(minimum_gain_points),
        "evaluated_examples": int(values("baseline").size),
        "paired_questions": {
            "fit": int(summary["splits"]["fit_paired_questions"]),
            "select": int(summary["splits"]["select_paired_questions"]),
        },
        "selected_ridge": float(summary["selected_ridge"]),
        "selected_interventions": summary["selected_interventions"],
        "test_arms": summary["test_arms"],
        "comparisons": comparisons,
        "interpretation": (
            "A question-conditioned correction learned from same-question "
            "counterfactuals improves final-test answers beyond unconditional and "
            "shuffled controls."
            if confirmed
            else "The paired counterfactual map did not clear all frozen accuracy "
            "and specificity gates; it is not a demonstrated correction mechanism."
        ),
    }
