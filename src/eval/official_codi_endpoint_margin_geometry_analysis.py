"""Preregistered gates for the answer-colon margin-geometry experiment.

Two hypotheses are primary and everything else is exploratory:

``margin_specificity``
    At the primary rank, does the closed-form margin subspace damage held-out
    gold-answer NLL more than energy-matched random subspaces of the same rank?
    This is the direct rerun of the failed rank-three confirmation with a
    continuous outcome and a selector matched to the test statistic, so a
    negative result here cannot be blamed on either.

``effective_rank``
    What is the smallest rank whose retained subspace preserves the preregistered
    fraction of baseline first-token accuracy?  This is the sufficiency question
    that removal arms structurally cannot answer.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


PRIMARY_FAMILY = "margin"
REFERENCE_FAMILIES = ("answer_conditioned", "parameter_aware")


def _paired_bootstrap(
    deltas: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError("bootstrap needs a nonempty per-example delta vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, deltas.size, size=(samples, deltas.size))
    means = deltas[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _standard_error(deltas: np.ndarray) -> float:
    if deltas.size < 2:
        return float("inf")
    return float(deltas.std(ddof=1) / np.sqrt(deltas.size))


def continuous_comparison(
    baseline: np.ndarray,
    arm: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Paired damage on a continuous outcome; positive means the arm hurt."""
    if baseline.shape != arm.shape:
        raise ValueError("arm and baseline outcomes must be exactly paired")
    deltas = np.asarray(arm, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    half = deltas.size // 2
    lower, upper = _paired_bootstrap(deltas, samples=bootstrap_samples, seed=seed)
    error = _standard_error(deltas)
    return {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "standard_error": error,
        "z_score": float(deltas.mean() / error) if np.isfinite(error) and error > 0 else 0.0,
        "bootstrap_95_ci": [lower, upper],
        "positive_in_both_halves": bool(
            deltas[:half].mean() > 0 and deltas[half:].mean() > 0
        ),
        "deterministic_half_deltas": [
            float(deltas[:half].mean()),
            float(deltas[half:].mean()),
        ],
    }


def binary_comparison(baseline: np.ndarray, arm: np.ndarray) -> dict:
    """First-token accuracy loss, kept only to quantify the power difference."""
    baseline = np.asarray(baseline, dtype=bool)
    arm = np.asarray(arm, dtype=bool)
    if baseline.shape != arm.shape:
        raise ValueError("arm and baseline correctness must be exactly paired")
    deltas = baseline.astype(np.float64) - arm.astype(np.float64)
    error = _standard_error(deltas)
    return {
        "baseline_accuracy": float(baseline.mean()),
        "arm_accuracy": float(arm.mean()),
        "accuracy_loss": float(deltas.mean()),
        "accuracy_loss_percentage_points": float(deltas.mean() * 100.0),
        "standard_error": error,
        "z_score": float(deltas.mean() / error) if np.isfinite(error) and error > 0 else 0.0,
        "correct_to_wrong": int(np.sum(baseline & ~arm)),
        "wrong_to_correct": int(np.sum(~baseline & arm)),
    }


def _arm_key(name: str, mode: str, semantics: str) -> str:
    return f"{name}|{mode}|{semantics}"


def _subspace_name(family: str, rank: int, state: int) -> str:
    return f"{family}_k{rank:03d}_s{state}"


def _matched_random_names(
    subspace_names: Sequence[str] | set[str], family: str, rank: int, state: int
) -> list[str]:
    prefix = f"random_matched_{family}_k{rank:03d}_s{state}_r"
    return sorted(name for name in subspace_names if name.startswith(prefix))


def _empirical_p(selected: float, controls: Sequence[float]) -> float:
    return float(
        (1 + sum(float(value) >= selected for value in controls)) / (len(controls) + 1)
    )


def analyze_margin_geometry(
    sweep: Mapping,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    alpha: float = 0.05,
    primary_rank: int = 3,
    state: int = 12,
    retention_threshold: float = 0.90,
    maximum_calibration_relative_energy_error: float = 5e-3,
    maximum_selected_overlap: float = 0.20,
) -> dict:
    if not 0 < alpha < 1 or not 0 < retention_threshold <= 1:
        raise ValueError("analysis thresholds are invalid")
    arms = sweep["arms"]
    baseline = sweep["baseline"]
    baseline_nll = np.asarray(baseline["nll"], dtype=np.float64)
    baseline_top1 = np.asarray(baseline["top1_correct"], dtype=bool)
    baseline_accuracy = float(baseline_top1.mean())
    if baseline_accuracy <= 0:
        raise RuntimeError("baseline first-token accuracy must be positive")

    def evaluate(name: str, mode: str, semantics: str, seed_offset: int) -> dict | None:
        key = _arm_key(name, mode, semantics)
        arm = arms.get(key)
        if arm is None:
            return None
        record = continuous_comparison(
            baseline_nll,
            np.asarray(arm["nll"], dtype=np.float64),
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + seed_offset,
        )
        record["binary"] = binary_comparison(
            baseline_top1, np.asarray(arm["top1_correct"], dtype=bool)
        )
        record["margin_delta"] = float(
            np.asarray(baseline["margin"], dtype=np.float64).mean()
            - np.asarray(arm["margin"], dtype=np.float64).mean()
        )
        record["removed_projection_rms"] = float(arm["removed_projection_rms"])
        record["subspace"] = arm["subspace"]
        record["arm"] = key
        return record

    # ---- Primary 1: margin specificity at the primary rank -----------------
    selected_name = _subspace_name(PRIMARY_FAMILY, primary_rank, state)
    selected = evaluate(selected_name, "remove", "mean", 1)
    if selected is None:
        raise RuntimeError("the primary margin arm is missing from the sweep")
    subspace_names = {key.split("|")[0] for key in arms}
    control_names = _matched_random_names(
        subspace_names, PRIMARY_FAMILY, primary_rank, state
    )
    if not control_names:
        raise RuntimeError("energy-matched random controls are required")
    controls = []
    for index, name in enumerate(control_names):
        control = evaluate(name, "remove", "mean", 1000 + index)
        if control is not None:
            controls.append(control)
    control_deltas = [value["mean_delta"] for value in controls]
    empirical_p = _empirical_p(selected["mean_delta"], control_deltas)
    energy_errors = []
    overlaps = []
    attainable = []
    for control in controls:
        spec = control["subspace"]
        target = float(spec.get("calibration_target_energy", 0.0))
        achieved = float(spec.get("calibration_achieved_energy", 0.0))
        if target > 0:
            energy_errors.append(abs(achieved - target) / target)
        overlaps.append(float(spec.get("selected_overlap", 0.0)))
        attainable.append(spec.get("target_attainable") is not False)
    # A control that could not reach the selection's own calibration energy is
    # inadmissible: that is precisely the transport asymmetry that made the
    # completed state-12 null conservative.
    matching_passed = bool(
        energy_errors
        and all(attainable)
        and max(energy_errors) <= maximum_calibration_relative_energy_error
        and max(overlaps) <= maximum_selected_overlap
    )
    specificity_supported = bool(
        selected["mean_delta"] > 0
        and selected["positive_in_both_halves"]
        and selected["bootstrap_95_ci"][0] > 0
        and empirical_p <= alpha
        and matching_passed
    )

    # ---- Primary 2: effective dimensionality from retention ----------------
    def retention_curve(family: str) -> list[dict]:
        curve = []
        for key, arm in arms.items():
            name, mode, semantics = key.split("|")
            if mode != "retain" or semantics != "mean":
                continue
            spec = arm["subspace"]
            if spec["family"] != family or spec["state"] != state:
                continue
            accuracy = float(np.asarray(arm["top1_correct"], dtype=bool).mean())
            curve.append(
                {
                    "rank": int(spec["rank"]),
                    "arm": key,
                    "first_token_accuracy": accuracy,
                    "retained_fraction": accuracy / baseline_accuracy,
                    "mean_gold_nll": float(np.asarray(arm["nll"], dtype=np.float64).mean()),
                    "random_replicate": spec.get("random_replicate"),
                }
            )
        return sorted(curve, key=lambda row: (row["rank"], row["arm"]))

    def effective_rank(curve: Sequence[Mapping]) -> int | None:
        by_rank: dict[int, list[float]] = {}
        for row in curve:
            by_rank.setdefault(int(row["rank"]), []).append(
                float(row["retained_fraction"])
            )
        for rank in sorted(by_rank):
            if float(np.median(by_rank[rank])) >= retention_threshold:
                return rank
        return None

    retention = {
        family: retention_curve(family)
        for family in ("margin", "answer_nll", "energy", "readout", "random_matched")
    }
    effective_ranks = {
        family: effective_rank(curve) for family, curve in retention.items()
    }

    # ---- Reference selectors: was the completed confirmation underpowered? --
    references = {}
    for index, family in enumerate(REFERENCE_FAMILIES):
        name = _subspace_name(family, primary_rank, state)
        record = evaluate(name, "remove", "mean", 5000 + index)
        if record is None:
            continue
        reference_controls = [
            value
            for value in (
                evaluate(control, "remove", "mean", 6000 + index * 500 + offset)
                for offset, control in enumerate(
                    _matched_random_names(subspace_names, family, primary_rank, state)
                )
            )
            if value is not None
        ]
        record["empirical_matched_random_p"] = _empirical_p(
            record["mean_delta"], [value["mean_delta"] for value in reference_controls]
        )
        record["matched_random_replicates"] = len(reference_controls)
        record["binary_empirical_matched_random_p"] = _empirical_p(
            record["binary"]["accuracy_loss"],
            [value["binary"]["accuracy_loss"] for value in reference_controls],
        )
        references[family] = record

    # ---- Ablation semantics -------------------------------------------------
    semantics_report = {}
    for semantics_index, semantics in enumerate(("mean", "zero", "resample")):
        record = evaluate(selected_name, "remove", semantics, 8000 + semantics_index)
        if record is not None:
            semantics_report[semantics] = record

    status = (
        "margin_specificity_supported"
        if specificity_supported
        else "margin_specificity_not_supported"
    )
    return {
        "analysis": "official_codi_endpoint_margin_geometry",
        "status": status,
        "alpha": alpha,
        "primary_rank": primary_rank,
        "state": state,
        "evaluated_examples": int(baseline_top1.size),
        "baseline_first_token_accuracy": baseline_accuracy,
        "baseline_mean_gold_nll": float(baseline_nll.mean()),
        "decision_rule": (
            "Margin specificity is supported only when the closed-form margin "
            "subspace raises held-out gold-answer NLL, the effect is positive in "
            "both deterministic halves, the paired bootstrap lower bound is "
            "positive, the empirical energy-matched random p is at most alpha, and "
            "calibration energy matching passes. Effective rank is reported "
            "separately and never gates specificity."
        ),
        "primary_margin_specificity": {
            "arm": selected["arm"],
            "result": selected,
            "empirical_matched_random_p": empirical_p,
            "matched_random_replicates": len(controls),
            "matched_random_mean_delta": float(np.mean(control_deltas)),
            "matched_random_median_delta": float(np.median(control_deltas)),
            "matched_random_95_percentile": float(np.quantile(control_deltas, 0.95)),
            "selected_delta_percentile": float(
                (1 + sum(value < selected["mean_delta"] for value in control_deltas))
                / (len(control_deltas) + 1)
            ),
            "maximum_calibration_relative_energy_error": (
                max(energy_errors) if energy_errors else None
            ),
            "maximum_selected_overlap": max(overlaps) if overlaps else None,
            "calibration_matching_passed": matching_passed,
            "supported": specificity_supported,
        },
        "effective_dimensionality": {
            "retention_threshold": retention_threshold,
            "effective_rank_by_family": effective_ranks,
            "retention_curves": retention,
        },
        "reference_selectors": references,
        "ablation_semantics": semantics_report,
        "power_note": (
            "z_score compares the paired mean to its own standard error on the "
            "same 1,319 questions; the continuous and binary z-scores of the same "
            "arm are what quantify how much power the exact-match outcome discarded."
        ),
        "speed_claim": False,
    }
