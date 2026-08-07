"""Single-hypothesis analysis for parameter-aware state-12 confirmation."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.eval.official_codi_endpoint_inference_ablation_analysis import _comparison


PRIMARY_ARM = "remove_parameter_aware_state12_primary"


def analyze_parameter_state12_confirmation(
    runs: Sequence[dict],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    alpha: float = 0.05,
    minimum_endpoint_coverage: float = 0.95,
    maximum_rms_ratio_deviation: float = 0.10,
) -> dict:
    if (
        bootstrap_samples <= 0
        or not 0 < alpha < 1
        or not 0 < minimum_endpoint_coverage <= 1
        or not 0 < maximum_rms_ratio_deviation < 1
    ):
        raise ValueError("confirmation thresholds are invalid")
    by_arm = {}
    for run in runs:
        arm = str(run["arm"])
        if arm in by_arm:
            raise ValueError(f"duplicate confirmation arm {arm}")
        by_arm[arm] = run
    if "baseline" not in by_arm or PRIMARY_ARM not in by_arm:
        raise ValueError("baseline and preregistered state-12 primary arm are required")
    baseline = np.asarray(by_arm["baseline"]["correctness"], dtype=bool)
    reached = np.asarray(by_arm["baseline"]["endpoint_reached"], dtype=bool)
    if baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("baseline correctness must be a nonempty vector")
    if float(reached.mean()) < minimum_endpoint_coverage:
        raise RuntimeError("answer-cue coverage is below the confirmation gate")

    comparisons = {}
    for index, (arm, run) in enumerate(sorted(by_arm.items())):
        if arm == "baseline":
            continue
        values = np.asarray(run["correctness"], dtype=bool)
        arm_reached = np.asarray(run["endpoint_reached"], dtype=bool)
        if values.shape != baseline.shape or not np.array_equal(arm_reached, reached):
            raise ValueError(f"{arm} is not exactly paired to baseline")
        family = (run.get("spec") or {}).get("family")
        comparison = _comparison(
            baseline,
            values,
            reached,
            bootstrap_samples=(
                bootstrap_samples
                if family == "selected_primary"
                else min(1_000, bootstrap_samples)
            ),
            seed=bootstrap_seed + index,
        )
        comparison["spec"] = run.get("spec")
        comparison["intervention_diagnostics"] = run.get("intervention_diagnostics")
        comparisons[arm] = comparison

    primary = comparisons[PRIMARY_ARM]
    random_names = [
        arm
        for arm, value in comparisons.items()
        if value["spec"].get("family") == "matched_random_primary"
    ]
    if not random_names:
        raise ValueError("matched-random state-12 controls are required")
    random_losses = [comparisons[arm]["accuracy_loss"] for arm in random_names]
    empirical_p = float(
        (1 + sum(value >= primary["accuracy_loss"] for value in random_losses))
        / (len(random_losses) + 1)
    )
    calibration_errors = []
    overlaps = []
    random_rms = []
    target_energies = []
    for arm in random_names:
        value = comparisons[arm]
        spec = value["spec"]
        target = float(spec["calibration_target_energy_by_state"]["12"])
        achieved = float(spec["calibration_achieved_energy_by_state"]["12"])
        target_energies.append(target)
        calibration_errors.append(abs(achieved - target) / max(target, 1e-12))
        overlaps.append(float(spec["selected_overlap_by_state"]["12"]))
        random_rms.append(
            float(
                value["intervention_diagnostics"]["removed_projection_rms_by_state"][
                    "12"
                ]
            )
        )
    selected_rms = float(
        primary["intervention_diagnostics"]["removed_projection_rms_by_state"]["12"]
    )
    if not np.isfinite(selected_rms) or selected_rms <= 0:
        raise RuntimeError("selected evaluation projection RMS is not positive and finite")
    if max(target_energies) - min(target_energies) > 1e-8 * max(target_energies):
        raise RuntimeError("matched-random arms do not share one calibration target")
    target_energy = target_energies[0]
    median_random_rms = float(np.median(random_rms))
    rms_ratio = median_random_rms / selected_rms
    rms_transport_passed = bool(
        abs(rms_ratio - 1.0) <= maximum_rms_ratio_deviation
    )
    statistical_passed = bool(
        primary["accuracy_loss"] > 0
        and primary["positive_in_both_halves"]
        and primary["bootstrap_95_ci"][0] > 0
        and primary["mcnemar_one_sided_p"] <= alpha
        and empirical_p <= alpha
    )
    calibration_passed = bool(
        max(calibration_errors) <= 2e-5 and max(overlaps) <= 0.20
    )
    confirmed = statistical_passed and calibration_passed and rms_transport_passed
    primary["empirical_matched_random_p"] = empirical_p
    primary["calibration_matching_passed"] = calibration_passed
    primary["evaluation_rms_transport_passed"] = rms_transport_passed
    primary["parameter_aware_state12_confirmed"] = confirmed
    status = (
        "confirmed"
        if confirmed
        else "evaluation_magnitude_transport_failed"
        if statistical_passed and calibration_passed and not rms_transport_passed
        else "not_confirmed"
    )
    return {
        "analysis": "official_codi_parameter_aware_state12_confirmation",
        "status": status,
        "evaluated_examples": int(baseline.size),
        "forced_cue_baseline_accuracy": float(baseline.mean()),
        "native_reproduction_accuracy": by_arm["baseline"].get(
            "native_reproduction_accuracy"
        ),
        "answer_cue_endpoint_coverage": float(reached.mean()),
        "primary_arm": PRIMARY_ARM,
        "primary_result": primary,
        "matched_random_null": {
            "replicates": len(random_losses),
            "accuracy_losses": random_losses,
            "mean_accuracy_loss": float(np.mean(random_losses)),
            "median_accuracy_loss": float(np.median(random_losses)),
            "accuracy_loss_95_percentile": float(np.quantile(random_losses, 0.95)),
            "selected_loss_percentile": float(
                (1 + sum(value < primary["accuracy_loss"] for value in random_losses))
                / (len(random_losses) + 1)
            ),
            "calibration_target_energy": target_energy,
            "maximum_calibration_relative_energy_error": max(calibration_errors),
            "maximum_normalized_selected_overlap": max(overlaps),
            "selected_evaluation_projection_rms": selected_rms,
            "random_evaluation_projection_rms": {
                "mean": float(np.mean(random_rms)),
                "median": median_random_rms,
                "minimum": float(np.min(random_rms)),
                "maximum": float(np.max(random_rms)),
            },
            "median_random_to_selected_rms_ratio": rms_ratio,
            "maximum_allowed_rms_ratio_deviation": maximum_rms_ratio_deviation,
            "evaluation_rms_transport_passed": rms_transport_passed,
        },
        "alpha": alpha,
        "multiplicity": "one preregistered primary hypothesis; no correction required",
        "decision_rule": (
            "Confirm only when paired loss is positive in both deterministic halves, "
            "the bootstrap lower bound is positive, one-sided exact McNemar p<=alpha, "
            "empirical matched-random p<=alpha, calibration matching passes, and the "
            "median random/selected evaluation RMS ratio is within the preregistered band."
        ),
        "parameter_aware_state12_confirmed": confirmed,
        "speed_claim": False,
        "comparisons": comparisons,
    }
