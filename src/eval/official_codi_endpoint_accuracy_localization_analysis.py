"""Confirmatory and hierarchical analysis for endpoint accuracy localization."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.eval.official_codi_endpoint_inference_ablation_analysis import (
    _comparison,
    _holm,
)
from src.mech.endpoint_accuracy_localization import LOCALIZATION_METHODS
from src.mech.endpoint_retention import RETENTION_COMMON_STATES


def _arm(method: str, suffix: str) -> str:
    return f"remove_{method}_{suffix}"


def _confirmatory_flag(comparison: dict, adjusted_mcnemar: float, alpha: float) -> bool:
    return bool(
        comparison["accuracy_loss"] > 0
        and comparison["positive_in_both_halves"]
        and comparison["bootstrap_95_ci"][0] > 0
        and adjusted_mcnemar <= alpha
    )


def analyze_endpoint_accuracy_localization(
    runs: Sequence[dict],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    familywise_alpha: float = 0.05,
    minimum_endpoint_coverage: float = 0.95,
) -> dict:
    """Analyze method-level matched nulls, then localize only passing parents."""
    if bootstrap_samples <= 0 or not 0 < familywise_alpha < 1:
        raise ValueError("bootstrap samples must be positive and alpha must be in (0,1)")
    by_arm = {}
    for run in runs:
        name = str(run["arm"])
        if name in by_arm:
            raise ValueError(f"duplicate localization arm {name}")
        by_arm[name] = run
    if "baseline" not in by_arm:
        raise ValueError("localization baseline is required")
    baseline = np.asarray(by_arm["baseline"]["correctness"], dtype=bool)
    reached = np.asarray(by_arm["baseline"]["endpoint_reached"], dtype=bool)
    if baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("baseline correctness must be a nonempty vector")
    if float(reached.mean()) < minimum_endpoint_coverage:
        raise RuntimeError("answer-cue coverage is below the localization gate")

    comparisons = {}
    correctness = {"baseline": baseline}
    for index, (name, run) in enumerate(sorted(by_arm.items())):
        if name == "baseline":
            continue
        values = np.asarray(run["correctness"], dtype=bool)
        arm_reached = np.asarray(run["endpoint_reached"], dtype=bool)
        if values.shape != baseline.shape or not np.array_equal(arm_reached, reached):
            raise ValueError(f"{name} is not paired to the baseline endpoint vector")
        correctness[name] = values
        family = (run.get("spec") or {}).get("family")
        comparisons[name] = _comparison(
            baseline,
            values,
            reached,
            bootstrap_samples=(
                bootstrap_samples
                if family != "matched_random_joint"
                else min(1_000, bootstrap_samples)
            ),
            seed=bootstrap_seed + index,
        )
        comparisons[name]["spec"] = run.get("spec")
        comparisons[name]["intervention_diagnostics"] = run.get(
            "intervention_diagnostics"
        )

    joint_names = [_arm(method, "joint") for method in LOCALIZATION_METHODS]
    joint_mcnemar_adjusted = _holm(
        {name: comparisons[name]["mcnemar_one_sided_p"] for name in joint_names}
    )
    random_nulls = {}
    empirical_raw = {}
    for method, name in zip(LOCALIZATION_METHODS, joint_names):
        null_names = [
            arm_name
            for arm_name, value in comparisons.items()
            if value["spec"].get("family") == "matched_random_joint"
            and value["spec"].get("matched_method") == method
        ]
        if not null_names:
            raise ValueError(f"no activation-energy-matched null for {method}")
        losses = [comparisons[arm_name]["accuracy_loss"] for arm_name in null_names]
        selected_loss = comparisons[name]["accuracy_loss"]
        empirical_raw[name] = float(
            (1 + sum(value >= selected_loss for value in losses)) / (len(losses) + 1)
        )
        actual_rms = {
            str(state): [
                float(
                    comparisons[arm_name]["intervention_diagnostics"][
                        "removed_projection_rms_by_state"
                    ][str(state)]
                )
                for arm_name in null_names
            ]
            for state in RETENTION_COMMON_STATES
        }
        calibration_relative_errors = []
        overlaps = []
        calibration_targets = {}
        for arm_name in null_names:
            spec = comparisons[arm_name]["spec"]
            for state in RETENTION_COMMON_STATES:
                target = float(spec["calibration_target_energy_by_state"][str(state)])
                achieved = float(spec["calibration_achieved_energy_by_state"][str(state)])
                calibration_targets[str(state)] = target
                calibration_relative_errors.append(
                    abs(achieved - target) / max(target, 1e-12)
                )
                overlaps.append(float(spec["selected_overlap_by_state"][str(state)]))
        random_nulls[method] = {
            "replicates": len(losses),
            "accuracy_losses": losses,
            "mean_accuracy_loss": float(np.mean(losses)),
            "accuracy_loss_95_percentile": float(np.quantile(losses, 0.95)),
            "maximum_calibration_relative_energy_error": max(calibration_relative_errors),
            "maximum_normalized_selected_overlap": max(overlaps),
            "calibration_target_energy_by_state": calibration_targets,
            "selected_evaluation_projection_rms_by_state": comparisons[name][
                "intervention_diagnostics"
            ]["removed_projection_rms_by_state"],
            "evaluation_projection_rms_by_state": {
                state: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
                for state, values in actual_rms.items()
            },
        }
    empirical_adjusted = _holm(empirical_raw)

    critical_joint = []
    for name in joint_names:
        comparison = comparisons[name]
        comparison["holm_adjusted_mcnemar_p"] = joint_mcnemar_adjusted[name]
        comparison["empirical_matched_random_p"] = empirical_raw[name]
        comparison["holm_adjusted_empirical_matched_random_p"] = empirical_adjusted[name]
        comparison["accuracy_critical_joint_subspace"] = bool(
            _confirmatory_flag(
                comparison, joint_mcnemar_adjusted[name], familywise_alpha
            )
            and empirical_adjusted[name] <= familywise_alpha
        )
        if comparison["accuracy_critical_joint_subspace"]:
            critical_joint.append(name)

    localization = {}
    for method in LOCALIZATION_METHODS:
        parent_name = _arm(method, "joint")
        parent_passed = parent_name in critical_joint
        state_names = [_arm(method, f"state{state}") for state in RETENTION_COMMON_STATES]
        state_adjusted = _holm(
            {name: comparisons[name]["mcnemar_one_sided_p"] for name in state_names}
        )
        states = {}
        for state, name in zip(RETENTION_COMMON_STATES, state_names):
            comparison = comparisons[name]
            comparison["holm_adjusted_mcnemar_p_within_method_states"] = state_adjusted[name]
            state_critical = bool(
                parent_passed
                and _confirmatory_flag(comparison, state_adjusted[name], familywise_alpha)
            )
            states[str(state)] = {
                "arm": name,
                "accuracy_loss": comparison["accuracy_loss"],
                "accuracy_loss_percentage_points": comparison[
                    "accuracy_loss_percentage_points"
                ],
                "holm_adjusted_mcnemar_p": state_adjusted[name],
                "accuracy_critical_state": state_critical,
            }

        single_names = []
        reduced_names = []
        labels = []
        for state in RETENTION_COMMON_STATES:
            for slot in range(3):
                single_names.append(_arm(method, f"s{state}_d{slot}"))
                reduced_names.append(
                    _arm(method, f"joint_except_s{state}_d{slot}")
                )
                labels.append((state, slot))
        single_adjusted = _holm(
            {name: comparisons[name]["mcnemar_one_sided_p"] for name in single_names}
        )
        rescue_comparisons = {}
        for offset, (name, reduced_name) in enumerate(zip(single_names, reduced_names)):
            # Keeping this direction while removing the other five is beneficial
            # when the reduced arm is more accurate than the full joint ablation.
            rescue_comparisons[name] = _comparison(
                correctness[reduced_name],
                correctness[parent_name],
                reached,
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed + 100_000 + offset,
            )
        rescue_adjusted = _holm(
            {
                name: value["mcnemar_one_sided_p"]
                for name, value in rescue_comparisons.items()
            }
        )
        directions = {}
        for label, single_name, reduced_name in zip(labels, single_names, reduced_names):
            state, slot = label
            single = comparisons[single_name]
            rescue = rescue_comparisons[single_name]
            individually_necessary = bool(
                parent_passed
                and _confirmatory_flag(
                    single, single_adjusted[single_name], familywise_alpha
                )
            )
            rescues_joint = bool(
                parent_passed
                and _confirmatory_flag(
                    rescue, rescue_adjusted[single_name], familywise_alpha
                )
            )
            pc_index = single["spec"].get("residual_pc_index")
            key = f"state{state}_slot{slot}_pc{pc_index}"
            directions[key] = {
                "state": state,
                "direction_slot": slot,
                "residual_pc_index": pc_index,
                "single_removal_arm": single_name,
                "joint_minus_one_arm": reduced_name,
                "single_accuracy_loss": single["accuracy_loss"],
                "single_accuracy_loss_percentage_points": single[
                    "accuracy_loss_percentage_points"
                ],
                "single_holm_adjusted_mcnemar_p": single_adjusted[single_name],
                "rescue_accuracy_gain_over_joint": rescue["accuracy_loss"],
                "rescue_accuracy_gain_percentage_points": rescue[
                    "accuracy_loss_percentage_points"
                ],
                "rescue_holm_adjusted_mcnemar_p": rescue_adjusted[single_name],
                "individually_necessary": individually_necessary,
                "rescues_joint_ablation": rescues_joint,
                "accuracy_core_direction": individually_necessary and rescues_joint,
            }
        joint_loss = comparisons[parent_name]["accuracy_loss"]
        localization[method] = {
            "parent_joint_passed": parent_passed,
            "states": states,
            "directions": directions,
            "interaction_diagnostics": {
                "joint_accuracy_loss": joint_loss,
                "sum_single_accuracy_losses": float(
                    sum(comparisons[name]["accuracy_loss"] for name in single_names)
                ),
                "joint_minus_sum_single_losses": float(
                    joint_loss
                    - sum(comparisons[name]["accuracy_loss"] for name in single_names)
                ),
                "sum_state_accuracy_losses": float(
                    sum(comparisons[name]["accuracy_loss"] for name in state_names)
                ),
                "joint_minus_sum_state_losses": float(
                    joint_loss
                    - sum(comparisons[name]["accuracy_loss"] for name in state_names)
                ),
            },
        }

    negative_name = "remove_energy_joint_negative_control"
    return {
        "analysis": "official_codi_answer_colon_accuracy_localization",
        "evaluated_examples": int(baseline.size),
        "forced_cue_baseline_accuracy": float(baseline.mean()),
        "native_reproduction_accuracy": by_arm["baseline"].get(
            "native_reproduction_accuracy"
        ),
        "answer_cue_endpoint_coverage": float(reached.mean()),
        "bootstrap_samples": bootstrap_samples,
        "familywise_alpha": familywise_alpha,
        "confirmatory_joint_methods": list(LOCALIZATION_METHODS),
        "critical_joint_subspaces": critical_joint,
        "negative_control": {
            "arm": negative_name,
            "accuracy_loss": comparisons[negative_name]["accuracy_loss"],
            "accuracy_loss_percentage_points": comparisons[negative_name][
                "accuracy_loss_percentage_points"
            ],
            "bootstrap_95_ci": comparisons[negative_name]["bootstrap_95_ci"],
        },
        "matched_random_null": random_nulls,
        "localization": localization,
        "comparisons": comparisons,
        "decision_rule": (
            "Joint parent: positive loss in both deterministic halves, positive "
            "bootstrap lower bound, Holm-adjusted one-sided McNemar p<=alpha, and "
            "Holm-adjusted empirical activation-energy-matched-random p<=alpha. "
            "State/direction claims are hierarchical and require the joint parent."
        ),
        "speed_claim": False,
    }
