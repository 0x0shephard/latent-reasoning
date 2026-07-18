"""Paired causal-effect summaries for latent-state interventions.

The unit of resampling is an evaluation question.  Every model/condition is aligned by
exact question and gold answer before effects are calculated, so the difference in
differences compares like with like rather than subtracting unrelated aggregate scores.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping

from src.eval.compare_runs import (
    EvalRun,
    _bootstrap_ci,
    _percentile,
    align_records,
    compare_runs,
    validate_alignment,
)


def analyze_position_sweep(
    baseline: EvalRun,
    positions: Mapping[int, EvalRun],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Compare each single-position intervention only against the shared baseline."""
    if not positions:
        raise ValueError("at least one position run is required")
    results = {}
    baseline_macro = None
    for position, run in sorted(positions.items()):
        if position < 0:
            raise ValueError("positions must be non-negative")
        label = f"p{position}"
        paired = compare_runs(
            {"baseline": baseline, label: run},
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        comparison = paired["comparisons"][0]
        baseline_macro = paired["runs"]["baseline"]["macro_mean"]
        results[str(position)] = {
            "path": str(run.path),
            "macro_accuracy": paired["runs"][label]["macro_mean"],
            "macro_effect": comparison["macro_accuracy_delta"],
            "macro_effect_95ci": comparison["macro_accuracy_delta_95ci"],
            "datasets": comparison["datasets"],
        }
    most_harmful = min(results, key=lambda key: results[key]["macro_effect"])
    return {
        "schema_version": 1,
        "analysis": "single_position_intervention_sweep",
        "effect_definition": "intervention_minus_baseline",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "baseline_path": str(baseline.path),
        "baseline_macro_accuracy": baseline_macro,
        "positions": results,
        "most_harmful_position": int(most_harmful),
    }


def render_position_sweep_markdown(report: Mapping) -> str:
    lines = [
        "# Latent-position intervention sweep",
        "",
        f"Effect is intervention minus baseline. Bootstrap samples: "
        f"{report['bootstrap_samples']}; seed: {report['seed']}.",
        "",
        f"Baseline macro accuracy: {report['baseline_macro_accuracy']:.4f}.",
        "",
        "| Position | Accuracy | Effect | 95% paired CI |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for position, result in report["positions"].items():
        low, high = result["macro_effect_95ci"]
        lines.append(
            f"| {position} | {result['macro_accuracy']:.4f} | "
            f"{result['macro_effect']:+.4f} | [{low:+.4f}, {high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            f"Most harmful position: **{report['most_harmful_position']}**.",
            "",
            "## Dataset effects",
            "",
            "| Position | Dataset | Effect | 95% CI | McNemar p |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for position, result in report["positions"].items():
        for dataset, values in result["datasets"].items():
            low, high = values["accuracy_delta_95ci"]
            lines.append(
                f"| {position} | {dataset} | {values['accuracy_delta']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | {values['mcnemar_exact_p']:.4g} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def _aligned_four(
    left_baseline: EvalRun,
    left_intervention: EvalRun,
    right_baseline: EvalRun,
    right_intervention: EvalRun,
    dataset: str,
):
    reference = left_baseline.datasets[dataset]
    _, left_changed = align_records(
        reference, left_intervention.datasets[dataset], dataset
    )
    _, right_base = align_records(reference, right_baseline.datasets[dataset], dataset)
    _, right_changed = align_records(
        reference, right_intervention.datasets[dataset], dataset
    )
    return tuple(reference), left_changed, right_base, right_changed


def _did_values(rows) -> tuple[list[int], list[int], list[int]]:
    left_base, left_changed, right_base, right_changed = rows
    left_effects = [
        int(changed.correct) - int(base.correct)
        for base, changed in zip(left_base, left_changed)
    ]
    right_effects = [
        int(changed.correct) - int(base.correct)
        for base, changed in zip(right_base, right_changed)
    ]
    did = [right - left for left, right in zip(left_effects, right_effects)]
    return left_effects, right_effects, did


def _macro_did_ci(
    aligned: Mapping[str, tuple], *, samples: int, rng: random.Random
) -> list[float]:
    estimates = []
    for _ in range(samples):
        dataset_estimates = []
        for rows in aligned.values():
            _, _, did = _did_values(rows)
            count = len(did)
            dataset_estimates.append(
                sum(did[rng.randrange(count)] for _ in range(count)) / count
            )
        estimates.append(sum(dataset_estimates) / len(dataset_estimates))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def compare_intervention_effects(
    *,
    left_name: str,
    left_baseline: EvalRun,
    left_intervention: EvalRun,
    right_name: str,
    right_baseline: EvalRun,
    right_intervention: EvalRun,
    intervention_name: str,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Estimate ``right intervention effect - left intervention effect``.

    A negative difference in differences means the intervention harms the right-hand
    method more.  This captures question-level sampling uncertainty, but not variation
    across training seeds.
    """
    if not left_name or not right_name or left_name == right_name:
        raise ValueError("left and right method names must be distinct and non-empty")
    if not intervention_name:
        raise ValueError("intervention_name cannot be empty")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    for run in (left_intervention, right_baseline, right_intervention):
        validate_alignment(left_baseline, run)

    rng = random.Random(seed)
    aligned = {}
    datasets = {}
    for dataset in sorted(left_baseline.datasets):
        rows = _aligned_four(
            left_baseline,
            left_intervention,
            right_baseline,
            right_intervention,
            dataset,
        )
        aligned[dataset] = rows
        left_effects, right_effects, did = _did_values(rows)
        datasets[dataset] = {
            "count": len(did),
            f"{left_name}_effect": sum(left_effects) / len(left_effects),
            f"{right_name}_effect": sum(right_effects) / len(right_effects),
            "difference_in_differences": sum(did) / len(did),
            "difference_in_differences_95ci": _bootstrap_ci(
                did, samples=bootstrap_samples, rng=rng
            ),
        }

    left_macro = sum(value[f"{left_name}_effect"] for value in datasets.values()) / len(
        datasets
    )
    right_macro = sum(
        value[f"{right_name}_effect"] for value in datasets.values()
    ) / len(datasets)
    return {
        "schema_version": 1,
        "analysis": "paired_difference_in_differences",
        "intervention": intervention_name,
        "left_method": left_name,
        "right_method": right_name,
        "effect_definition": "intervention_minus_baseline",
        "difference_in_differences_definition": f"{right_name}_effect_minus_{left_name}_effect",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "paths": {
            f"{left_name}_baseline": str(left_baseline.path),
            f"{left_name}_intervention": str(left_intervention.path),
            f"{right_name}_baseline": str(right_baseline.path),
            f"{right_name}_intervention": str(right_intervention.path),
        },
        "datasets": datasets,
        "macro": {
            f"{left_name}_effect": left_macro,
            f"{right_name}_effect": right_macro,
            "difference_in_differences": right_macro - left_macro,
            "difference_in_differences_95ci": _macro_did_ci(
                aligned, samples=bootstrap_samples, rng=rng
            ),
        },
        "uncertainty_scope": "evaluation questions only; training-seed variance excluded",
    }


def render_did_markdown(report: Mapping) -> str:
    left = report["left_method"]
    right = report["right_method"]
    macro = report["macro"]
    low, high = macro["difference_in_differences_95ci"]
    lines = [
        f"# {report['intervention']} difference in differences",
        "",
        "Effects are intervention minus baseline. A negative difference in differences "
        f"means {right} is harmed more than {left}.",
        "",
        f"Macro {left} effect: {macro[f'{left}_effect']:+.4f}.",
        f"Macro {right} effect: {macro[f'{right}_effect']:+.4f}.",
        f"Difference in differences ({right} minus {left}): "
        f"{macro['difference_in_differences']:+.4f} "
        f"(95% paired bootstrap CI {low:+.4f} to {high:+.4f}).",
        "",
        "| Dataset | " + left + " effect | " + right + " effect | DID | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, values in report["datasets"].items():
        low, high = values["difference_in_differences_95ci"]
        lines.append(
            f"| {dataset} | {values[f'{left}_effect']:+.4f} | "
            f"{values[f'{right}_effect']:+.4f} | "
            f"{values['difference_in_differences']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "Uncertainty is conditional on these checkpoints and covers evaluation-question "
            "sampling only; training-seed variance is excluded, so additional seeds are "
            "required for a method-level claim.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
