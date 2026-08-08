"""Exact-match confirmation of the accuracy-bearing colon-state PC band.

The analytic tier established, on held-out first-token accuracy, that principal
components 4-31 of the student colon state carry 85.9% of the model's accuracy
while the leading components 0-3 carry 6.7% despite holding 82.3% of the variance.
Those are first-token numbers. This module applies the preregistered gates to real
greedy decoding scored by numeric exact match.

Three gates, frozen before any exact-match outcome was read:

``sufficiency``
    Retaining only PCs 4-31 preserves at least 70% of baseline accuracy.
``dissociation``
    Retaining only PCs 0-3 preserves at most 20% of baseline, and the primary band
    beats it with a positive paired bootstrap lower bound.
``necessity``
    Removing PCs 4-31 costs at least 20 accuracy points, with a positive lower bound.

All three must pass. The random-subspace arms are descriptive: the specificity null
was established analytically with 200 replicates, and generation is too expensive to
rebuild it here.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def gsm8k_accuracy_from_summary(payload: dict) -> float:
    """Read GSM8K accuracy from an official-CODI evaluation summary.

    ``evaluate_official_codi`` writes ``datasets[name]`` as a bare float. Other
    summaries in the project carry a nested dict or a flat ``gsm8k_accuracy``, so
    all three shapes are accepted. This lives in one place because writing the
    lookup twice is what produced the float/dict crash it now handles.
    """
    value = (payload.get("datasets") or {}).get("gsm8k")
    if isinstance(value, dict):
        value = value.get("accuracy")
    if value is None:
        value = payload.get("gsm8k_accuracy", payload.get("accuracy"))
    if value is None:
        raise ValueError("summary has no GSM8K accuracy")
    return float(value)


def _paired_accuracy_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap ``mean(left) - mean(right)`` over questions, keeping pairs intact."""
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired accuracy vectors must share a nonempty 1-D shape")
    generator = np.random.default_rng(seed)
    deltas = left.astype(np.float64) - right.astype(np.float64)
    indices = generator.integers(0, deltas.size, size=(samples, deltas.size))
    means = deltas[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _exact_mcnemar_one_sided(baseline: np.ndarray, arm: np.ndarray) -> float:
    """One-sided exact McNemar p that the arm is worse than the baseline."""
    from math import comb

    lost = int(np.sum(baseline & ~arm))
    gained = int(np.sum(~baseline & arm))
    total = lost + gained
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(lost, total + 1))
    return float(tail / (2**total))


def _band_key(band: Sequence[int]) -> tuple[int, int]:
    return int(band[0]), int(band[1])


def analyze_band_confirmation(
    runs: Sequence[dict],
    *,
    primary_band: Sequence[int],
    control_band: Sequence[int],
    majority_band: Sequence[int],
    minimum_primary_retention: float,
    maximum_control_retention: float,
    minimum_primary_removal_points: float,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    alpha: float = 0.05,
    minimum_endpoint_coverage: float = 0.95,
    reproduction_accuracy: float | None = None,
    maximum_baseline_accuracy_drift: float | None = None,
) -> dict:
    by_arm: dict[tuple, dict] = {}
    baseline_run = None
    for run in runs:
        if run.get("arm") == "baseline":
            if baseline_run is not None:
                raise ValueError("duplicate baseline arm")
            baseline_run = run
            continue
        key = (_band_key(run["band"]), str(run["mode"])) if run.get("band") else (
            str(run["arm"]),
            str(run["mode"]),
        )
        if key in by_arm:
            raise ValueError(f"duplicate confirmation arm {key}")
        by_arm[key] = run
    if baseline_run is None:
        raise ValueError("the baseline arm is required")

    baseline = np.asarray(baseline_run["correctness"], dtype=bool)
    reached = np.asarray(baseline_run["endpoint_reached"], dtype=bool)
    if float(reached.mean()) < minimum_endpoint_coverage:
        raise RuntimeError("answer-cue coverage is below the confirmation gate")
    baseline_accuracy = float(baseline.mean())
    if baseline_accuracy <= 0:
        raise RuntimeError("baseline accuracy must be positive")

    drift = None
    drift_passed = True
    if reproduction_accuracy is not None and maximum_baseline_accuracy_drift is not None:
        drift = abs(baseline_accuracy - float(reproduction_accuracy))
        drift_passed = bool(drift <= float(maximum_baseline_accuracy_drift))

    def fetch(band: Sequence[int], mode: str) -> dict | None:
        run = by_arm.get((_band_key(band), mode))
        if run is None:
            return None
        values = np.asarray(run["correctness"], dtype=bool)
        if values.shape != baseline.shape:
            raise ValueError(f"band {band} {mode} is not paired to the baseline")
        accuracy = float(values.mean())
        lower, upper = _paired_accuracy_bootstrap(
            baseline.astype(np.float64),
            values.astype(np.float64),
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        return {
            "arm": run.get("arm"),
            "band": list(_band_key(band)),
            "mode": mode,
            "accuracy": accuracy,
            "retained_fraction": accuracy / baseline_accuracy,
            "accuracy_loss_points": (baseline_accuracy - accuracy) * 100.0,
            "loss_bootstrap_95_ci_points": [lower * 100.0, upper * 100.0],
            "mcnemar_one_sided_p": _exact_mcnemar_one_sided(baseline, values),
            "variance_share": run.get("variance_share"),
            "correct": int(values.sum()),
        }

    primary_retain = fetch(primary_band, "retain")
    control_retain = fetch(control_band, "retain")
    primary_remove = fetch(primary_band, "remove")
    if primary_retain is None or control_retain is None or primary_remove is None:
        raise RuntimeError(
            "the primary retention, control retention, and primary removal arms are required"
        )

    # Gate 2 also compares the two retention arms to each other, paired by question.
    primary_values = np.asarray(
        by_arm[(_band_key(primary_band), "retain")]["correctness"], dtype=bool
    )
    control_values = np.asarray(
        by_arm[(_band_key(control_band), "retain")]["correctness"], dtype=bool
    )
    contrast_lower, contrast_upper = _paired_accuracy_bootstrap(
        primary_values.astype(np.float64),
        control_values.astype(np.float64),
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )

    random_retention = []
    for (key, mode), run in sorted(by_arm.items(), key=lambda item: str(item[0])):
        if mode != "retain" or not str(run.get("arm", "")).startswith("random_matched"):
            continue
        values = np.asarray(run["correctness"], dtype=bool)
        random_retention.append(
            {
                "arm": run["arm"],
                "accuracy": float(values.mean()),
                "retained_fraction": float(values.mean()) / baseline_accuracy,
            }
        )

    sufficiency = bool(
        primary_retain["retained_fraction"] >= minimum_primary_retention
    )
    dissociation = bool(
        control_retain["retained_fraction"] <= maximum_control_retention
        and contrast_lower > 0
    )
    necessity = bool(
        primary_remove["accuracy_loss_points"] >= minimum_primary_removal_points * 100.0
        and primary_remove["loss_bootstrap_95_ci_points"][0] > 0
        and primary_remove["mcnemar_one_sided_p"] <= alpha
    )
    confirmed = bool(sufficiency and dissociation and necessity and drift_passed)
    status = (
        "band_confirmed"
        if confirmed
        else "baseline_drift_failed"
        if not drift_passed
        else "band_not_confirmed"
    )
    return {
        "analysis": "official_codi_endpoint_band_confirmation",
        "status": status,
        "band_confirmed": confirmed,
        "alpha": alpha,
        "evaluated_examples": int(baseline.size),
        "baseline_accuracy": baseline_accuracy,
        "baseline_correct": int(baseline.sum()),
        "answer_cue_endpoint_coverage": float(reached.mean()),
        "baseline_accuracy_drift": drift,
        "baseline_drift_passed": drift_passed,
        "primary_band": list(_band_key(primary_band)),
        "control_band": list(_band_key(control_band)),
        "majority_band": list(_band_key(majority_band)),
        "gates": {
            "sufficiency": {
                "threshold": minimum_primary_retention,
                "observed": primary_retain["retained_fraction"],
                "passed": sufficiency,
            },
            "dissociation": {
                "threshold": maximum_control_retention,
                "control_retained_fraction": control_retain["retained_fraction"],
                "primary_minus_control_points": (
                    primary_retain["accuracy"] - control_retain["accuracy"]
                )
                * 100.0,
                "contrast_bootstrap_95_ci_points": [
                    contrast_lower * 100.0,
                    contrast_upper * 100.0,
                ],
                "passed": dissociation,
            },
            "necessity": {
                "threshold_points": minimum_primary_removal_points * 100.0,
                "observed_points": primary_remove["accuracy_loss_points"],
                "bootstrap_95_ci_points": primary_remove["loss_bootstrap_95_ci_points"],
                "mcnemar_one_sided_p": primary_remove["mcnemar_one_sided_p"],
                "passed": necessity,
            },
        },
        "arms": {
            "primary_retain": primary_retain,
            "control_retain": control_retain,
            "primary_remove": primary_remove,
            "majority_retain": fetch(majority_band, "retain"),
            "control_remove": fetch(control_band, "remove"),
            "prefix_retain": fetch((0, 32), "retain"),
            "complement_retain": fetch((32, 768), "retain"),
        },
        "random_retention_controls": random_retention,
        "decision_rule": (
            "Confirmed only when retaining the primary band preserves at least the "
            "preregistered fraction of baseline exact match, retaining the leading "
            "control band preserves at most its ceiling with a positive paired "
            "advantage for the primary band, removing the primary band costs at "
            "least the preregistered points with a positive bootstrap lower bound "
            "and exact McNemar p<=alpha, and the forced-cue baseline has not drifted."
        ),
        "specificity_note": (
            "Random-subspace retention arms are descriptive. The specificity null "
            "was established analytically with 200 energy-matched replicates; "
            "generation is too expensive to rebuild it at this arm count."
        ),
        "speed_claim": False,
    }
