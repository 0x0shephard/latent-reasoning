"""Frozen gates for the latent-trajectory detection experiment.

Two preregistered questions, one final read each:

* **correctness**: does the select-chosen trajectory cell, combined with the
  endpoint margin, predict first-token correctness decisively better than the
  margin alone?  "Decisively" is a 0.02 AUC threshold — §49 measured that a
  ~0.013 increment cannot be bounded away from zero on 439 test questions, so a
  gate this experiment can actually resolve must ask for more.
* **answer identity**: on the final-test questions the model gets *wrong*, does a
  probe on the chosen trajectory cell recover the gold first answer token better
  than the same probe class reading the endpoint state, and better than the
  majority class?  This is the gate that would justify an editing experiment:
  it asks whether the trajectory still knows an answer the endpoint discarded.

Only a passed answer-identity gate licenses proposing a latent-state editing
experiment; a correctness-only pass is a detection finding and nothing more.
"""
from __future__ import annotations

import numpy as np

from src.eval.official_codi_correctness_tracks_analysis import _auc_bootstrap
from src.eval.official_codi_paired_correction_analysis import paired_interval
from src.mech.latent_trajectory_detect import LATENT_TRAJECTORY_CONTRACT


LATENT_TRAJECTORY_ANALYSIS_VERSION = 1


def _accuracy_contrast(left, right, *, samples, seed, alpha):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    interval = paired_interval(
        left, right, samples=samples, seed=seed, alpha=alpha
    )
    return {
        "difference_points": float((left.mean() - right.mean()) * 100),
        "bootstrap_ci_points": [100 * interval[0], 100 * interval[1]],
    }


def analyze_latent_trajectory_detect(summary: dict, artifact: dict, settings) -> dict:
    if summary.get("contract") != LATENT_TRAJECTORY_CONTRACT:
        raise RuntimeError("summary belongs to another contract")
    if artifact.get("contract") != LATENT_TRAJECTORY_CONTRACT:
        raise RuntimeError("artifact belongs to another contract")
    if summary["partition_sha256"] != artifact.get("partition_sha256"):
        raise RuntimeError("summary and artifact partitions differ")

    samples = int(settings.bootstrap_samples)
    seed = int(settings.bootstrap_seed)
    alpha = float(settings.alpha)

    # ---- correctness gate --------------------------------------------------
    labels = artifact["test_correct"].numpy().astype(np.int64)
    primary_scores = artifact["correctness_scores"]["trajectory_plus_margin"].numpy()
    margin_scores = artifact["correctness_scores"]["margin"].numpy()
    if primary_scores.shape != labels.shape or margin_scores.shape != labels.shape:
        raise ValueError("correctness scores and labels must be exactly paired")
    correctness = summary["correctness"]
    optimizer_valid = bool(
        correctness["selected"]["optimization"]["converged"]
        and correctness["margin_baseline"]["optimization"]["converged"]
    )
    delta_lower, delta_upper = _auc_bootstrap(
        primary_scores, margin_scores, labels, samples=samples, seed=seed
    )
    delta_auc = float(
        correctness["selected"]["test_auc"]
        - correctness["margin_baseline"]["test_auc"]
    )
    correctness_passed = bool(
        optimizer_valid
        and delta_auc >= float(settings.minimum_delta_auc)
        and delta_lower > 0.0
    )

    # ---- answer-identity gate ----------------------------------------------
    wrong = artifact["test_wrong_mask"].numpy().astype(bool)
    if int(wrong.sum()) == 0:
        raise RuntimeError("the final test split has no wrong-answer questions")
    gold = artifact["test_gold_first_token"].numpy()
    trajectory_hits = (
        artifact["identity_predictions"]["trajectory"].numpy() == gold
    )[wrong]
    endpoint_hits = (
        artifact["identity_predictions"]["endpoint"].numpy() == gold
    )[wrong]
    majority_hits = (
        artifact["identity_predictions"]["majority"].numpy() == gold
    )[wrong]
    versus_endpoint = _accuracy_contrast(
        trajectory_hits, endpoint_hits, samples=samples, seed=seed + 1, alpha=alpha
    )
    versus_majority = _accuracy_contrast(
        trajectory_hits, majority_hits, samples=samples, seed=seed + 2, alpha=alpha
    )
    strongest_baseline_points = max(
        float(endpoint_hits.mean()), float(majority_hits.mean())
    ) * 100
    identity_gain_points = float(trajectory_hits.mean()) * 100 - strongest_baseline_points
    identity_passed = bool(
        identity_gain_points >= float(settings.minimum_identity_gain_points)
        and versus_endpoint["bootstrap_ci_points"][0] > 0.0
        and versus_majority["bootstrap_ci_points"][0] > 0.0
    )

    if identity_passed and correctness_passed:
        status = "latent_trajectory_both_supported"
    elif identity_passed:
        status = "latent_trajectory_answer_identity_supported"
    elif correctness_passed:
        status = "latent_trajectory_detect_only_supported"
    else:
        status = "latent_trajectory_not_supported"

    return {
        "analysis": "official_codi_latent_trajectory_detect",
        "analysis_version": LATENT_TRAJECTORY_ANALYSIS_VERSION,
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "status": status,
        "editing_experiment_justified": identity_passed,
        "splits": summary["splits"],
        "parity_gate": summary["parity_gate"],
        "correctness_gate": {
            "passed": correctness_passed,
            "optimizer_valid": optimizer_valid,
            "selected_cell": correctness["selected"]["cell"],
            "primary_auc": float(correctness["selected"]["test_auc"]),
            "baseline_auc": float(correctness["margin_baseline"]["test_auc"]),
            "delta_auc": delta_auc,
            "delta_ci": [delta_lower, delta_upper],
            "minimum_delta_auc": float(settings.minimum_delta_auc),
        },
        "answer_identity_gate": {
            "passed": identity_passed,
            "selected_cell": summary["answer_identity"]["selected"]["cell"],
            "wrong_questions": int(wrong.sum()),
            "trajectory_accuracy": float(trajectory_hits.mean()),
            "endpoint_probe_accuracy": float(endpoint_hits.mean()),
            "majority_accuracy": float(majority_hits.mean()),
            "gain_over_strongest_baseline_points": identity_gain_points,
            "minimum_gain_points": float(settings.minimum_identity_gain_points),
            "versus_endpoint": versus_endpoint,
            "versus_majority": versus_majority,
            "unseen_gold_class_fraction": float(
                summary["answer_identity"]["unseen_gold_class_fraction"]
            ),
        },
        "correctness_selection": correctness["selection_curve"],
        "identity_selection": summary["answer_identity"]["selection_curve"],
        "interpretation": (
            "The latent trajectory retains recoverable answer identity beyond the "
            "endpoint; a latent-state editing experiment is justified."
            if identity_passed
            else (
                "The trajectory adds correctness detection over the margin but does "
                "not recover discarded answer identity; no editing experiment is "
                "justified."
                if correctness_passed
                else "Neither frozen gate passed; the latent trajectory is not shown "
                "to hold linearly recoverable signal beyond the endpoint, and no "
                "editing experiment is justified."
            )
        ),
    }
