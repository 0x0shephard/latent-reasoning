"""Gate the test-like, convergence-audited correctness detector replication."""
from __future__ import annotations

import numpy as np

from src.eval.official_codi_correctness_tracks_analysis import _auc_bootstrap


DETECT_REPLICATION_ANALYSIS_VERSION = 1
DETECT_REPLICATION_CONTRACT = (
    "frozen_checkpoint_answer_colon_test_like_detect_replication_v1"
)


def analyze_detect_replication(payload: dict, outcomes: dict, settings) -> dict:
    """Apply the frozen detect gate and surface its optimization certificate."""
    if payload.get("contract") != DETECT_REPLICATION_CONTRACT:
        raise RuntimeError("detect sweep belongs to another contract")
    labels = outcomes["labels"].numpy().astype(np.int64)
    primary = str(settings.primary_probe)
    baseline = str(settings.baseline_probe)
    probes = payload["probes"]
    scores = outcomes["scores"]
    if primary not in probes or baseline not in probes:
        raise KeyError("the primary or baseline detect probe is missing")
    if primary not in scores or baseline not in scores:
        raise KeyError("paired test scores are missing")

    primary_scores = scores[primary].numpy()
    baseline_scores = scores[baseline].numpy()
    if primary_scores.shape != labels.shape or baseline_scores.shape != labels.shape:
        raise ValueError("detect scores and labels must be exactly paired")
    lower, upper = _auc_bootstrap(
        primary_scores,
        baseline_scores,
        labels,
        samples=int(settings.bootstrap_samples),
        seed=int(settings.bootstrap_seed),
    )
    delta = float(probes[primary]["test_auc"] - probes[baseline]["test_auc"])
    optimizer_valid = bool(
        probes[primary]["optimization"]["converged"]
        and probes[baseline]["optimization"]["converged"]
    )
    passed = bool(
        optimizer_valid
        and delta >= float(settings.minimum_delta_auc)
        and lower > 0.0
    )
    return {
        "analysis_version": DETECT_REPLICATION_ANALYSIS_VERSION,
        "contract": DETECT_REPLICATION_CONTRACT,
        "status": (
            "test_like_detect_supported"
            if passed
            else "test_like_detect_not_supported"
        ),
        "population": payload["population"],
        "splits": payload["splits"],
        "primary_probe": primary,
        "baseline_probe": baseline,
        "primary_auc": float(probes[primary]["test_auc"]),
        "baseline_auc": float(probes[baseline]["test_auc"]),
        "delta_auc": delta,
        "delta_ci": [lower, upper],
        "minimum_delta_auc": float(settings.minimum_delta_auc),
        "optimizer_valid": optimizer_valid,
        "passed": passed,
        "probes": {
            name: {
                "ridge": float(entry["ridge"]),
                "feature_count": int(entry["feature_count"]),
                "select_auc": float(entry["select_auc"]),
                "test_auc": float(entry["test_auc"]),
                "optimization": entry["optimization"],
            }
            for name, entry in sorted(probes.items())
        },
        "interpretation": (
            "This replication estimates the detector on a held-out partition of "
            "GSM8K test, which is test-like relative to the original GSM8K-train "
            "calibration. Its smaller final test split gives wider uncertainty, "
            "and the dataset has been inspected by the project before, so this is "
            "a corrective replication rather than a pristine preregistration."
        ),
    }
