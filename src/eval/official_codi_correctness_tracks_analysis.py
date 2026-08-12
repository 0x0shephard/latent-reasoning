"""Preregistered gates for the three correctness tracks.

Each track has one primary gate, fixed before any test number is read, and each
is stated as a comparison against the thing that would otherwise explain the
result:

detect
    Not "is the probe above chance" -- a probe can sit at 0.70 while the model's
    own margin sits at 0.874, in which case the state told us nothing usable.
    The gate is the *increment* over a margin-only probe.

steer
    Not "did accuracy go up" -- any perturbation moves a few of 1319 questions.
    The gate is a bootstrap lower bound above zero *and* a margin over the best
    of the matched random directions drawn inside the same band.

project
    Not "does the correct-only subspace retain accuracy" -- the class-blind one
    already does. The gate is whether restricting to correct examples *beats* the
    class-blind subspace at matched rank.

A failed gate is a result. The steer track in particular is expected to fail if
the volume-knob account of the leading directions is right, and saying so in
advance is what makes the outcome informative either way.
"""
from __future__ import annotations

import numpy as np

from src.eval.official_codi_endpoint_band_confirmation_analysis import (
    _exact_mcnemar_one_sided,
    _paired_accuracy_bootstrap,
)

CORRECTNESS_TRACKS_ANALYSIS_VERSION = 1


def _auc_bootstrap(
    scores_left: np.ndarray,
    scores_right: np.ndarray,
    labels: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the AUC difference, resampling questions rather than pairs.

    AUC is not a mean over questions, so the paired-mean bootstrap used for
    accuracy does not apply; each replicate recomputes both AUCs on the same
    resampled question set, which keeps the comparison paired.
    """
    generator = np.random.default_rng(seed)
    size = labels.size
    indices = generator.integers(0, size, size=(samples, size))
    deltas = np.empty(samples, dtype=np.float64)
    for row in range(samples):
        pick = indices[row]
        drawn = labels[pick]
        if drawn.min() == drawn.max():
            deltas[row] = 0.0
            continue
        deltas[row] = _auc(scores_left[pick], drawn) - _auc(scores_right[pick], drawn)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware AUC, matching :func:`src.mech...roc_auc`.

    Bootstrap resampling draws the same question more than once by construction,
    so ties are the normal case here rather than an edge case, and breaking them
    by array position would bias every replicate.
    """
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_scores[1:] != sorted_scores[:-1], [True]))
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop + 1) / 2.0
    positives = labels.sum()
    negatives = labels.size - positives
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _random_arm_names(steer: dict, prefix: str) -> list[str]:
    return sorted(name for name in steer if name.startswith(prefix))


def analyze_correctness_tracks(
    payload: dict,
    outcomes: dict,
    settings,
) -> dict:
    """Apply the three primary gates and every secondary comparison."""
    samples = int(settings.bootstrap_samples)
    seed = int(settings.bootstrap_seed)
    labels = outcomes["labels"].numpy().astype(np.int64)
    baseline = outcomes["baseline"].numpy().astype(bool)

    # ---------------- detect ----------------
    detect = payload["detect"]
    margin_scores = outcomes["detect"]["margin"].numpy()
    primary_probe = settings.detect_primary_probe
    if primary_probe not in detect:
        raise KeyError(f"detect probe {primary_probe} is missing from the sweep")
    probe_scores = outcomes["detect"][primary_probe].numpy()
    low, high = _auc_bootstrap(
        probe_scores, margin_scores, labels, samples=samples, seed=seed
    )
    delta = detect[primary_probe]["test_auc"] - detect["margin"]["test_auc"]
    detect_report = {
        "primary_probe": primary_probe,
        "probe_auc": detect[primary_probe]["test_auc"],
        "margin_auc": detect["margin"]["test_auc"],
        "delta_auc": delta,
        "delta_ci": [low, high],
        "minimum_delta_auc": settings.minimum_detect_delta_auc,
        "passed": bool(low > 0.0 and delta >= settings.minimum_detect_delta_auc),
        "all_probes": {
            name: {"test_auc": entry["test_auc"], "select_auc": entry["select_auc"]}
            for name, entry in sorted(detect.items())
        },
    }

    # ---------------- steer ----------------
    steer = payload["steer"]
    primary_arm = settings.steer_primary_arm
    if primary_arm not in steer:
        raise KeyError(f"steer arm {primary_arm} is missing from the sweep")
    arm_outcomes = outcomes["steer"][primary_arm].numpy().astype(bool)
    low, high = _paired_accuracy_bootstrap(
        arm_outcomes, baseline, samples=samples, seed=seed
    )
    base_accuracy = payload["baseline_first_token_accuracy"]
    gain = steer[primary_arm]["test_accuracy"] - base_accuracy
    random_names = _random_arm_names(steer, "random_band_r")
    best_random = max(
        (steer[name]["test_accuracy"] for name in random_names), default=base_accuracy
    )
    steer_report = {
        "primary_arm": primary_arm,
        "selected_alpha": steer[primary_arm]["selected_alpha"],
        "baseline_accuracy": base_accuracy,
        "arm_accuracy": steer[primary_arm]["test_accuracy"],
        "gain_points": 100.0 * gain,
        "gain_ci_points": [100.0 * low, 100.0 * high],
        "best_random_band_accuracy": best_random,
        "margin_over_random_points": 100.0
        * (steer[primary_arm]["test_accuracy"] - best_random),
        "mcnemar_p_worse": _exact_mcnemar_one_sided(baseline, arm_outcomes),
        "minimum_gain_points": settings.minimum_steer_gain_points,
        "passed": bool(
            low > 0.0
            and 100.0 * gain >= settings.minimum_steer_gain_points
            and steer[primary_arm]["test_accuracy"] > best_random
        ),
        "all_arms": {
            name: {
                "test_accuracy": entry["test_accuracy"],
                "selected_alpha": entry["selected_alpha"],
                "band_profile": entry["band_profile"],
            }
            for name, entry in sorted(steer.items())
            if not name.startswith("random_")
        },
        "random_controls": {
            "band": [steer[name]["test_accuracy"] for name in random_names],
            "global": [
                steer[name]["test_accuracy"]
                for name in _random_arm_names(steer, "random_global_r")
            ],
        },
    }

    # ---------------- project ----------------
    project = payload["project"]
    rank = str(settings.project_primary_rank)
    if rank not in project:
        raise KeyError(f"project rank {rank} is missing from the sweep")
    correct_outcomes = outcomes["project"][rank]["correct_only"].numpy().astype(bool)
    blind_outcomes = outcomes["project"][rank]["class_blind"].numpy().astype(bool)
    low, high = _paired_accuracy_bootstrap(
        correct_outcomes, blind_outcomes, samples=samples, seed=seed
    )
    advantage = (
        project[rank]["correct_only"]["accuracy"]
        - project[rank]["class_blind"]["accuracy"]
    )
    project_report = {
        "rank": int(settings.project_primary_rank),
        "correct_only_accuracy": project[rank]["correct_only"]["accuracy"],
        "class_blind_accuracy": project[rank]["class_blind"]["accuracy"],
        "advantage_points": 100.0 * advantage,
        "advantage_ci_points": [100.0 * low, 100.0 * high],
        "overlap_with_class_blind": project[rank]["overlap_with_class_blind"],
        "minimum_advantage_points": settings.minimum_project_advantage_points,
        "passed": bool(
            low > 0.0
            and 100.0 * advantage >= settings.minimum_project_advantage_points
        ),
        "by_rank": {
            key: {
                "class_blind": entry["class_blind"]["accuracy"],
                "correct_only": entry["correct_only"]["accuracy"],
                "incorrect_only": entry["incorrect_only"]["accuracy"],
                "mean_cosine": entry["overlap_with_class_blind"]["mean_cosine"],
            }
            for key, entry in project.items()
            if key != "accuracy_band"
        },
    }

    return {
        "analysis_version": CORRECTNESS_TRACKS_ANALYSIS_VERSION,
        "contract": payload["contract"],
        "splits": payload["splits"],
        "geometry": _geometry_report(payload["geometry"]),
        "detect": detect_report,
        "steer": steer_report,
        "project": project_report,
        "tracks_passed": {
            "detect": detect_report["passed"],
            "steer": steer_report["passed"],
            "project": project_report["passed"],
        },
    }


def _geometry_report(geometry: dict) -> dict:
    """Summarise the class split, with the mean difference judged against its null."""
    null = geometry["random_split_null"]
    observed_share = geometry["mean_difference_bands"]["0:4"]
    observed_norm = geometry["mean_difference_norm"]
    shares = np.asarray(null["band_shares"], dtype=np.float64)
    norms = np.asarray(null["norms"], dtype=np.float64)
    return {
        "between_class_fraction": geometry["between_class_fraction"],
        "mean_difference_norm": observed_norm,
        "lift_band_share": observed_share,
        "accuracy_band_share": geometry["mean_difference_bands"]["4:32"],
        "fisher_lift_band_share": geometry["fisher_bands"]["0:4"],
        "fisher_accuracy_band_share": geometry["fisher_bands"]["4:32"],
        "variance_shares": geometry["variance_shares"],
        "null": {
            "replicates": int(null["replicates"]),
            "median_lift_band_share": float(np.median(shares)),
            "share_exceedances": int((shares >= observed_share).sum()),
            "median_norm": float(np.median(norms)),
            "norm_ratio": float(observed_norm / max(np.median(norms), 1e-12)),
        },
    }
