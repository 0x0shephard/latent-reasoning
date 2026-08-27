"""Frozen gates for the latent value-injection experiment.

Two preregistered gates, both on one read of the frozen 439-question final split,
conditioned on the baseline arm's own outcomes:

* **values causally used** (corruption): on baseline-correct questions, injecting
  plausible wrong values (gold intermediate + 1) must reduce numeric exact match
  at least 5 points more than injecting matched random numeric tokens, with a
  positive paired-bootstrap lower bound.
* **values repairable** (repair): on baseline-wrong questions, injecting the gold
  intermediates must recover numeric exact match at least 3 points more than the
  matched random arm, with a positive paired-bootstrap lower bound.

The random arm is the shared control: every arm applies identically many edits at
identical scale and differs only in which value it writes. Unlike every §43–§52
edit, this intervention enters *inside the latent loop*, so it propagates through
the KV cache and all later thoughts.
"""
from __future__ import annotations

import numpy as np

from src.eval.official_codi_paired_correction_analysis import paired_interval
from src.mech.latent_value_injection import VALUE_INJECTION_CONTRACT


VALUE_INJECTION_ANALYSIS_VERSION = 1


def _gate(left, right, *, minimum_points, samples, seed, alpha):
    interval = paired_interval(left, right, samples=samples, seed=seed, alpha=alpha)
    difference = float(
        (np.asarray(left, dtype=float) - np.asarray(right, dtype=float)).mean() * 100
    )
    return {
        "difference_points": difference,
        "bootstrap_ci_points": [100 * interval[0], 100 * interval[1]],
        "minimum_points": float(minimum_points),
        "passed": bool(difference >= float(minimum_points) and interval[0] > 0.0),
    }


def analyze_value_injection(summary: dict, outcomes: dict, settings) -> dict:
    if summary.get("contract") != VALUE_INJECTION_CONTRACT:
        raise RuntimeError("summary belongs to another contract")
    required = ("baseline", "gold", "offset", "random")
    for arm in required:
        if arm not in outcomes:
            raise ValueError(f"missing test outcomes for arm {arm!r}")
    lengths = {arm: len(outcomes[arm]["numeric_correct"]) for arm in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"arms are not row-aligned: {lengths}")
    injectable = np.asarray(outcomes["baseline"]["injectable"], dtype=bool)
    baseline = np.asarray(outcomes["baseline"]["numeric_correct"], dtype=bool)
    gold = np.asarray(outcomes["gold"]["numeric_correct"], dtype=float)
    offset = np.asarray(outcomes["offset"]["numeric_correct"], dtype=float)
    random_arm = np.asarray(outcomes["random"]["numeric_correct"], dtype=float)

    samples = int(settings.bootstrap_samples)
    seed = int(settings.bootstrap_seed)
    alpha = float(settings.alpha)

    correct_rows = baseline & injectable
    wrong_rows = (~baseline) & injectable
    if not correct_rows.any() or not wrong_rows.any():
        raise RuntimeError("both baseline outcome groups must be non-empty")

    # Corruption: damage relative to random on baseline-correct rows.
    corruption = _gate(
        1.0 - offset[correct_rows],
        1.0 - random_arm[correct_rows],
        minimum_points=settings.minimum_corruption_points,
        samples=samples,
        seed=seed,
        alpha=alpha,
    )
    corruption.update(
        {
            "baseline_correct_rows": int(correct_rows.sum()),
            "offset_accuracy": float(offset[correct_rows].mean()),
            "random_accuracy": float(random_arm[correct_rows].mean()),
        }
    )

    # Repair: recovery relative to random on baseline-wrong rows.
    repair = _gate(
        gold[wrong_rows],
        random_arm[wrong_rows],
        minimum_points=settings.minimum_repair_points,
        samples=samples,
        seed=seed + 1,
        alpha=alpha,
    )
    repair.update(
        {
            "baseline_wrong_rows": int(wrong_rows.sum()),
            "gold_accuracy": float(gold[wrong_rows].mean()),
            "random_accuracy": float(random_arm[wrong_rows].mean()),
        }
    )

    used = bool(corruption["passed"])
    repairable = bool(repair["passed"])
    if used and repairable:
        status = "values_used_and_repairable"
    elif used:
        status = "values_used_not_repairable"
    elif repairable:
        status = "values_repairable_only"
    else:
        status = "value_injection_not_supported"
    return {
        "analysis": "official_codi_latent_value_injection",
        "analysis_version": VALUE_INJECTION_ANALYSIS_VERSION,
        "contract": VALUE_INJECTION_CONTRACT,
        "status": status,
        "selected_beta": summary["selected_beta"],
        "beta_selection": summary["beta_selection"],
        "splits": summary["splits"],
        "gates": {"values_causally_used": corruption, "values_repairable": repair},
        "descriptive": {
            arm: {
                "test_accuracy_all_rows": float(
                    np.asarray(outcomes[arm]["numeric_correct"], dtype=float).mean()
                ),
                "diagnostics": outcomes[arm].get("diagnostics"),
            }
            for arm in required
        },
        "interpretation": (
            "Corrupting the workspace's value slots with plausible wrong numbers "
            "damages answers beyond matched random edits: the decoded values are "
            "causally consumed."
            + (
                " Writing the gold values additionally repairs wrong answers "
                "beyond the matched control."
                if repairable
                else " Writing the gold values did not repair wrong answers "
                "beyond the matched control: the workspace is causally used but "
                "not repairable by additive value injection."
            )
            if used
            else "Neither gate passed; the decoded workspace values are not shown "
            "to be causally consumed under this additive injection."
        ),
    }
