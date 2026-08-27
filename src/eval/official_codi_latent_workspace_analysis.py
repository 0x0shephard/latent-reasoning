"""Frozen gates for the latent-workspace confirmation.

Four preregistered gates on one read of the frozen 439-question final split, with
thresholds set from the §53 fit/select observations before that read:

* **content** — mean recovery of gold intermediates beats the seeded derangement
  null by ≥ 10 points, positive paired-bootstrap lower bound.
* **structure** — gold-intermediate hits at even thoughts are ≤ 10% of all hits,
  and every odd thought's per-question hit rate is ≥ 0.3.
* **correct/wrong gap** — recovery on correctly answered questions beats recovery
  on wrongly answered ones by ≥ 5 points, positive two-sample bootstrap lower
  bound.
* **faithful readout** — on wrong questions, the model's own first answer token
  appears among the decoded thought numbers at least 4 points more often than the
  gold first token, positive paired-bootstrap lower bound.

The thought-to-step alignment table is preregistered as *descriptive only*: the
fit/select observation is that alignment is absent, and no gate depends on it.
"""
from __future__ import annotations

import numpy as np

from src.eval.official_codi_paired_correction_analysis import paired_interval
from src.mech.latent_workspace import LATENT_WORKSPACE_CONTRACT


LATENT_WORKSPACE_ANALYSIS_VERSION = 1


def two_sample_interval(left, right, *, samples: int, seed: int, alpha: float):
    """Bootstrap CI for mean(left) − mean(right) with independent groups."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.size == 0 or right.size == 0:
        raise ValueError("both groups must be non-empty 1-D vectors")
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=float)
    for index in range(int(samples)):
        means[index] = generator.choice(left, left.size).mean() - generator.choice(
            right, right.size
        ).mean()
    return [
        float(value)
        for value in np.quantile(means, [float(alpha) / 2, 1 - float(alpha) / 2])
    ]


def _paired_gate(left, right, *, minimum_points, samples, seed, alpha):
    interval = paired_interval(left, right, samples=samples, seed=seed, alpha=alpha)
    difference = float(
        (np.asarray(left, dtype=float) - np.asarray(right, dtype=float)).mean() * 100
    )
    return {
        "difference_points": difference,
        "bootstrap_ci_points": [100 * interval[0], 100 * interval[1]],
        "minimum_points": float(minimum_points),
        "passed": bool(
            difference >= float(minimum_points) and interval[0] > 0.0
        ),
    }


def analyze_latent_workspace(summary: dict, artifact: dict, settings) -> dict:
    if summary.get("contract") != LATENT_WORKSPACE_CONTRACT:
        raise RuntimeError("summary belongs to another contract")
    if artifact.get("contract") != LATENT_WORKSPACE_CONTRACT:
        raise RuntimeError("artifact belongs to another contract")
    if summary["partition_sha256"] != artifact.get("partition_sha256"):
        raise RuntimeError("summary and artifact partitions differ")

    samples = int(settings.bootstrap_samples)
    seed = int(settings.bootstrap_seed)
    alpha = float(settings.alpha)

    recovery = artifact["test_recovery"].numpy()
    null_recovery = artifact["test_null_recovery"].numpy()
    scored = artifact["test_scored_mask"].numpy().astype(bool)
    correct = artifact["test_correct"].numpy().astype(bool)
    if recovery.shape != null_recovery.shape or recovery.shape != scored.shape:
        raise ValueError("recovery vectors must be exactly paired")

    content = _paired_gate(
        recovery[scored],
        null_recovery[scored],
        minimum_points=settings.minimum_content_points,
        samples=samples,
        seed=seed,
        alpha=alpha,
    )
    content.update(
        {
            "mean_recovery": float(recovery[scored].mean()),
            "mean_null_recovery": float(null_recovery[scored].mean()),
            "scored_questions": int(scored.sum()),
        }
    )

    hit_matrix = artifact["test_thought_hits"].numpy().astype(bool)[scored]
    per_thought = hit_matrix.sum(axis=0)
    total_hits = int(per_thought.sum())
    odd = [1, 3, 5]
    even = [0, 2, 4]
    even_share = (
        float(per_thought[even].sum()) / total_hits if total_hits else 1.0
    )
    odd_rates = [float(hit_matrix[:, slot].mean()) for slot in odd]
    structure = {
        "hits_per_thought": [int(v) for v in per_thought],
        "even_hit_share": even_share,
        "maximum_even_hit_share": float(settings.maximum_even_hit_share),
        "odd_hit_rates": odd_rates,
        "minimum_odd_hit_rate": float(settings.minimum_odd_hit_rate),
        "passed": bool(
            even_share <= float(settings.maximum_even_hit_share)
            and all(
                rate >= float(settings.minimum_odd_hit_rate) for rate in odd_rates
            )
        ),
    }

    gap_interval = two_sample_interval(
        recovery[scored & correct],
        recovery[scored & ~correct],
        samples=samples,
        seed=seed + 1,
        alpha=alpha,
    )
    gap_points = float(
        (recovery[scored & correct].mean() - recovery[scored & ~correct].mean())
        * 100
    )
    gap = {
        "recovery_correct": float(recovery[scored & correct].mean()),
        "recovery_wrong": float(recovery[scored & ~correct].mean()),
        "difference_points": gap_points,
        "bootstrap_ci_points": [100 * v for v in gap_interval],
        "minimum_points": float(settings.minimum_gap_points),
        "passed": bool(
            gap_points >= float(settings.minimum_gap_points)
            and gap_interval[0] > 0.0
        ),
    }

    own_hits = artifact["test_own_token_in_thoughts"].numpy().astype(float)
    gold_hits = artifact["test_gold_token_in_thoughts"].numpy().astype(float)
    wrong = ~correct
    tracing = _paired_gate(
        own_hits[wrong],
        gold_hits[wrong],
        minimum_points=settings.minimum_tracing_points,
        samples=samples,
        seed=seed + 2,
        alpha=alpha,
    )
    tracing.update(
        {
            "own_token_rate": float(own_hits[wrong].mean()),
            "gold_token_rate": float(gold_hits[wrong].mean()),
            "wrong_questions": int(wrong.sum()),
        }
    )

    gates = {
        "content": content,
        "structure": structure,
        "correct_wrong_gap": gap,
        "faithful_readout": tracing,
    }
    passed = [name for name, gate in gates.items() if gate["passed"]]
    confirmed = len(passed) == len(gates)
    return {
        "analysis": "official_codi_latent_workspace",
        "analysis_version": LATENT_WORKSPACE_ANALYSIS_VERSION,
        "contract": LATENT_WORKSPACE_CONTRACT,
        "status": (
            "workspace_confirmed" if confirmed else "workspace_not_confirmed"
        ),
        "gates_passed": passed,
        "workspace_confirmed": confirmed,
        "splits": summary["splits"],
        "gates": gates,
        "alignment_table_descriptive": summary["alignment_table"],
        "interpretation": (
            "The latent thoughts are a measurable workspace: they carry the gold "
            "intermediate values far above a matched null, in alternating value "
            "slots, with less of the right content on wrongly answered questions, "
            "and the endpoint's wrong answers trace back to the workspace's own "
            "numbers."
            if confirmed
            else "Not every frozen workspace gate passed; only the passed gates "
            "may be asserted, and the §53 exploration is not confirmed as a whole."
        ),
    }
