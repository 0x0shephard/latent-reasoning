"""Held-out analysis for sparse answer-aligned KV gradient components."""
from __future__ import annotations

import random
import statistics
from typing import Mapping, Sequence


KV_GRADIENT_SIGNAL_ANALYSIS_SCHEMA_VERSION = 1


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return sum(float(value) for value in values) / len(values)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> list[float]:
    if not values or samples <= 0:
        raise ValueError("bootstrap requires values and positive samples")
    count = len(values)
    estimates = sorted(
        _mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(samples)
    )
    return [
        float(estimates[max(0, int(0.025 * samples))]),
        float(estimates[min(samples - 1, int(0.975 * samples))]),
    ]


def _comparison(
    batches: Sequence[Mapping],
    *,
    kind: str,
    left: str,
    right: str,
    bootstrap_samples: int,
    rng: random.Random,
) -> dict:
    """Return mean left-loss minus right-loss; positive means right is better."""
    values = []
    for batch in batches:
        if left == "no_target":
            left_losses = batch["validation"]["no_target_losses"]
        else:
            left_losses = batch["kinds"][kind]["conditions"][left][
                "validation_losses"
            ]
        if right == "no_target":
            right_losses = batch["validation"]["no_target_losses"]
        else:
            right_losses = batch["kinds"][kind]["conditions"][right][
                "validation_losses"
            ]
        if len(left_losses) != len(right_losses):
            raise ValueError("paired validation loss vectors differ in length")
        values.append(
            _mean(
                [
                    float(left_value) - float(right_value)
                    for left_value, right_value in zip(left_losses, right_losses)
                ]
            )
        )
    return {
        "mean": _mean(values),
        "bootstrap_95ci": _bootstrap_mean_ci(
            values,
            samples=bootstrap_samples,
            rng=rng,
        ),
        "bootstrap_unit": "paired_update_batch",
        "positive_batch_fraction": sum(value > 0 for value in values) / len(values),
    }


def analyze_kv_gradient_signal_batches(
    batches: Sequence[Mapping],
    *,
    primary_kind: str = "key",
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    if not batches:
        raise ValueError("at least one completed batch is required")
    kinds = tuple(sorted(batches[0]["kinds"]))
    if primary_kind not in kinds:
        raise ValueError("primary kind is absent from completed batches")
    required_conditions = {
        "full",
        "sparse_aligned",
        "random_sparse",
        "shuffled_sparse",
        "complement",
    }
    for batch in batches:
        if tuple(sorted(batch["kinds"])) != kinds:
            raise ValueError("KV kinds changed across completed batches")
        for kind in kinds:
            if set(batch["kinds"][kind]["conditions"]) != required_conditions:
                raise ValueError(f"condition set changed for {kind}")

    rng = random.Random(seed)
    summaries = {}
    for kind in kinds:
        comparisons = {
            "sparse_vs_no_target": _comparison(
                batches,
                kind=kind,
                left="no_target",
                right="sparse_aligned",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "sparse_vs_full": _comparison(
                batches,
                kind=kind,
                left="full",
                right="sparse_aligned",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "sparse_vs_random": _comparison(
                batches,
                kind=kind,
                left="random_sparse",
                right="sparse_aligned",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "sparse_vs_shuffled": _comparison(
                batches,
                kind=kind,
                left="shuffled_sparse",
                right="sparse_aligned",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "sparse_vs_complement": _comparison(
                batches,
                kind=kind,
                left="complement",
                right="sparse_aligned",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "full_vs_no_target": _comparison(
                batches,
                kind=kind,
                left="no_target",
                right="full",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            "complement_vs_no_target": _comparison(
                batches,
                kind=kind,
                left="no_target",
                right="complement",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
        }
        sparse_cosines = [
            float(
                batch["kinds"][kind]["conditions"]["sparse_aligned"][
                    "gradient_alignment"
                ]["cosine"]
            )
            for batch in batches
        ]
        positive_sparse_tests = all(
            comparisons[name]["bootstrap_95ci"][0] > 0
            for name in (
                "sparse_vs_no_target",
                "sparse_vs_full",
                "sparse_vs_random",
                "sparse_vs_shuffled",
                "sparse_vs_complement",
            )
        )
        positive_alignment = statistics.median(sparse_cosines) > 0
        complement_not_helpful = (
            comparisons["complement_vs_no_target"]["bootstrap_95ci"][1] <= 0
        )
        if positive_sparse_tests and positive_alignment and complement_not_helpful:
            classification = "sparse_answer_aligned_component_only_supported"
        elif positive_sparse_tests and positive_alignment:
            classification = "sparse_answer_aligned_component_supported_not_only"
        else:
            classification = "sparse_answer_aligned_component_not_supported"
        summaries[kind] = {
            "comparisons": comparisons,
            "sparse_gradient_alignment": {
                "mean_cosine": _mean(sparse_cosines),
                "median_cosine": float(statistics.median(sparse_cosines)),
                "positive_batch_fraction": sum(value > 0 for value in sparse_cosines)
                / len(sparse_cosines),
            },
            "criteria": {
                "all_sparse_comparison_lower_bounds_positive": positive_sparse_tests,
                "median_sparse_gradient_cosine_positive": positive_alignment,
                "complement_95ci_upper_not_positive": complement_not_helpful,
            },
            "classification": classification,
        }

    primary_classification = summaries[primary_kind]["classification"]
    return {
        "schema_version": KV_GRADIENT_SIGNAL_ANALYSIS_SCHEMA_VERSION,
        "analysis": "official_codi_sparse_answer_aligned_kv_gradient_signal",
        "primary_kind": primary_kind,
        "completed_batches": len(batches),
        "evaluated_validation_examples": sum(
            len(batch["validation"]["no_target_losses"]) for batch in batches
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_unit": "paired_update_batch",
        "by_kind": summaries,
        "gate": (
            "primary_sparse_component_only_supported"
            if primary_classification
            == "sparse_answer_aligned_component_only_supported"
            else (
                "primary_sparse_component_supported_not_only"
                if primary_classification
                == "sparse_answer_aligned_component_supported_not_only"
                else "primary_sparse_component_not_supported"
            )
        ),
        "interpretation_boundary": (
            "This test concerns one-step held-out gold-answer loss at the frozen "
            "official CODI checkpoint. A positive gate does not establish exact-match "
            "accuracy improvement or long-run distillation benefit."
        ),
    }


def render_kv_gradient_signal_markdown(report: Mapping) -> str:
    lines = [
        "# Official CODI sparse answer-aligned KV gradient signal",
        "",
        "## Outcome",
        "",
        f"Predefined primary gate: **{str(report['gate']).replace('_', ' ')}**.",
        "",
        (
            "Positive comparison values mean that the frozen calibration-selected "
            "sparse component produces lower held-out answer loss."
        ),
        "",
        "## Held-out comparisons",
        "",
        (
            "| KV kind | Comparison | Mean advantage | 95% CI | "
            "Primary classification |"
        ),
        "| --- | --- | ---: | ---: | --- |",
    ]
    labels = {
        "sparse_vs_no_target": "sparse vs no target",
        "sparse_vs_full": "sparse vs full KV",
        "sparse_vs_random": "sparse vs random sparse",
        "sparse_vs_shuffled": "sparse vs shuffled sparse",
        "sparse_vs_complement": "sparse vs complement",
        "full_vs_no_target": "full KV vs no target",
        "complement_vs_no_target": "complement vs no target",
    }
    for kind, payload in report["by_kind"].items():
        for name, comparison in payload["comparisons"].items():
            low, high = comparison["bootstrap_95ci"]
            lines.append(
                f"| {kind} | {labels[name]} | {comparison['mean']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] | "
                f"{payload['classification'].replace('_', ' ')} |"
            )
    lines.extend(
        [
            "",
            "## Gradient alignment",
            "",
            "| KV kind | Median cosine | Positive batch fraction |",
            "| --- | ---: | ---: |",
        ]
    )
    for kind, payload in report["by_kind"].items():
        alignment = payload["sparse_gradient_alignment"]
        lines.append(
            f"| {kind} | {alignment['median_cosine']:+.6f} | "
            f"{alignment['positive_batch_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            f"- Primary KV kind: {report['primary_kind']}",
            f"- Completed update batches: {report['completed_batches']}",
            (
                "- Held-out validation examples: "
                f"{report['evaluated_validation_examples']}"
            ),
            f"- Bootstrap samples: {report['bootstrap_samples']}",
            "- Calibration, update, and validation question groups are disjoint.",
            "- Every auxiliary gradient is energy-matched to the full KV gradient.",
            "- Every total parameter update has the same L2 norm.",
            "",
            "## Interpretation boundary",
            "",
            str(report["interpretation_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
