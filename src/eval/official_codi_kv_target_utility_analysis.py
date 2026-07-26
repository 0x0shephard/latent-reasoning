"""Aggregate hierarchical official-CODI KV target marginal-utility batches."""
from __future__ import annotations

import random
import statistics
from typing import Mapping, Sequence


TARGET_UTILITY_ANALYSIS_SCHEMA_VERSION = 1


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return sum(float(value) for value in values) / len(values)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
    alpha: float = 0.05,
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    count = len(values)
    estimates = sorted(
        _mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(samples)
    )
    low_index = max(0, int((alpha / 2) * samples))
    high_index = min(samples - 1, int((1 - alpha / 2) * samples))
    return [float(estimates[low_index]), float(estimates[high_index])]


def _paired_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    """Return left minus right at the same validation-example grain."""
    if len(left) != len(right):
        raise ValueError("paired loss vectors must have equal length")
    return [float(a) - float(b) for a, b in zip(left, right)]


def analyze_target_utility_batches(
    batches: Sequence[Mapping],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Classify target groups from held-out answer-loss changes.

    Positive utility means the candidate update yields lower answer loss:

      no-target loss - candidate loss
      shuffled-target loss - candidate loss
    """
    if not batches:
        raise ValueError("at least one completed batch is required")
    group_names = tuple(sorted(batches[0]["groups"]))
    if not group_names:
        raise ValueError("completed batches contain no target groups")
    for batch in batches:
        if tuple(sorted(batch["groups"])) != group_names:
            raise ValueError("target group set changed across completed batches")
    rng = random.Random(seed)
    original_losses: list[float] = []
    no_target_losses: list[float] = []
    for batch in batches:
        original_losses.extend(batch["validation"]["original_losses"])
        no_target_losses.extend(batch["validation"]["no_target_losses"])
    if len(original_losses) != len(no_target_losses):
        raise ValueError("validation baseline vectors are misaligned")

    summaries: dict[str, dict] = {}
    for name in group_names:
        candidate_losses: list[float] = []
        shuffled_losses: list[float] = []
        batch_vs_no_target: list[float] = []
        batch_vs_shuffled: list[float] = []
        batch_vs_original: list[float] = []
        batch_no_target_vs_original: list[float] = []
        alignment_cosines: list[float] = []
        alignment_dots: list[float] = []
        candidate_train_losses: list[float] = []
        shuffled_train_losses: list[float] = []
        definition = None
        for batch in batches:
            payload = batch["groups"][name]
            definition = definition or payload["definition"]
            original_batch = batch["validation"]["original_losses"]
            no_target_batch = batch["validation"]["no_target_losses"]
            candidate_batch = payload["candidate_validation_losses"]
            shuffled_batch = payload["shuffled_validation_losses"]
            candidate_losses.extend(candidate_batch)
            shuffled_losses.extend(shuffled_batch)
            batch_vs_no_target.append(
                _mean(_paired_difference(no_target_batch, candidate_batch))
            )
            batch_vs_shuffled.append(
                _mean(_paired_difference(shuffled_batch, candidate_batch))
            )
            batch_vs_original.append(
                _mean(_paired_difference(original_batch, candidate_batch))
            )
            batch_no_target_vs_original.append(
                _mean(_paired_difference(original_batch, no_target_batch))
            )
            alignment_cosines.append(
                float(payload["gradient_alignment"]["candidate"]["cosine"])
            )
            alignment_dots.append(
                float(payload["gradient_alignment"]["candidate"]["dot"])
            )
            candidate_train_losses.append(float(payload["candidate_train_loss"]))
            shuffled_train_losses.append(float(payload["shuffled_train_loss"]))
        if not (
            len(candidate_losses)
            == len(shuffled_losses)
            == len(no_target_losses)
            == len(original_losses)
        ):
            raise ValueError(f"validation rows are misaligned for target group {name}")

        vs_no_target = _paired_difference(no_target_losses, candidate_losses)
        vs_shuffled = _paired_difference(shuffled_losses, candidate_losses)
        # The virtual parameter update is shared by every validation example in a
        # batch. Bootstrap paired update batches rather than pretending those examples
        # are independent experimental units.
        vs_no_target_ci = _bootstrap_mean_ci(
            batch_vs_no_target,
            samples=bootstrap_samples,
            rng=rng,
        )
        vs_shuffled_ci = _bootstrap_mean_ci(
            batch_vs_shuffled,
            samples=bootstrap_samples,
            rng=rng,
        )
        median_cosine = float(statistics.median(alignment_cosines))
        if (
            vs_no_target_ci[0] > 0.0
            and vs_shuffled_ci[0] > 0.0
            and median_cosine > 0.0
        ):
            classification = "helpful_target_family"
        elif (
            vs_no_target_ci[1] < 0.0
            and vs_shuffled_ci[1] < 0.0
            and median_cosine < 0.0
        ):
            classification = "interfering_target_family"
        else:
            classification = "neutral_or_inconclusive_target_family"
        summaries[name] = {
            "definition": definition,
            "evaluated_examples": len(candidate_losses),
            "completed_batches": len(batches),
            "mean_train_distillation_loss": _mean(candidate_train_losses),
            "mean_shuffled_train_distillation_loss": _mean(
                shuffled_train_losses
            ),
            "mean_original_answer_loss": _mean(original_losses),
            "mean_no_target_answer_loss": _mean(no_target_losses),
            "mean_candidate_answer_loss": _mean(candidate_losses),
            "mean_shuffled_answer_loss": _mean(shuffled_losses),
            "candidate_utility_vs_no_target": {
                "mean": _mean(batch_vs_no_target),
                "bootstrap_95ci": vs_no_target_ci,
                "bootstrap_unit": "paired_update_batch",
                "positive_example_fraction": sum(
                    value > 0.0 for value in vs_no_target
                )
                / len(vs_no_target),
            },
            "candidate_utility_vs_shuffled": {
                "mean": _mean(batch_vs_shuffled),
                "bootstrap_95ci": vs_shuffled_ci,
                "bootstrap_unit": "paired_update_batch",
                "positive_example_fraction": sum(
                    value > 0.0 for value in vs_shuffled
                )
                / len(vs_shuffled),
            },
            "candidate_utility_vs_original": {
                "mean": _mean(batch_vs_original),
                "bootstrap_95ci": _bootstrap_mean_ci(
                    batch_vs_original,
                    samples=bootstrap_samples,
                    rng=rng,
                ),
                "bootstrap_unit": "paired_update_batch",
            },
            "no_target_utility_vs_original": {
                "mean": _mean(batch_no_target_vs_original),
                "bootstrap_95ci": _bootstrap_mean_ci(
                    batch_no_target_vs_original,
                    samples=bootstrap_samples,
                    rng=rng,
                ),
                "bootstrap_unit": "paired_update_batch",
            },
            "gradient_alignment": {
                "mean_cosine": _mean(alignment_cosines),
                "median_cosine": median_cosine,
                "positive_batch_fraction": sum(
                    value > 0.0 for value in alignment_cosines
                )
                / len(alignment_cosines),
                "mean_dot": _mean(alignment_dots),
            },
            "classification": classification,
        }

    classifications = {
        name: payload["classification"] for name, payload in summaries.items()
    }
    return {
        "schema_version": TARGET_UTILITY_ANALYSIS_SCHEMA_VERSION,
        "analysis": "official_codi_hierarchical_kv_target_marginal_utility",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_unit": "paired_update_batch",
        "completed_batches": len(batches),
        "evaluated_validation_examples": len(original_losses),
        "target_groups": summaries,
        "classifications": classifications,
        "screen_status": (
            "helpful_target_family_found"
            if "helpful_target_family" in classifications.values()
            else "no_helpful_target_family_at_this_granularity"
        ),
        "interpretation_boundary": (
            "This is a short-horizon, equal-update-norm optimization screen. "
            "Positive utility supports refining the target family and running a "
            "held-out causal test. It does not establish long-run training benefit."
        ),
    }


def render_target_utility_markdown(report: Mapping) -> str:
    lines = [
        "# Official CODI hierarchical KV target utility",
        "",
        "## Outcome",
        "",
        f"Screen status: **{str(report['screen_status']).replace('_', ' ')}**.",
        "",
        (
            "A positive value means the correctly paired target update produces "
            "lower held-out gold-answer loss. Every update has the same parameter "
            "L2 norm."
        ),
        "",
        "## Target-family results",
        "",
        (
            "| Target | Candidate vs no target | 95% CI | "
            "Candidate vs shuffled | 95% CI | Median gradient cosine | Class |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, payload in report["target_groups"].items():
        no_target = payload["candidate_utility_vs_no_target"]
        shuffled = payload["candidate_utility_vs_shuffled"]
        no_low, no_high = no_target["bootstrap_95ci"]
        shuf_low, shuf_high = shuffled["bootstrap_95ci"]
        lines.append(
            f"| {name} | {no_target['mean']:+.6f} | "
            f"[{no_low:+.6f}, {no_high:+.6f}] | "
            f"{shuffled['mean']:+.6f} | "
            f"[{shuf_low:+.6f}, {shuf_high:+.6f}] | "
            f"{payload['gradient_alignment']['median_cosine']:+.6f} | "
            f"{payload['classification'].replace('_', ' ')} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            f"- Completed batches: {report['completed_batches']}",
            (
                "- Held-out validation examples: "
                f"{report['evaluated_validation_examples']}"
            ),
            f"- Bootstrap samples: {report['bootstrap_samples']}",
            "- Correctly paired targets are compared with shuffled-pairing targets.",
            "- Keys and values are separate target families.",
            "",
            "## Interpretation boundary",
            "",
            str(report["interpretation_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
