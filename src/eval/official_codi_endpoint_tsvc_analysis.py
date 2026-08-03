"""Paired analysis for the official-CODI endpoint TSV-C utility gate."""
from __future__ import annotations

import random
import statistics
from typing import Mapping, Sequence


ENDPOINT_TSVC_ANALYSIS_SCHEMA_VERSION = 1
PRIMARY_CONTROLS = (
    "answer_only",
    "random_rank77",
    "bottom_rank77",
    "shuffled_top77",
)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return sum(float(value) for value in values) / len(values)


def _bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> list[float]:
    if not values or samples <= 0:
        raise ValueError("bootstrap requires values and a positive sample count")
    count = len(values)
    estimates = sorted(
        _mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(samples)
    )
    return [
        float(estimates[max(0, int(0.025 * samples))]),
        float(estimates[min(samples - 1, int(0.975 * samples))]),
    ]


def _sign_flip_pvalue(
    values: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> float:
    """One-sided paired randomization p-value for a positive mean advantage."""
    observed = _mean(values)
    extreme = 0
    for _ in range(samples):
        null = _mean(
            [value if rng.random() < 0.5 else -value for value in values]
        )
        if null >= observed:
            extreme += 1
    return float((extreme + 1) / (samples + 1))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, (count - index) * float(value))
        adjusted[name] = min(1.0, running)
    return adjusted


def _batch_mean_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("paired validation loss vectors must have equal length")
    return _mean([float(a) - float(b) for a, b in zip(left, right)])


def analyze_endpoint_scope(
    batches: Sequence[Mapping],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    if not batches:
        raise ValueError("at least one completed endpoint TSV-C batch is required")
    scope = str(batches[0]["scope"])
    arm_names = tuple(sorted(batches[0]["arms"]))
    required_arms = {
        "full",
        "learned_top77",
        "random_rank77",
        "bottom_rank77",
        "shuffled_top77",
        "complement",
    }
    if not required_arms.issubset(arm_names):
        raise ValueError("endpoint TSV-C batches are missing preregistered arms")
    for batch in batches:
        if str(batch["scope"]) != scope or tuple(sorted(batch["arms"])) != arm_names:
            raise ValueError("scope or arm identity changed across batches")

    rng = random.Random(seed)
    comparison_values: dict[str, list[float]] = {
        control: [] for control in PRIMARY_CONTROLS
    }
    learned_vs_full: list[float] = []
    learned_vs_complement: list[float] = []
    arm_losses: dict[str, list[float]] = {name: [] for name in arm_names}
    answer_only_losses: list[float] = []
    original_losses: list[float] = []
    learned_cosines: list[float] = []

    for batch in batches:
        validation = batch["validation"]
        learned = batch["arms"]["learned_top77"]["validation_losses"]
        answer = validation["answer_only_losses"]
        comparison_values["answer_only"].append(
            _batch_mean_difference(answer, learned)
        )
        for control in PRIMARY_CONTROLS[1:]:
            comparison_values[control].append(
                _batch_mean_difference(
                    batch["arms"][control]["validation_losses"],
                    learned,
                )
            )
        learned_vs_full.append(
            _batch_mean_difference(
                batch["arms"]["full"]["validation_losses"], learned
            )
        )
        learned_vs_complement.append(
            _batch_mean_difference(
                batch["arms"]["complement"]["validation_losses"], learned
            )
        )
        original_losses.extend(validation["original_losses"])
        answer_only_losses.extend(answer)
        for arm in arm_names:
            arm_losses[arm].extend(batch["arms"][arm]["validation_losses"])
        learned_cosines.append(
            float(batch["arms"]["learned_top77"]["gradient_alignment"]["cosine"])
        )

    raw_pvalues = {
        name: _sign_flip_pvalue(values, samples=bootstrap_samples, rng=rng)
        for name, values in comparison_values.items()
    }
    adjusted = holm_adjust(raw_pvalues)
    comparisons = {}
    for name, values in comparison_values.items():
        ci = _bootstrap_ci(values, samples=bootstrap_samples, rng=rng)
        comparisons[name] = {
            "mean_advantage": _mean(values),
            "bootstrap_95ci": ci,
            "one_sided_sign_flip_p": raw_pvalues[name],
            "holm_adjusted_p": adjusted[name],
            "passes": ci[0] > 0.0 and adjusted[name] < 0.05,
        }
    median_cosine = float(statistics.median(learned_cosines))
    gate_passes = all(value["passes"] for value in comparisons.values()) and (
        median_cosine > 0.0
    )
    return {
        "schema_version": ENDPOINT_TSVC_ANALYSIS_SCHEMA_VERSION,
        "analysis": "official_codi_endpoint_tsvc_scope_utility",
        "scope": scope,
        "completed_update_batches": len(batches),
        "evaluated_validation_examples": len(original_losses),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_unit": "paired_update_batch",
        "learned_top77_comparisons": comparisons,
        "learned_top77_vs_full": {
            "mean_advantage": _mean(learned_vs_full),
            "bootstrap_95ci": _bootstrap_ci(
                learned_vs_full, samples=bootstrap_samples, rng=rng
            ),
            "interpretation": "positive means learned rank-77 beats the full endpoint target",
        },
        "learned_top77_vs_complement": {
            "mean_advantage": _mean(learned_vs_complement),
            "bootstrap_95ci": _bootstrap_ci(
                learned_vs_complement, samples=bootstrap_samples, rng=rng
            ),
        },
        "gradient_alignment": {
            "median_cosine": median_cosine,
            "mean_cosine": _mean(learned_cosines),
            "positive_batch_fraction": sum(value > 0 for value in learned_cosines)
            / len(learned_cosines),
        },
        "mean_heldout_answer_loss": {
            "original": _mean(original_losses),
            "answer_only": _mean(answer_only_losses),
            **{name: _mean(values) for name, values in arm_losses.items()},
        },
        "gate": {
            "status": (
                "learned_endpoint_subspace_supported"
                if gate_passes
                else "learned_endpoint_subspace_not_supported"
            ),
            "supported": gate_passes,
            "required_comparisons": list(PRIMARY_CONTROLS),
            "familywise_alpha": 0.05,
            "multiple_testing": "Holm over four one-sided paired sign-flip tests",
            "requires_positive_median_gradient_cosine": True,
        },
    }


def combine_endpoint_scope_reports(
    all_layers: Mapping,
    layer11: Mapping,
) -> dict:
    if all_layers.get("scope") != "endpoint_all_layers":
        raise ValueError("primary report must be endpoint_all_layers")
    if layer11.get("scope") != "endpoint_layer11":
        raise ValueError("secondary report must be endpoint_layer11")
    primary_request = all_layers.get("request", {})
    secondary_request = layer11.get("request", {})
    identity_fields = (
        "checkpoint_sha256",
        "dataset_fingerprint",
        "basis_sha256",
        "basis_request_sha256",
        "rank",
        "update_indices",
        "validation_indices",
        "batch_size",
        "relative_update_norm",
    )
    if any(primary_request.get(field) is None for field in identity_fields):
        raise ValueError("endpoint scope summaries are missing matched request identity")
    if any(
        primary_request.get(field) != secondary_request.get(field)
        for field in identity_fields
    ):
        raise ValueError("primary and secondary endpoint scopes are not matched")
    primary = bool(all_layers["gate"]["supported"])
    secondary = bool(layer11["gate"]["supported"])
    if primary:
        status = "endpoint_tsvc_training_authorized"
    elif secondary:
        status = "layer11_only_requires_fresh_confirmation"
    else:
        status = "endpoint_tsvc_not_supported"
    return {
        "schema_version": ENDPOINT_TSVC_ANALYSIS_SCHEMA_VERSION,
        "analysis": "official_codi_endpoint_tsvc_utility_gate",
        "status": status,
        "training_authorized": primary,
        "primary_scope": "endpoint_all_layers",
        "secondary_scope": "endpoint_layer11",
        "scopes": {
            "endpoint_all_layers": dict(all_layers),
            "endpoint_layer11": dict(layer11),
        },
        "interpretation_boundary": (
            "This is a local, one-step, equal-update-norm test at the final official "
            "CODI checkpoint. A positive primary gate authorizes a separately "
            "preregistered compute-matched training experiment; it does not itself "
            "establish long-run accuracy improvement."
        ),
    }


def render_endpoint_tsvc_markdown(report: Mapping) -> str:
    lines = [
        "# Official CODI endpoint TSV-C-inspired utility gate",
        "",
        "## Outcome",
        "",
        f"Predefined decision: **{str(report['status']).replace('_', ' ')}**.",
        "",
        (
            "The primary scope is CODI's native all-layer answer-cue endpoint. "
            "Layer 11 is a secondary localization test."
        ),
    ]
    for scope_name in ("endpoint_all_layers", "endpoint_layer11"):
        scope = report["scopes"][scope_name]
        lines.extend(
            [
                "",
                f"## {scope_name}",
                "",
                f"Gate: **{scope['gate']['status'].replace('_', ' ')}**.",
                "",
                "| Learned top-77 versus | Mean advantage | 95% CI | Holm p | Pass |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for name in PRIMARY_CONTROLS:
            value = scope["learned_top77_comparisons"][name]
            low, high = value["bootstrap_95ci"]
            lines.append(
                f"| {name} | {value['mean_advantage']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] | "
                f"{value['holm_adjusted_p']:.6g} | "
                f"{'yes' if value['passes'] else 'no'} |"
            )
        full = scope["learned_top77_vs_full"]
        low, high = full["bootstrap_95ci"]
        lines.extend(
            [
                "",
                (
                    "Learned top-77 minus full-target utility: "
                    f"{full['mean_advantage']:+.6f} "
                    f"(95% CI [{low:+.6f}, {high:+.6f}])."
                ),
                (
                    "Median learned-subspace answer-gradient cosine: "
                    f"{scope['gradient_alignment']['median_cosine']:+.6f}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Rank: 77 of 768 hidden dimensions",
            "- Primary scope: all 12 transformer blocks at the final latent endpoint",
            "- Secondary scope: transformer block 11 at the same endpoint",
            "- Bootstrap and randomization unit: paired update batch",
            "- Every auxiliary gradient is matched to the full endpoint-gradient norm",
            "- Every total parameter update has the same L2 norm",
            "",
            "## Interpretation boundary",
            "",
            str(report["interpretation_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
