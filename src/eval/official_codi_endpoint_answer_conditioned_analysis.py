"""Analysis for the answer-conditioned CODI endpoint experiment."""
from __future__ import annotations

import random
import statistics
from typing import Mapping, Sequence

from src.eval.official_codi_endpoint_tsvc_analysis import holm_adjust
from src.mech.endpoint_answer_conditioned import (
    ANSWER_CONDITIONED_ARMS,
    ANSWER_CONDITIONED_PRIMARY_CONTROLS,
    ANSWER_CONDITIONED_SCOPE,
)


ANSWER_CONDITIONED_ANALYSIS_SCHEMA_VERSION = 1


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize empty values")
    return sum(float(value) for value in values) / len(values)


def _bootstrap_ci(
    values: Sequence[float], *, samples: int, rng: random.Random
) -> list[float]:
    count = len(values)
    if not count or samples <= 0:
        raise ValueError("bootstrap requires values and positive samples")
    estimates = sorted(
        _mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(samples)
    )
    return [
        float(estimates[max(0, int(0.025 * samples))]),
        float(estimates[min(samples - 1, int(0.975 * samples))]),
    ]


def _sign_flip_pvalue(
    values: Sequence[float], *, samples: int, rng: random.Random
) -> float:
    observed = _mean(values)
    extreme = 0
    for _ in range(samples):
        null = _mean(
            [value if rng.random() < 0.5 else -value for value in values]
        )
        if null >= observed:
            extreme += 1
    return float((extreme + 1) / (samples + 1))


def _advantage(control: Sequence[float], candidate: Sequence[float]) -> float:
    if len(control) != len(candidate):
        raise ValueError("paired loss vectors must have equal lengths")
    return _mean([float(left) - float(right) for left, right in zip(control, candidate)])


def analyze_answer_conditioned_utility(
    batches: Sequence[Mapping],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    if not batches:
        raise ValueError("at least one answer-conditioned utility batch is required")
    required = set(ANSWER_CONDITIONED_ARMS)
    for batch in batches:
        if batch.get("scope") != ANSWER_CONDITIONED_SCOPE:
            raise ValueError("answer-conditioned scope changed across batches")
        if set(batch.get("arms", {})) != required:
            raise ValueError("answer-conditioned batch arms are incomplete")

    comparisons = {name: [] for name in ANSWER_CONDITIONED_PRIMARY_CONTROLS}
    candidate_vs_full: list[float] = []
    candidate_vs_complement: list[float] = []
    original_losses: list[float] = []
    answer_only_losses: list[float] = []
    arm_losses = {name: [] for name in ANSWER_CONDITIONED_ARMS}
    cosines: list[float] = []
    for batch in batches:
        candidate = batch["arms"]["answer_conditioned"]["validation_losses"]
        answer = batch["validation"]["answer_only_losses"]
        comparisons["answer_only"].append(_advantage(answer, candidate))
        for name in ANSWER_CONDITIONED_PRIMARY_CONTROLS[1:]:
            comparisons[name].append(
                _advantage(batch["arms"][name]["validation_losses"], candidate)
            )
        candidate_vs_full.append(
            _advantage(batch["arms"]["full_blocks"]["validation_losses"], candidate)
        )
        candidate_vs_complement.append(
            _advantage(batch["arms"]["complement"]["validation_losses"], candidate)
        )
        original_losses.extend(batch["validation"]["original_losses"])
        answer_only_losses.extend(answer)
        for name in ANSWER_CONDITIONED_ARMS:
            arm_losses[name].extend(batch["arms"][name]["validation_losses"])
        cosines.append(
            float(
                batch["arms"]["answer_conditioned"]["gradient_alignment"]["cosine"]
            )
        )

    rng = random.Random(seed)
    raw_pvalues = {
        name: _sign_flip_pvalue(values, samples=bootstrap_samples, rng=rng)
        for name, values in comparisons.items()
    }
    adjusted = holm_adjust(raw_pvalues)
    results = {}
    for name, values in comparisons.items():
        interval = _bootstrap_ci(values, samples=bootstrap_samples, rng=rng)
        results[name] = {
            "mean_advantage": _mean(values),
            "bootstrap_95ci": interval,
            "one_sided_sign_flip_p": raw_pvalues[name],
            "holm_adjusted_p": adjusted[name],
            "passes": interval[0] > 0 and adjusted[name] < 0.05,
        }
    median_cosine = float(statistics.median(cosines))
    supported = all(value["passes"] for value in results.values()) and median_cosine > 0
    return {
        "schema_version": ANSWER_CONDITIONED_ANALYSIS_SCHEMA_VERSION,
        "analysis": "official_codi_endpoint_answer_conditioned_utility",
        "scope": ANSWER_CONDITIONED_SCOPE,
        "completed_update_batches": len(batches),
        "evaluated_validation_examples": len(original_losses),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_unit": "paired_update_batch",
        "candidate_comparisons": results,
        "candidate_vs_full_blocks": {
            "mean_advantage": _mean(candidate_vs_full),
            "bootstrap_95ci": _bootstrap_ci(
                candidate_vs_full, samples=bootstrap_samples, rng=rng
            ),
        },
        "candidate_vs_complement": {
            "mean_advantage": _mean(candidate_vs_complement),
            "bootstrap_95ci": _bootstrap_ci(
                candidate_vs_complement, samples=bootstrap_samples, rng=rng
            ),
        },
        "gradient_alignment": {
            "median_cosine": median_cosine,
            "mean_cosine": _mean(cosines),
            "positive_batch_fraction": sum(value > 0 for value in cosines) / len(cosines),
        },
        "mean_heldout_answer_loss": {
            "original": _mean(original_losses),
            "answer_only": _mean(answer_only_losses),
            **{name: _mean(values) for name, values in arm_losses.items()},
        },
        "gate": {
            "status": (
                "answer_conditioned_endpoint_supported"
                if supported
                else "answer_conditioned_endpoint_not_supported"
            ),
            "supported": supported,
            "required_comparisons": list(ANSWER_CONDITIONED_PRIMARY_CONTROLS),
            "familywise_alpha": 0.05,
            "multiple_testing": (
                "Holm over five one-sided paired sign-flip tests"
            ),
            "requires_positive_median_gradient_cosine": True,
        },
    }


def build_answer_conditioned_final_report(
    basis_payload: Mapping,
    utility: Mapping | None,
) -> dict:
    selection = dict(basis_payload.get("selection", {}))
    metadata = dict(basis_payload.get("metadata", {}))
    if metadata.get("native_parity_gate", {}).get("status") != "passed":
        raise ValueError("answer-conditioned artifact lacks native parity")
    if selection.get("status") == "no_stable_positive_directions":
        if utility is not None:
            raise ValueError("utility cannot exist when no candidate was selected")
        status = "no_stable_answer_conditioned_directions"
        authorized = False
    else:
        if selection.get("status") != "candidate_selected" or utility is None:
            raise ValueError("selected candidate requires a completed utility report")
        if utility.get("scope") != ANSWER_CONDITIONED_SCOPE:
            raise ValueError("utility scope does not match the selected candidate")
        request = utility.get("request", {})
        if request.get("basis_request_sha256") != basis_payload.get("request_sha256"):
            raise ValueError("utility and selection artifacts are not matched")
        authorized = bool(utility.get("gate", {}).get("supported"))
        status = (
            "answer_conditioned_training_authorized"
            if authorized
            else "answer_conditioned_endpoint_not_supported"
        )
    return {
        "schema_version": ANSWER_CONDITIONED_ANALYSIS_SCHEMA_VERSION,
        "analysis": "official_codi_endpoint_answer_conditioned_final_gate",
        "status": status,
        "training_authorized": authorized,
        "selection": selection,
        "native_parity_gate": metadata["native_parity_gate"],
        "utility": dict(utility) if utility is not None else None,
        "interpretation_boundary": (
            "This is a fresh-partition, block-only, answer-conditioned local utility "
            "screen at the final official CODI checkpoint. A pass authorizes only a "
            "separately preregistered training experiment, not an accuracy claim."
        ),
    }


def render_answer_conditioned_markdown(report: Mapping) -> str:
    selection = report["selection"]
    lines = [
        "# Official CODI answer-conditioned endpoint spectral experiment",
        "",
        "## Outcome",
        "",
        f"Predefined decision: **{str(report['status']).replace('_', ' ')}**.",
        "",
        "## Direction selection",
        "",
        f"Selection status: **{selection['status'].replace('_', ' ')}**.",
        f"Rank by state: `{selection['rank_by_state']}`.",
        f"Total selected rank: {selection['total_rank']}.",
        "The embedding state is excluded; state 1 through state 12 are transformer blocks.",
    ]
    utility = report.get("utility")
    if utility is not None:
        lines.extend(
            [
                "",
                "## Held-out equal-update-norm utility",
                "",
                f"Gate: **{utility['gate']['status'].replace('_', ' ')}**.",
                "",
                "| Answer-conditioned versus | Mean advantage | 95% CI | Holm p | Pass |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for name in ANSWER_CONDITIONED_PRIMARY_CONTROLS:
            value = utility["candidate_comparisons"][name]
            low, high = value["bootstrap_95ci"]
            lines.append(
                f"| {name} | {value['mean_advantage']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] | "
                f"{value['holm_adjusted_p']:.6g} | "
                f"{'yes' if value['passes'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                (
                    "Median candidate/answer-gradient cosine: "
                    f"{utility['gradient_alignment']['median_cosine']:+.6f}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Residual PCs and answer-conditioned scores use disjoint partitions.",
            "- All questions from the completed seed-11 corrected experiment are excluded.",
            "- Selection requires positive means, the z threshold, and BH-FDR survival in both halves.",
            "- Every auxiliary gradient is matched to the full block-target gradient norm.",
            "- Every total parameter update has the same L2 norm.",
            "- Bootstrap and randomization operate on paired update batches.",
            "",
            "## Interpretation boundary",
            "",
            str(report["interpretation_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
