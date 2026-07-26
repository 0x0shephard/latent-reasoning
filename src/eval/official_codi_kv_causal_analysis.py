"""Paired causal analysis for official CODI KV-subspace interventions."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Sequence

from src.eval.compare_runs import (
    EvalRecord,
    _bootstrap_ci,
    _mcnemar_exact_p,
    align_records,
    load_eval_run,
)


CAUSAL_ANALYSIS_SCHEMA_VERSION = 1


def _accuracy(records: Sequence[EvalRecord]) -> float:
    return sum(record.correct for record in records) / len(records)


def _paired_delta(
    left: Sequence[EvalRecord],
    right: Sequence[EvalRecord],
    *,
    bootstrap_samples: int,
    rng: random.Random,
) -> dict:
    """Return ``right minus left`` paired accuracy statistics."""
    deltas = [int(r.correct) - int(l.correct) for l, r in zip(left, right)]
    left_only = sum(l.correct and not r.correct for l, r in zip(left, right))
    right_only = sum(not l.correct and r.correct for l, r in zip(left, right))
    return {
        "delta_definition": "right_minus_left",
        "accuracy_delta": sum(deltas) / len(deltas),
        "accuracy_delta_95ci": _bootstrap_ci(
            deltas,
            samples=bootstrap_samples,
            rng=rng,
        ),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": _mcnemar_exact_p(left_only, right_only),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjust a named family of p-values while preserving monotonicity."""
    count = len(p_values)
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def analyze_causal_interventions(
    evaluation_root: str | Path,
    *,
    scopes: Sequence[str],
    primary_positions: Sequence[int] = (4, 5),
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    familywise_alpha: float = 0.05,
) -> dict:
    """Analyze learned directions against energy-matched random directions.

    The primary causal contrast is always learned minus random at the same
    intervention mode and latent-position scope. Retain tests sufficiency and remove
    tests necessity. Intervention-minus-baseline effects are reported descriptively.
    """
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be between zero and one")
    root = Path(evaluation_root).expanduser().resolve()
    baseline_run = load_eval_run(root / "baseline")
    if set(baseline_run.datasets) != {"gsm8k"}:
        raise ValueError("causal evaluation must contain only GSM8K")
    baseline = baseline_run.datasets["gsm8k"]
    rng = random.Random(seed)
    comparisons: dict[str, dict] = {}

    for scope in scopes:
        for mode in ("retain", "remove"):
            learned_name = f"{mode}_learned_{scope}"
            random_name = f"{mode}_random_{scope}"
            learned_run = load_eval_run(root / learned_name)
            random_run = load_eval_run(root / random_name)
            learned = learned_run.datasets["gsm8k"]
            random_rows = random_run.datasets["gsm8k"]
            baseline_aligned, learned = align_records(
                baseline, learned, "gsm8k"
            )
            baseline_again, random_rows = align_records(
                baseline, random_rows, "gsm8k"
            )
            if baseline_aligned != baseline_again:
                raise ValueError("baseline alignment changed across intervention arms")
            learned, random_rows = align_records(
                learned, random_rows, "gsm8k"
            )
            comparison_name = f"{mode}_{scope}"
            comparisons[comparison_name] = {
                "mode": mode,
                "scope": scope,
                "count": len(baseline),
                "baseline_accuracy": _accuracy(baseline),
                "learned_accuracy": _accuracy(learned),
                "random_accuracy": _accuracy(random_rows),
                "learned_minus_baseline": _paired_delta(
                    baseline_aligned,
                    learned,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                ),
                "random_minus_baseline": _paired_delta(
                    baseline_again,
                    random_rows,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                ),
                "learned_minus_random": _paired_delta(
                    random_rows,
                    learned,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                ),
            }

    primary_names = [
        f"{mode}_p{position}"
        for position in primary_positions
        for mode in ("retain", "remove")
    ]
    missing = [name for name in primary_names if name not in comparisons]
    if missing:
        raise ValueError(f"primary comparisons are missing: {missing}")
    adjusted = holm_adjust(
        {
            name: comparisons[name]["learned_minus_random"]["mcnemar_exact_p"]
            for name in primary_names
        }
    )
    supported_tests: list[str] = []
    for name in primary_names:
        result = comparisons[name]
        specificity = result["learned_minus_random"]
        low, high = specificity["accuracy_delta_95ci"]
        direction_pass = (
            specificity["accuracy_delta"] > 0.0 and low > 0.0
            if result["mode"] == "retain"
            else specificity["accuracy_delta"] < 0.0 and high < 0.0
        )
        test = {
            "hypothesis": (
                "learned retention is more accurate than energy-matched random "
                "retention"
                if result["mode"] == "retain"
                else "removing learned directions is more harmful than removing "
                "energy-matched random directions"
            ),
            "direction_pass": direction_pass,
            "holm_adjusted_p": adjusted[name],
            "familywise_alpha": familywise_alpha,
            "supported": direction_pass and adjusted[name] < familywise_alpha,
        }
        result["primary_gate"] = test
        if test["supported"]:
            supported_tests.append(name)

    return {
        "schema_version": CAUSAL_ANALYSIS_SCHEMA_VERSION,
        "analysis": "official_codi_student_kv_spectral_causality",
        "evaluation_root": str(root),
        "dataset": "gsm8k",
        "evaluated_count": len(baseline),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "primary_positions": list(primary_positions),
        "familywise_alpha": familywise_alpha,
        "multiple_testing": (
            "Holm correction across retain/remove at positions "
            + ", ".join(str(position) for position in primary_positions)
        ),
        "gate": {
            "status": (
                "learned_subspace_causality_supported"
                if supported_tests
                else "learned_subspace_causality_not_supported"
            ),
            "supported_primary_tests": supported_tests,
            "required_evidence": (
                "directional learned-versus-energy-matched-random paired accuracy "
                "difference, 95% paired-bootstrap interval excluding zero, and "
                "Holm-adjusted exact McNemar p below alpha"
            ),
        },
        "comparisons": comparisons,
        "interpretation_boundary": (
            "A positive primary test shows that the learned rank-four student KV "
            "directions have position-specific causal value beyond an "
            "energy-matched random rank-four subspace in the official CODI "
            "checkpoint. It does not establish that the directions encode a "
            "human-interpretable reasoning algorithm or that distilling them will "
            "improve a newly trained student."
        ),
    }


def render_causal_markdown(report: Mapping) -> str:
    lines = [
        "# Official CODI KV spectral causality",
        "",
        "## Outcome",
        "",
        (
            "Predefined primary gate: "
            f"**{str(report['gate']['status']).replace('_', ' ')}**."
        ),
        "",
        (
            "Learned rank-four student KV directions are compared with "
            "energy-matched random rank-four directions on the same full GSM8K "
            "examples. Retain tests sufficiency. Remove tests necessity."
        ),
        "",
        "## Evaluation contract",
        "",
        f"Evaluated examples: {report['evaluated_count']}",
        f"Bootstrap samples: {report['bootstrap_samples']}",
        "Primary latent positions: "
        + ", ".join(str(item) for item in report["primary_positions"]),
        f"Multiple testing: {report['multiple_testing']}",
        "",
        "## Primary position results",
        "",
        (
            "| Position | Intervention | Baseline | Learned | Random | "
            "Learned minus random | 95% CI | Holm p | Gate |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    primary = {f"p{position}" for position in report["primary_positions"]}
    for result in report["comparisons"].values():
        if result["scope"] not in primary:
            continue
        specificity = result["learned_minus_random"]
        low, high = specificity["accuracy_delta_95ci"]
        gate = result["primary_gate"]
        lines.append(
            f"| {result['scope'][1:]} | {result['mode']} | "
            f"{result['baseline_accuracy']:.4f} | "
            f"{result['learned_accuracy']:.4f} | "
            f"{result['random_accuracy']:.4f} | "
            f"{specificity['accuracy_delta']:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | "
            f"{gate['holm_adjusted_p']:.4g} | "
            f"{'supported' if gate['supported'] else 'not supported'} |"
        )

    lines.extend(
        [
            "",
            "## Position sweep",
            "",
            (
                "| Scope | Intervention | Learned effect vs baseline | "
                "Random effect vs baseline | Learned minus random |"
            ),
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for result in report["comparisons"].values():
        lines.append(
            f"| {result['scope']} | {result['mode']} | "
            f"{result['learned_minus_baseline']['accuracy_delta']:+.4f} | "
            f"{result['random_minus_baseline']['accuracy_delta']:+.4f} | "
            f"{result['learned_minus_random']['accuracy_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(report["interpretation_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)
