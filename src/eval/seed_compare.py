"""Aggregate matched evaluation runs across independent training seeds."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from src.eval.compare_runs import EvalRun, validate_alignment


def _summary(values: Sequence[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty seed collection")
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "values": list(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": None if sample_sd is None else sample_sd / math.sqrt(len(values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def _accuracies(run: EvalRun) -> tuple[dict[str, float], float]:
    datasets = {
        name: sum(row.correct for row in records) / len(records)
        for name, records in sorted(run.datasets.items())
    }
    return datasets, statistics.fmean(datasets.values())


def compare_seeded_runs(runs: Mapping[str, Mapping[int, EvalRun]]) -> dict:
    """Report per-seed metrics and paired method deltas.

    All methods must contain the same seeds and exact question/gold snapshots.  Seed
    variability is summarized with sample standard deviation rather than a question-level
    bootstrap; with three seeds, the individual values are more honest than a fragile CI.
    """
    if len(runs) != 2:
        raise ValueError("seed comparison requires exactly two methods")
    methods = list(runs)
    seed_sets = {method: set(seed_runs) for method, seed_runs in runs.items()}
    seeds = sorted(seed_sets[methods[0]])
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(seed_sets[method] != set(seeds) for method in methods[1:]):
        raise ValueError(f"methods do not have matched seeds: {seed_sets}")
    if len(seeds) < 3:
        raise ValueError("at least three matched seeds are required")

    reference = runs[methods[0]][seeds[0]]
    for method in methods:
        for seed in seeds:
            validate_alignment(reference, runs[method][seed])

    per_seed: dict[str, dict[str, dict]] = {}
    aggregates: dict[str, dict] = {}
    for method in methods:
        per_seed[method] = {}
        dataset_values: dict[str, list[float]] = {}
        macro_values = []
        for seed in seeds:
            datasets, macro = _accuracies(runs[method][seed])
            per_seed[method][str(seed)] = {"datasets": datasets, "macro_mean": macro}
            macro_values.append(macro)
            for dataset, accuracy in datasets.items():
                dataset_values.setdefault(dataset, []).append(accuracy)
        aggregates[method] = {
            "datasets": {
                dataset: _summary(values)
                for dataset, values in sorted(dataset_values.items())
            },
            "macro_mean": _summary(macro_values),
        }

    left, right = methods
    dataset_deltas: dict[str, list[float]] = {}
    macro_deltas = []
    per_seed_deltas = {}
    for seed in seeds:
        left_result = per_seed[left][str(seed)]
        right_result = per_seed[right][str(seed)]
        deltas = {
            dataset: right_result["datasets"][dataset]
            - left_result["datasets"][dataset]
            for dataset in left_result["datasets"]
        }
        macro_delta = right_result["macro_mean"] - left_result["macro_mean"]
        per_seed_deltas[str(seed)] = {"datasets": deltas, "macro_mean": macro_delta}
        macro_deltas.append(macro_delta)
        for dataset, delta in deltas.items():
            dataset_deltas.setdefault(dataset, []).append(delta)

    return {
        "schema_version": 1,
        "methods": methods,
        "seeds": seeds,
        "delta_definition": f"{right}_minus_{left}",
        "per_seed": per_seed,
        "aggregates": aggregates,
        "paired_seed_deltas": {
            "per_seed": per_seed_deltas,
            "datasets": {
                dataset: _summary(values)
                for dataset, values in sorted(dataset_deltas.items())
            },
            "macro_mean": _summary(macro_deltas),
        },
        "uncertainty_note": (
            "Sample SD and individual seed values describe training-seed variation. "
            "With only three seeds, no asymptotic method-level confidence interval is claimed."
        ),
    }


def render_seed_markdown(report: Mapping) -> str:
    methods = report["methods"]
    seeds = report["seeds"]
    datasets = sorted(report["aggregates"][methods[0]]["datasets"])
    lines = [
        "# Multi-seed CODI vs KaVa comparison",
        "",
        f"Matched seeds: {', '.join(map(str, seeds))}.",
        "",
        "## Per-seed macro accuracy",
        "",
        "| Seed | " + " | ".join(methods) + " | Delta |",
        "| ---: | " + " | ".join("---:" for _ in methods) + " | ---: |",
    ]
    for seed in seeds:
        values = [report["per_seed"][method][str(seed)]["macro_mean"] for method in methods]
        delta = report["paired_seed_deltas"]["per_seed"][str(seed)]["macro_mean"]
        lines.append(
            f"| {seed} | " + " | ".join(f"{value:.4f}" for value in values) + f" | {delta:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Across-seed summary",
            "",
            "| Metric | " + " | ".join(methods) + " | Paired delta |",
            "| --- | " + " | ".join("---:" for _ in methods) + " | ---: |",
        ]
    )
    for metric in datasets + ["Macro"]:
        key = "macro_mean" if metric == "Macro" else "datasets"
        method_cells = []
        for method in methods:
            item = (
                report["aggregates"][method]["macro_mean"]
                if metric == "Macro"
                else report["aggregates"][method]["datasets"][metric]
            )
            method_cells.append(f"{item['mean']:.4f} ± {item['sample_sd']:.4f}")
        delta_item = (
            report["paired_seed_deltas"]["macro_mean"]
            if metric == "Macro"
            else report["paired_seed_deltas"]["datasets"][metric]
        )
        lines.append(
            f"| {metric} | " + " | ".join(method_cells) + f" | {delta_item['mean']:+.4f} ± {delta_item['sample_sd']:.4f} |"
        )
    lines.extend(["", report["uncertainty_note"], ""])
    return "\n".join(lines)
