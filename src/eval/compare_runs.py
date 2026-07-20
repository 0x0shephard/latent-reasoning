"""Paired comparison utilities for evaluation JSONL artifacts.

Aggregate accuracy alone hides whether two methods solve the same examples.  This
module treats the saved per-example predictions as the source of truth, verifies that
runs are aligned, and reports paired uncertainty and disagreement statistics.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

from src.data.answer_extract import extract_final_number, normalize_number


@dataclass(frozen=True)
class EvalRecord:
    question: str
    gold: str
    generation: str
    correct: bool


@dataclass(frozen=True)
class EvalRun:
    path: Path
    datasets: Mapping[str, tuple[EvalRecord, ...]]


def _read_jsonl(path: Path) -> tuple[EvalRecord, ...]:
    records: list[EvalRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            missing = {"question", "gold", "generation", "correct"} - raw.keys()
            if missing:
                raise ValueError(
                    f"missing fields at {path}:{line_number}: {sorted(missing)}"
                )
            if type(raw["correct"]) is not bool:
                raise ValueError(f"correct must be boolean at {path}:{line_number}")
            records.append(
                EvalRecord(
                    question=str(raw["question"]),
                    gold=str(raw["gold"]),
                    generation=str(raw["generation"]),
                    correct=raw["correct"],
                )
            )
    if not records:
        raise ValueError(f"evaluation file is empty: {path}")
    return tuple(records)


def load_eval_run(path: str | Path) -> EvalRun:
    """Load every dataset JSONL in an ``eval/step_*`` directory."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"evaluation directory does not exist: {root}")
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no evaluation JSONL files under {root}")
    datasets = {file.stem: _read_jsonl(file) for file in files}
    return EvalRun(path=root, datasets=datasets)


def _gold_key(value: str) -> tuple[str, str | Decimal]:
    normalized = normalize_number(value)
    if normalized is None:
        return "text", value.strip()
    # Keep the Decimal itself as the key. Decimal equality and hashing are numeric,
    # so equivalent serialized golds such as "1", "1.0", and "1.00" align while
    # still preserving exact values without a float conversion.
    return "number", normalized


def _record_key(record: EvalRecord) -> tuple[str, tuple[str, str | Decimal]]:
    return record.question, _gold_key(record.gold)


def align_records(
    left: Sequence[EvalRecord], right: Sequence[EvalRecord], dataset: str
) -> tuple[tuple[EvalRecord, ...], tuple[EvalRecord, ...]]:
    if len(left) != len(right):
        raise ValueError(f"{dataset} count mismatch: {len(left)} != {len(right)}")
    left_counts = Counter(_record_key(row) for row in left)
    right_counts = Counter(_record_key(row) for row in right)
    if left_counts != right_counts:
        left_only = sum((left_counts - right_counts).values())
        right_only = sum((right_counts - left_counts).values())
        raise ValueError(
            f"{dataset} example mismatch: left_only={left_only}, right_only={right_only}"
        )

    right_by_key: dict[tuple, deque[EvalRecord]] = defaultdict(deque)
    for row in right:
        right_by_key[_record_key(row)].append(row)
    aligned_right = tuple(right_by_key[_record_key(row)].popleft() for row in left)
    return tuple(left), aligned_right


def validate_alignment(left: EvalRun, right: EvalRun) -> None:
    """Reject paired analysis unless datasets and exact question/gold multisets match."""
    left_names = set(left.datasets)
    right_names = set(right.datasets)
    if left_names != right_names:
        raise ValueError(
            "dataset mismatch: "
            f"left_only={sorted(left_names - right_names)}, "
            f"right_only={sorted(right_names - left_names)}"
        )
    for dataset in sorted(left_names):
        align_records(left.datasets[dataset], right.datasets[dataset], dataset)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _bootstrap_ci(
    deltas: Sequence[int], *, samples: int, rng: random.Random
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    count = len(deltas)
    estimates = [
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _mcnemar_exact_p(left_only: int, right_only: int) -> float:
    """Two-sided exact McNemar p-value using the discordant-pair binomial."""
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _run_dataset_summary(records: Sequence[EvalRecord]) -> dict:
    count = len(records)
    correct = sum(row.correct for row in records)
    blank = sum(not row.generation.strip() for row in records)
    unparseable = sum(extract_final_number(row.generation) is None for row in records)
    return {
        "count": count,
        "correct": correct,
        "accuracy": correct / count,
        "blank_generations": blank,
        "unparseable_generations": unparseable,
    }


def _paired_dataset_summary(
    left: Sequence[EvalRecord],
    right: Sequence[EvalRecord],
    *,
    bootstrap_samples: int,
    rng: random.Random,
) -> dict:
    deltas = [int(r.correct) - int(l.correct) for l, r in zip(left, right)]
    both_correct = sum(l.correct and r.correct for l, r in zip(left, right))
    left_only = sum(l.correct and not r.correct for l, r in zip(left, right))
    right_only = sum(not l.correct and r.correct for l, r in zip(left, right))
    both_wrong = len(left) - both_correct - left_only - right_only
    return {
        "count": len(left),
        "left_accuracy": sum(row.correct for row in left) / len(left),
        "right_accuracy": sum(row.correct for row in right) / len(right),
        "accuracy_delta": sum(deltas) / len(deltas),
        "accuracy_delta_95ci": _bootstrap_ci(
            deltas, samples=bootstrap_samples, rng=rng
        ),
        "both_correct": both_correct,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": _mcnemar_exact_p(left_only, right_only),
    }


def _macro_bootstrap_ci(
    left: EvalRun,
    right: EvalRun,
    *,
    samples: int,
    rng: random.Random,
) -> list[float]:
    estimates: list[float] = []
    names = sorted(left.datasets)
    for _ in range(samples):
        dataset_deltas = []
        for name in names:
            left_rows, right_rows = align_records(
                left.datasets[name], right.datasets[name], name
            )
            paired = list(zip(left_rows, right_rows))
            count = len(paired)
            delta = sum(
                int(paired[index][1].correct) - int(paired[index][0].correct)
                for index in (rng.randrange(count) for _ in range(count))
            ) / count
            dataset_deltas.append(delta)
        estimates.append(sum(dataset_deltas) / len(dataset_deltas))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def compare_runs(
    runs: Mapping[str, EvalRun], *, bootstrap_samples: int = 10_000, seed: int = 0
) -> dict:
    """Build a serializable multi-run report with every pairwise comparison."""
    if len(runs) < 2:
        raise ValueError("at least two evaluation runs are required")
    if len(set(runs)) != len(runs):
        raise ValueError("run names must be unique")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    run_report = {}
    for name, run in runs.items():
        datasets = {
            dataset: _run_dataset_summary(records)
            for dataset, records in sorted(run.datasets.items())
        }
        run_report[name] = {
            "path": str(run.path),
            "datasets": datasets,
            "macro_mean": sum(item["accuracy"] for item in datasets.values())
            / len(datasets),
        }

    comparisons = []
    rng = random.Random(seed)
    for (left_name, left), (right_name, right) in combinations(runs.items(), 2):
        validate_alignment(left, right)
        datasets = {}
        for dataset in sorted(left.datasets):
            left_rows, right_rows = align_records(
                left.datasets[dataset], right.datasets[dataset], dataset
            )
            datasets[dataset] = _paired_dataset_summary(
                left_rows,
                right_rows,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
        macro_delta = (
            run_report[right_name]["macro_mean"] - run_report[left_name]["macro_mean"]
        )
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "delta_definition": "right_minus_left",
                "datasets": datasets,
                "macro_accuracy_delta": macro_delta,
                "macro_accuracy_delta_95ci": _macro_bootstrap_ci(
                    left,
                    right,
                    samples=bootstrap_samples,
                    rng=rng,
                ),
            }
        )
    return {
        "schema_version": 1,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "runs": run_report,
        "comparisons": comparisons,
    }


def render_markdown(report: Mapping) -> str:
    """Render the main results as a compact human-readable report."""
    dataset_names = sorted(
        {dataset for run in report["runs"].values() for dataset in run["datasets"]}
    )
    lines = [
        "# Phase 2 paired evaluation",
        "",
        f"Bootstrap samples: {report['bootstrap_samples']}; seed: {report['seed']}.",
        "",
        "## Accuracy",
        "",
        "| Run | " + " | ".join(dataset_names) + " | Macro |",
        "| --- | " + " | ".join("---:" for _ in dataset_names) + " | ---: |",
    ]
    for name, run in report["runs"].items():
        values = [
            f"{run['datasets'][dataset]['accuracy']:.4f}"
            if dataset in run["datasets"]
            else "—"
            for dataset in dataset_names
        ]
        lines.append(f"| {name} | " + " | ".join(values) + f" | {run['macro_mean']:.4f} |")

    lines.extend(["", "## Paired differences", ""])
    for comparison in report["comparisons"]:
        left = comparison["left"]
        right = comparison["right"]
        low, high = comparison["macro_accuracy_delta_95ci"]
        lines.extend(
            [
                f"### {right} minus {left}",
                "",
                f"Macro delta: {comparison['macro_accuracy_delta']:+.4f} "
                f"(95% paired bootstrap CI {low:+.4f} to {high:+.4f}).",
                "",
                "| Dataset | Delta | 95% CI | Left only | Right only | McNemar p |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for dataset, result in comparison["datasets"].items():
            low, high = result["accuracy_delta_95ci"]
            lines.append(
                f"| {dataset} | {result['accuracy_delta']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | {result['left_only_correct']} | "
                f"{result['right_only_correct']} | {result['mcnemar_exact_p']:.4g} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
