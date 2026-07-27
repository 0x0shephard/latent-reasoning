"""Analyze the preregistered KV-compression risk pilot and apply its gates."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.kv_risk_pilot import (
    adjacent_containment,
    atomic_json,
    failure_set,
    load_records,
    retention_from_condition,
)
from src.utils.config import load_config


PRIMARY_COMPRESSED = (
    "retain_0.90",
    "retain_0.50",
    "retain_0.25",
    "retain_0.10",
)
MIDDLE_BUDGETS = ("retain_0.50", "retain_0.25")


def _record_map(records: list[dict]) -> dict[str, dict]:
    mapped = {str(record["example_id"]): record for record in records}
    if len(mapped) != len(records):
        raise ValueError("duplicate example ids")
    return mapped


def _paired(full: list[dict], other: list[dict]) -> list[tuple[dict, dict]]:
    left = _record_map(full)
    right = _record_map(other)
    if left.keys() != right.keys():
        raise ValueError("condition example identities differ")
    return [(left[key], right[key]) for key in sorted(left)]


def _bootstrap_binary_ci(
    values: list[bool | int | float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = generator.integers(0, len(array), size=len(array))
        estimates[index] = float(array[sampled].mean())
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def _condition_comparison(
    full: list[dict],
    compressed: list[dict],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    pairs = _paired(full, compressed)
    full_correct = [pair for pair in pairs if bool(pair[0]["correct"])]
    full_incorrect = [pair for pair in pairs if not bool(pair[0]["correct"])]
    harmful = [not bool(right["correct"]) for _, right in full_correct]
    beneficial = [bool(right["correct"]) for _, right in full_incorrect]
    answer_disagreement = [
        left.get("prediction") != right.get("prediction")
        for left, right in pairs
    ]
    correctness_disagreement = [
        bool(left["correct"]) != bool(right["correct"])
        for left, right in pairs
    ]
    retained = sum(
        int(right["retained_total_token_steps"]) for _, right in pairs
    )
    full_cost = sum(
        int(left["retained_total_token_steps"]) for left, _ in pairs
    )
    return {
        "examples": len(pairs),
        "full_accuracy": float(np.mean([left["correct"] for left, _ in pairs])),
        "compressed_accuracy": float(
            np.mean([right["correct"] for _, right in pairs])
        ),
        "correct_to_incorrect": int(sum(harmful)),
        "correct_to_incorrect_rate_among_full_correct": (
            None if not harmful else float(np.mean(harmful))
        ),
        "correct_to_incorrect_rate_ci": _bootstrap_binary_ci(
            harmful,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "incorrect_to_correct": int(sum(beneficial)),
        "incorrect_to_correct_rate_among_full_incorrect": (
            None if not beneficial else float(np.mean(beneficial))
        ),
        "answer_disagreement_rate": float(np.mean(answer_disagreement)),
        "correctness_disagreement_rate": float(
            np.mean(correctness_disagreement)
        ),
        "requested_retention": retention_from_condition(
            str(compressed[0]["condition"])
        ),
        "median_realized_generated_retention": float(
            np.median(
                [
                    right["realized_generated_retention"]
                    for _, right in pairs
                ]
            )
        ),
        "paired_kv_token_step_saving": (
            None
            if full_cost <= 0
            else float((full_cost - retained) / full_cost)
        ),
    }


def _full_repeat(full: list[dict], repeat: list[dict]) -> dict:
    pairs = _paired(full, repeat)
    answer = [
        left.get("prediction") != right.get("prediction")
        for left, right in pairs
    ]
    text = [
        str(left.get("generation")) != str(right.get("generation"))
        for left, right in pairs
    ]
    correctness = [
        bool(left["correct"]) != bool(right["correct"])
        for left, right in pairs
    ]
    return {
        "examples": len(pairs),
        "exact_generation_disagreements": int(sum(text)),
        "answer_disagreements": int(sum(answer)),
        "correctness_disagreements": int(sum(correctness)),
        "answer_disagreement_rate": float(np.mean(answer)),
        "correctness_disagreement_rate": float(np.mean(correctness)),
    }


def _cross_validated_auc(
    full: list[dict],
    compressed: list[dict],
) -> dict:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pairs = [
        pair for pair in _paired(full, compressed) if bool(pair[0]["correct"])
    ]
    labels = np.asarray(
        [not bool(right["correct"]) for _, right in pairs],
        dtype=int,
    )
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives < 2 or negatives < 2:
        return {
            "status": "inconclusive_single_class_or_too_few_failures",
            "examples": len(labels),
            "positives": positives,
            "difficulty_only_auc": None,
            "difficulty_plus_uncertainty_auc": None,
        }

    # Level is optional. Missing values are imputed to the median; if every
    # value is missing, omit the column rather than manufacture a signal.
    use_level = any(left.get("level") is not None for left, _ in pairs)
    difficulty_rows: list[list[float]] = []
    combined_rows: list[list[float]] = []
    for left, _ in pairs:
        difficulty_row = [float(left["question_tokens"])]
        if use_level:
            difficulty_row.append(
                np.nan if left.get("level") is None else float(left["level"])
            )
        difficulty_rows.append(difficulty_row)
        combined_rows.append(
            [
                *difficulty_row,
                float(left.get("early_entropy_mean") or 0.0),
            ]
        )
    folds = min(5, positives, negatives)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

    def cross_validated(features: np.ndarray) -> float:
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=0,
                    ),
                ),
            ]
        )
        probabilities = cross_val_predict(
            pipeline,
            features,
            labels,
            cv=splitter,
            method="predict_proba",
        )[:, 1]
        return float(roc_auc_score(labels, probabilities))

    difficulty_auc = cross_validated(np.asarray(difficulty_rows, dtype=float))
    combined_auc = cross_validated(np.asarray(combined_rows, dtype=float))

    trace_lengths = np.asarray(
        [float(left["generated_tokens"]) for left, _ in pairs],
        dtype=float,
    )
    trace_auc = float(roc_auc_score(labels, trace_lengths))
    trace_auc = max(trace_auc, 1.0 - trace_auc)
    return {
        "status": "ok",
        "examples": len(labels),
        "positives": positives,
        "difficulty_features": [
            "question_tokens",
            *(["dataset_level"] if use_level else []),
        ],
        "uncertainty_feature": "early_entropy_mean",
        "folds": folds,
        "difficulty_only_auc": difficulty_auc,
        "difficulty_plus_uncertainty_auc": combined_auc,
        "posthoc_full_trace_length_auc": trace_auc,
        "note": (
            "Full trace length is diagnostic only and is not available to a "
            "pre-generation router."
        ),
    }


def _cache_size_risk(
    full: list[dict],
    compressed: list[dict],
) -> dict:
    """Measure whether larger uncompressed traces are more failure-prone.

    The point-biserial correlation is simply Pearson correlation where the
    outcome is binary.  It is descriptive rather than a preregistered gate;
    the oracle calculation below is the direct test of economic headroom.
    """
    pairs = [
        pair for pair in _paired(full, compressed) if bool(pair[0]["correct"])
    ]
    labels = np.asarray(
        [not bool(right["correct"]) for _, right in pairs],
        dtype=float,
    )
    if len(labels) < 3 or np.unique(labels).size < 2:
        return {
            "status": "inconclusive_single_class_or_too_few_failures",
            "examples": len(labels),
            "failures": int(labels.sum()),
            "generated_tokens_point_biserial": None,
            "full_kv_token_steps_point_biserial": None,
        }

    def correlation(values: list[float]) -> float | None:
        array = np.asarray(values, dtype=float)
        if np.std(array) == 0.0:
            return None
        return float(np.corrcoef(array, labels)[0, 1])

    return {
        "status": "ok",
        "examples": len(labels),
        "failures": int(labels.sum()),
        "generated_tokens_point_biserial": correlation(
            [float(left["generated_tokens"]) for left, _ in pairs]
        ),
        "full_kv_token_steps_point_biserial": correlation(
            [
                float(left["retained_total_token_steps"])
                for left, _ in pairs
            ]
        ),
        "interpretation": (
            "Positive values mean larger full-cache traces fail more often. "
            "This is descriptive and is not a go/no-go gate."
        ),
    }


def _oracle_saving(
    full: list[dict],
    compressed: list[dict],
    *,
    maximum_loss: float,
) -> dict:
    pairs = _paired(full, compressed)
    full_total = sum(
        int(left["retained_total_token_steps"]) for left, _ in pairs
    )
    allowed_harm = math.floor(maximum_loss * len(pairs) + 1e-12)
    candidates: list[tuple[bool, int, str]] = []
    for left, right in pairs:
        saving = int(left["retained_total_token_steps"]) - int(
            right["retained_total_token_steps"]
        )
        if saving <= 0:
            continue
        harmful = bool(left["correct"]) and not bool(right["correct"])
        candidates.append((harmful, saving, str(left["example_id"])))
    non_harmful = [value for value in candidates if not value[0]]
    harmful = sorted(
        (value for value in candidates if value[0]),
        key=lambda value: (-value[1], value[2]),
    )
    selected = non_harmful + harmful[:allowed_harm]
    saving = sum(value[1] for value in selected)
    return {
        "maximum_allowed_harmful_examples": allowed_harm,
        "compressed_examples": len(selected),
        "harmful_compressed_examples": sum(value[0] for value in selected),
        "kv_token_step_saving": (
            None if full_total <= 0 else float(saving / full_total)
        ),
    }


def _normalized_prediction(record: dict) -> str:
    prediction = record.get("prediction")
    if prediction is not None:
        return str(prediction)
    return str(record.get("generation", "")).strip()


def _stochastic_analysis(root: Path, cfg) -> dict:
    seeds = [int(value) for value in cfg.stochastic.seeds]
    retention = float(cfg.stochastic.retention)
    compressed_base = f"retain_{retention:.2f}"
    full_by_seed: dict[int, list[dict]] = {}
    compressed_by_seed: dict[int, list[dict]] = {}
    for seed in seeds:
        full_dir = root / "conditions" / f"full_seed{seed}"
        compressed_dir = (
            root / "conditions" / f"{compressed_base}_seed{seed}"
        )
        if not full_dir.is_dir() or not compressed_dir.is_dir():
            return {
                "status": "incomplete",
                "missing_seed": seed,
            }
        full_by_seed[seed] = load_records(full_dir)
        compressed_by_seed[seed] = load_records(compressed_dir)

    within_seed_answer: list[bool] = []
    within_seed_correctness: list[bool] = []
    for seed in seeds:
        for full, compressed in _paired(
            full_by_seed[seed],
            compressed_by_seed[seed],
        ):
            within_seed_answer.append(
                _normalized_prediction(full) != _normalized_prediction(compressed)
            )
            within_seed_correctness.append(
                bool(full["correct"]) != bool(compressed["correct"])
            )

    full_pair_answer: list[bool] = []
    full_pair_correctness: list[bool] = []
    for left_seed, right_seed in combinations(seeds, 2):
        for left, right in _paired(
            full_by_seed[left_seed],
            full_by_seed[right_seed],
        ):
            full_pair_answer.append(
                _normalized_prediction(left) != _normalized_prediction(right)
            )
            full_pair_correctness.append(
                bool(left["correct"]) != bool(right["correct"])
            )
    noise = float(np.mean(full_pair_answer))
    compression = float(np.mean(within_seed_answer))
    ratio = (
        math.inf
        if noise == 0.0 and compression > 0.0
        else (0.0 if noise == 0.0 else compression / noise)
    )
    return {
        "status": "complete",
        "seeds": seeds,
        "examples_per_seed": len(full_by_seed[seeds[0]]),
        "full_seed_pair_answer_disagreement_rate": noise,
        "full_seed_pair_correctness_disagreement_rate": float(
            np.mean(full_pair_correctness)
        ),
        "within_seed_compression_answer_disagreement_rate": compression,
        "within_seed_compression_correctness_disagreement_rate": float(
            np.mean(within_seed_correctness)
        ),
        "compression_to_noise_answer_disagreement_ratio": (
            "infinity" if math.isinf(ratio) else ratio
        ),
        "ratio_numeric": ratio,
    }


def _make_figure(
    output_path: Path,
    comparisons: dict[str, dict],
    containment: dict,
) -> None:
    import matplotlib.pyplot as plt

    names = list(PRIMARY_COMPRESSED)
    retained = [
        comparisons[name]["requested_retention"] * 100 for name in names
    ]
    failures = [
        comparisons[name]["correct_to_incorrect_rate_among_full_correct"] * 100
        for name in names
    ]
    containments = [
        value["containment"]
        for value in containment["comparisons"]
        if value["containment"] is not None
    ]
    labels = [
        f"{value['looser'].replace('retain_', '')}→"
        f"{value['tighter'].replace('retain_', '')}"
        for value in containment["comparisons"]
        if value["containment"] is not None
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(retained, failures, marker="o", color="#222222")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Requested generated-cache retention (%)")
    axes[0].set_ylabel("Full-correct answers that fail (%)")
    axes[0].set_title("Compression-induced failure")
    axes[0].grid(alpha=0.25)
    axes[1].bar(labels, [value * 100 for value in containments], color="#555555")
    axes[1].axhline(70, color="#a00000", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Failure-set containment (%)")
    axes[1].set_title("Cross-budget nesting")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _markdown(report: dict) -> str:
    lines = [
        "# KV-compression risk pilot",
        "",
        "## Outcome",
        "",
        f"Predefined decision: **{report['decision']['status'].replace('_', ' ')}**.",
        "",
        "This pilot asks whether compression failure is stable enough to justify "
        "building a selective risk predictor.",
        "",
        "## Dataset and deterministic control",
        "",
        f"- Selected dataset: `{report['selected_dataset']}`",
        f"- Examples: {report['examples']}",
        f"- Full-cache accuracy: {_fmt(report['full_accuracy'])}",
        f"- Full-repeat answer disagreements: "
        f"{report['full_repeat']['answer_disagreements']}",
        "",
        "## Budget sweep",
        "",
        "| Condition | Accuracy | Correct→incorrect | Failure rate | 95% CI | "
        "Incorrect→correct | Realized retention | KV saving |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in PRIMARY_COMPRESSED:
        value = report["comparisons"][name]
        lines.append(
            f"| {name} | {_fmt(value['compressed_accuracy'])} | "
            f"{value['correct_to_incorrect']} | "
            f"{_fmt(value['correct_to_incorrect_rate_among_full_correct'])} | "
            f"{_fmt(value['correct_to_incorrect_rate_ci'][0])}–"
            f"{_fmt(value['correct_to_incorrect_rate_ci'][1])} | "
            f"{value['incorrect_to_correct']} | "
            f"{_fmt(value['median_realized_generated_retention'])} | "
            f"{_fmt(value['paired_kv_token_step_saving'])} |"
        )
    lines.extend(
        [
            "",
            "Failure rate is conditional on the full-cache answer being correct.",
            "",
            "## Cross-budget nesting",
            "",
            "| Looser | Tighter | Failures | Intersection | Containment | Reversal |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for value in report["nesting"]["comparisons"]:
        lines.append(
            f"| {value['looser']} | {value['tighter']} | "
            f"{value['looser_failures']} | {value['intersection']} | "
            f"{_fmt(value['containment'])} | {_fmt(value['reversal_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Predictability and noise",
            "",
            f"- Chosen middle budget: `{report['actionable_budget']}`",
            f"- Difficulty-only AUROC: "
            f"{_fmt(report['difficulty_baseline'].get('difficulty_only_auc'))}",
            f"- Difficulty-plus-uncertainty AUROC: "
            f"{_fmt(report['difficulty_baseline'].get('difficulty_plus_uncertainty_auc'))}",
            f"- Full KV size versus failure correlation: "
            f"{_fmt(report['cache_size_risk'].get('full_kv_token_steps_point_biserial'))}",
            f"- Mean adjacent containment: "
            f"{_fmt(report['nesting']['mean_adjacent_containment'])}",
            f"- Compression/noise disagreement ratio: "
            f"{_fmt(report['stochastic'].get('compression_to_noise_answer_disagreement_ratio'))}",
            f"- Best oracle KV saving: "
            f"{_fmt(report['oracle']['best_kv_token_step_saving'])}",
            "",
            "## Gates",
            "",
            "| Gate | Status |",
            "| --- | --- |",
        ]
    )
    for name, value in report["decision"]["gates"].items():
        lines.append(f"| {name} | {value['status'].replace('_', ' ')} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A positive pilot supports developing a risk predictor. It does not "
            "establish cross-domain calibration, batched-serving savings, or the "
            "usefulness of any particular online feature.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    pilot_root = args.statistics / "pilot"
    stochastic_root = args.statistics / "stochastic"
    manifest = json.loads(
        (pilot_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("state") != "complete":
        raise RuntimeError("primary pilot is not complete")
    conditions_root = pilot_root / "conditions"
    full = load_records(conditions_root / "full")
    repeat = load_records(conditions_root / "full_repeat")
    if not full:
        raise RuntimeError("primary pilot has no records")

    bootstrap_samples = int(cfg.analysis.bootstrap_samples)
    bootstrap_seed = int(cfg.analysis.bootstrap_seed)
    comparisons = {
        name: _condition_comparison(
            full,
            load_records(conditions_root / name),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, name in enumerate(PRIMARY_COMPRESSED)
    }
    repeat_analysis = _full_repeat(full, repeat)
    failures = [
        (
            name,
            failure_set(full, load_records(conditions_root / name)),
        )
        for name in PRIMARY_COMPRESSED
    ]
    nesting = adjacent_containment(failures)

    minimum = float(cfg.gates.actionable_failure_rate_min)
    maximum = float(cfg.gates.actionable_failure_rate_max)
    actionable = [
        name
        for name in MIDDLE_BUDGETS
        if comparisons[name]["correct_to_incorrect_rate_among_full_correct"]
        is not None
        and minimum
        <= comparisons[name][
            "correct_to_incorrect_rate_among_full_correct"
        ]
        <= maximum
    ]
    actionable_budget = actionable[0] if actionable else None
    difficulty = (
        {
            "status": "not_run_no_actionable_middle_budget",
            "difficulty_only_auc": None,
            "difficulty_plus_uncertainty_auc": None,
        }
        if actionable_budget is None
        else _cross_validated_auc(
            full,
            load_records(conditions_root / actionable_budget),
        )
    )
    cache_size_risk = (
        {
            "status": "not_run_no_actionable_middle_budget",
            "full_kv_token_steps_point_biserial": None,
            "generated_tokens_point_biserial": None,
        }
        if actionable_budget is None
        else _cache_size_risk(
            full,
            load_records(conditions_root / actionable_budget),
        )
    )

    maximum_loss = float(cfg.gates.maximum_induced_accuracy_loss)
    oracle_by_budget = {
        name: _oracle_saving(
            full,
            load_records(conditions_root / name),
            maximum_loss=maximum_loss,
        )
        for name in PRIMARY_COMPRESSED
    }
    valid_oracle = [
        (name, value["kv_token_step_saving"])
        for name, value in oracle_by_budget.items()
        if value["kv_token_step_saving"] is not None
    ]
    best_budget, best_saving = max(
        valid_oracle,
        key=lambda value: value[1],
        default=(None, None),
    )
    stochastic = _stochastic_analysis(stochastic_root, cfg)

    gates: dict[str, dict] = {}
    gates["deterministic_repeat"] = {
        "status": (
            "passed"
            if repeat_analysis["answer_disagreements"] == 0
            else "failed"
        ),
        "observed": repeat_analysis["answer_disagreement_rate"],
        "required": 0.0,
    }
    gates["actionable_failure_band"] = {
        "status": "passed" if actionable_budget else "failed",
        "observed_budget": actionable_budget,
        "required": [minimum, maximum],
    }
    mean_containment = nesting["mean_adjacent_containment"]
    gates["nested_failures"] = {
        "status": (
            "inconclusive"
            if mean_containment is None
            else (
                "passed"
                if mean_containment
                >= float(cfg.gates.minimum_adjacent_containment)
                else "failed"
            )
        ),
        "observed": mean_containment,
        "required": float(cfg.gates.minimum_adjacent_containment),
    }
    auc = difficulty.get("difficulty_only_auc")
    gates["difficulty_leaves_headroom"] = {
        "status": (
            "inconclusive"
            if auc is None
            else (
                "passed"
                if auc < float(cfg.gates.maximum_difficulty_only_auc)
                else "failed"
            )
        ),
        "observed": auc,
        "required_less_than": float(
            cfg.gates.maximum_difficulty_only_auc
        ),
    }
    ratio = stochastic.get("ratio_numeric")
    gates["compression_exceeds_sampling_noise"] = {
        "status": (
            "inconclusive"
            if ratio is None
            else (
                "passed"
                if ratio >= float(cfg.gates.stochastic_noise_multiplier)
                else "failed"
            )
        ),
        "observed": (
            None if ratio is None else ("infinity" if math.isinf(ratio) else ratio)
        ),
        "required": float(cfg.gates.stochastic_noise_multiplier),
    }
    gates["oracle_memory_value"] = {
        "status": (
            "inconclusive"
            if best_saving is None
            else (
                "passed"
                if best_saving >= float(cfg.gates.minimum_oracle_kv_saving)
                else "failed"
            )
        ),
        "observed": best_saving,
        "required": float(cfg.gates.minimum_oracle_kv_saving),
    }
    primary_names = (
        "deterministic_repeat",
        "actionable_failure_band",
        "nested_failures",
        "difficulty_leaves_headroom",
        "compression_exceeds_sampling_noise",
        "oracle_memory_value",
    )
    status = (
        "go_to_risk_predictor_development"
        if all(gates[name]["status"] == "passed" for name in primary_names)
        else "pivot_or_stop"
    )
    report = {
        "schema_version": 1,
        "analysis": "kv_compression_risk_pilot",
        "selected_dataset": manifest["selected_dataset"],
        "examples": len(full),
        "full_accuracy": float(np.mean([record["correct"] for record in full])),
        "full_repeat": repeat_analysis,
        "comparisons": comparisons,
        "nesting": nesting,
        "actionable_budget": actionable_budget,
        "difficulty_baseline": difficulty,
        "cache_size_risk": cache_size_risk,
        "stochastic": {
            key: value
            for key, value in stochastic.items()
            if key != "ratio_numeric"
        },
        "oracle": {
            "maximum_induced_accuracy_loss": maximum_loss,
            "by_budget": oracle_by_budget,
            "best_budget": best_budget,
            "best_kv_token_step_saving": best_saving,
        },
        "decision": {
            "status": status,
            "gates": gates,
            "rule": "all preregistered primary gates must pass",
        },
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = analyze(args)
    atomic_json(args.output, report)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    figure_path = args.output.with_suffix(".png")
    _make_figure(
        figure_path,
        report["comparisons"],
        report["nesting"],
    )
    print(f"[analysis] decision={report['decision']['status']}")
    print(f"[analysis] wrote {args.output}")
    print(f"[analysis] wrote {markdown_path}")
    print(f"[analysis] wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
