"""Matched selector-specificity analysis for teacher/student KV prediction.

Each selector is evaluated with the same examples, split assignment, student states,
and within-batch shuffled nulls.  The primary per-group score is therefore

    rank-r held-out R²(actual pairing) - rank-r held-out R²(shuffled pairing).

Comparing that score across selectors asks whether R-KV chooses teacher trace positions
whose information is more predictably aligned with the student than uniformly spaced or
randomly selected positions.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence


SELECTOR_SPECIFICITY_REPORT_SCHEMA_VERSION = 1
KV_KINDS = ("key", "value")


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _group_key(group: Mapping) -> tuple[int, int, int]:
    return (
        int(group["layer"]),
        int(group["head"]),
        int(group["position"]),
    )


def _rank_group_scores(report: Mapping, kind: str, rank: int) -> dict:
    """Return matched position-wise actual, null, and signal R² by group."""
    rank_key = str(rank)
    actual_groups = report["pairings"]["actual"][kind]["position"]["groups"]
    shuffled_groups = {
        _group_key(group): group
        for group in report["pairings"]["shuffled"][kind]["position"]["groups"]
    }
    scores = {}
    for actual in actual_groups:
        key = _group_key(actual)
        shuffled = shuffled_groups.get(key)
        if shuffled is None:
            raise ValueError(f"shuffled report is missing group {key}")
        actual_r2 = actual["ranks"][rank_key]["mean_heldout_r2"]
        shuffled_r2 = shuffled["ranks"][rank_key]["mean_heldout_r2"]
        retention = actual["ranks"][rank_key]["mean_fraction_of_full_r2"]
        if actual_r2 is None or shuffled_r2 is None:
            continue
        actual_r2 = float(actual_r2)
        shuffled_r2 = float(shuffled_r2)
        scores[key] = {
            "actual_r2": actual_r2,
            "shuffled_r2": shuffled_r2,
            "signal_r2": actual_r2 - shuffled_r2,
            "rank_full_retention": (
                None if retention is None else float(retention)
            ),
        }
    return scores


def _paired_comparison(
    left: Mapping[tuple[int, int, int], Mapping],
    right: Mapping[tuple[int, int, int], Mapping],
) -> dict:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(
            "selector reports do not contain identical matched groups; "
            f"missing_left={missing_left[:3]}, missing_right={missing_right[:3]}"
        )
    deltas = [
        float(left[key]["signal_r2"]) - float(right[key]["signal_r2"])
        for key in sorted(left)
    ]
    return {
        "matched_groups": len(deltas),
        "median_signal_r2_delta": _median(deltas),
        "mean_signal_r2_delta": _mean(deltas),
        "fraction_left_above_right": (
            sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None
        ),
    }


def analyze_selector_specificity(
    selector_reports: Mapping[str, Mapping],
    *,
    random_selectors: Sequence[str],
    rank: int = 4,
    signal_margin: float = 0.01,
    required_win_fraction: float = 0.60,
) -> dict:
    """Compare R-KV against uniform and seeded-random trace selectors.

    The gate is preregistered at the KV-kind level.  R-KV must already pass the
    selector's own held-out actual-versus-shuffle gate, beat uniform selection by the
    requested median signal margin and win fraction, and beat the per-group median of
    all random selectors by the same thresholds.
    """
    required = {"rkv", "uniform", *random_selectors}
    missing = sorted(required - set(selector_reports))
    if missing:
        raise ValueError(f"missing selector reports: {missing}")
    if not random_selectors:
        raise ValueError("at least one random selector is required")
    if signal_margin < 0:
        raise ValueError("signal_margin must be non-negative")
    if not 0.0 <= required_win_fraction <= 1.0:
        raise ValueError("required_win_fraction must be in [0, 1]")

    by_kind = {}
    for kind in KV_KINDS:
        group_scores = {
            selector: _rank_group_scores(report, kind, rank)
            for selector, report in selector_reports.items()
            if selector in required
        }
        reference_keys = set(group_scores["rkv"])
        for selector, scores in group_scores.items():
            if set(scores) != reference_keys:
                raise ValueError(
                    f"selector {selector!r} has a different {kind} group set"
                )

        selector_summaries = {}
        for selector, scores in group_scores.items():
            actual = [float(value["actual_r2"]) for value in scores.values()]
            shuffled = [
                float(value["shuffled_r2"]) for value in scores.values()
            ]
            signal = [float(value["signal_r2"]) for value in scores.values()]
            retention = [
                float(value["rank_full_retention"])
                for value in scores.values()
                if value["rank_full_retention"] is not None
            ]
            selector_summaries[selector] = {
                "matched_groups": len(scores),
                "median_actual_r2": _median(actual),
                "median_shuffled_r2": _median(shuffled),
                "median_signal_r2": _median(signal),
                "fraction_actual_above_shuffle": (
                    sum(value > 0 for value in signal) / len(signal)
                    if signal
                    else None
                ),
                "median_rank_full_retention": _median(retention),
            }

        rkv_vs_uniform = _paired_comparison(
            group_scores["rkv"], group_scores["uniform"]
        )
        rkv_vs_random_each = {
            selector: _paired_comparison(
                group_scores["rkv"], group_scores[selector]
            )
            for selector in random_selectors
        }
        random_median_scores = {}
        for key in sorted(reference_keys):
            signal_values = [
                float(group_scores[selector][key]["signal_r2"])
                for selector in random_selectors
            ]
            random_median_scores[key] = {
                "signal_r2": float(statistics.median(signal_values))
            }
        rkv_vs_random_median = _paired_comparison(
            group_scores["rkv"], random_median_scores
        )

        by_position = []
        positions = sorted({key[2] for key in reference_keys})
        for position in positions:
            position_keys = [
                key for key in sorted(reference_keys) if key[2] == position
            ]
            rkv_signal = [
                float(group_scores["rkv"][key]["signal_r2"])
                for key in position_keys
            ]
            uniform_signal = [
                float(group_scores["uniform"][key]["signal_r2"])
                for key in position_keys
            ]
            random_signal = [
                float(
                    statistics.median(
                        [
                            group_scores[selector][key]["signal_r2"]
                            for selector in random_selectors
                        ]
                    )
                )
                for key in position_keys
            ]
            by_position.append(
                {
                    "position": position,
                    "groups": len(position_keys),
                    "rkv_median_signal_r2": _median(rkv_signal),
                    "uniform_median_signal_r2": _median(uniform_signal),
                    "random_median_signal_r2": _median(random_signal),
                    "rkv_minus_uniform_median_delta": _median(
                        [
                            left - right
                            for left, right in zip(rkv_signal, uniform_signal)
                        ]
                    ),
                    "rkv_minus_random_median_delta": _median(
                        [
                            left - right
                            for left, right in zip(rkv_signal, random_signal)
                        ]
                    ),
                }
            )

        rkv_gate = bool(
            selector_reports["rkv"]["gate"]["by_kind"][kind]["supported"]
        )
        supported = bool(
            rkv_gate
            and rkv_vs_uniform["median_signal_r2_delta"] >= signal_margin
            and rkv_vs_uniform["fraction_left_above_right"]
            >= required_win_fraction
            and rkv_vs_random_median["median_signal_r2_delta"]
            >= signal_margin
            and rkv_vs_random_median["fraction_left_above_right"]
            >= required_win_fraction
        )
        by_kind[kind] = {
            "supported": supported,
            "rkv_within_selector_gate_supported": rkv_gate,
            "selectors": selector_summaries,
            "rkv_vs_uniform": rkv_vs_uniform,
            "rkv_vs_random_median": rkv_vs_random_median,
            "rkv_vs_each_random": rkv_vs_random_each,
            "by_position": by_position,
        }

    if all(result["supported"] for result in by_kind.values()):
        status = "rkv_selector_specificity_supported_for_keys_and_values"
    elif any(result["supported"] for result in by_kind.values()):
        status = "rkv_selector_specificity_supported_for_one_kv_kind"
    else:
        status = "rkv_selector_specificity_not_supported"
    return {
        "schema_version": SELECTOR_SPECIFICITY_REPORT_SCHEMA_VERSION,
        "analysis": "matched_teacher_trace_selector_specificity",
        "gate": {
            "status": status,
            "rank": rank,
            "signal_definition": (
                "heldout rank-r R2(actual pairing) minus heldout rank-r "
                "R2(within-selector shuffled pairing)"
            ),
            "signal_margin": signal_margin,
            "required_win_fraction": required_win_fraction,
            "random_aggregation": "per-group median across seeded random selectors",
            "by_kind": by_kind,
            "interpretation": (
                "A positive gate supports the claim that R-KV selects teacher trace "
                "positions with more transferable linear KV signal than matched "
                "uniform and random selectors. It does not establish answer causality "
                "or an accuracy improvement."
            ),
        },
        "random_selectors": list(random_selectors),
        "by_kind": by_kind,
    }


def analyze_candidate_selector_specificity(
    selector_reports: Mapping[str, Mapping],
    *,
    candidate_selector: str,
    structured_controls: Sequence[str],
    random_selectors: Sequence[str],
    rank: int = 4,
    signal_margin: float = 0.01,
    required_win_fraction: float = 0.60,
) -> dict:
    """Test one candidate selector against structured and random controls.

    This is used for confirmatory selector development after the original R-KV
    specificity gate. The candidate must beat every named structured control and the
    per-group random median under the same within-selector-null-adjusted signal score.
    """
    if not structured_controls:
        raise ValueError("at least one structured control is required")
    if candidate_selector in structured_controls:
        raise ValueError("candidate selector cannot also be a structured control")
    required = {
        candidate_selector,
        *structured_controls,
        *random_selectors,
    }
    missing = sorted(required - set(selector_reports))
    if missing:
        raise ValueError(f"missing selector reports: {missing}")
    if not random_selectors:
        raise ValueError("at least one random selector is required")
    if signal_margin < 0:
        raise ValueError("signal_margin must be non-negative")
    if not 0.0 <= required_win_fraction <= 1.0:
        raise ValueError("required_win_fraction must be in [0, 1]")

    by_kind = {}
    for kind in KV_KINDS:
        group_scores = {
            selector: _rank_group_scores(
                selector_reports[selector], kind, rank
            )
            for selector in required
        }
        reference_keys = set(group_scores[candidate_selector])
        for selector, scores in group_scores.items():
            if set(scores) != reference_keys:
                raise ValueError(
                    f"selector {selector!r} has a different {kind} group set"
                )

        selector_summaries = {}
        for selector, scores in group_scores.items():
            actual = [float(value["actual_r2"]) for value in scores.values()]
            shuffled = [
                float(value["shuffled_r2"]) for value in scores.values()
            ]
            signal = [float(value["signal_r2"]) for value in scores.values()]
            retention = [
                float(value["rank_full_retention"])
                for value in scores.values()
                if value["rank_full_retention"] is not None
            ]
            selector_summaries[selector] = {
                "matched_groups": len(scores),
                "median_actual_r2": _median(actual),
                "median_shuffled_r2": _median(shuffled),
                "median_signal_r2": _median(signal),
                "fraction_actual_above_shuffle": (
                    sum(value > 0 for value in signal) / len(signal)
                    if signal
                    else None
                ),
                "median_rank_full_retention": _median(retention),
            }

        candidate_vs_controls = {
            control: _paired_comparison(
                group_scores[candidate_selector], group_scores[control]
            )
            for control in structured_controls
        }
        candidate_vs_each_random = {
            selector: _paired_comparison(
                group_scores[candidate_selector], group_scores[selector]
            )
            for selector in random_selectors
        }
        random_median_scores = {}
        for key in sorted(reference_keys):
            random_median_scores[key] = {
                "signal_r2": float(
                    statistics.median(
                        [
                            group_scores[selector][key]["signal_r2"]
                            for selector in random_selectors
                        ]
                    )
                )
            }
        candidate_vs_random_median = _paired_comparison(
            group_scores[candidate_selector], random_median_scores
        )

        by_position = []
        for position in sorted({key[2] for key in reference_keys}):
            position_keys = [
                key for key in sorted(reference_keys) if key[2] == position
            ]
            candidate_signal = [
                float(group_scores[candidate_selector][key]["signal_r2"])
                for key in position_keys
            ]
            control_signals = {
                control: [
                    float(group_scores[control][key]["signal_r2"])
                    for key in position_keys
                ]
                for control in structured_controls
            }
            random_signal = [
                float(
                    statistics.median(
                        [
                            group_scores[selector][key]["signal_r2"]
                            for selector in random_selectors
                        ]
                    )
                )
                for key in position_keys
            ]
            by_position.append(
                {
                    "position": position,
                    "groups": len(position_keys),
                    "candidate_median_signal_r2": _median(candidate_signal),
                    "control_median_signal_r2": {
                        control: _median(values)
                        for control, values in control_signals.items()
                    },
                    "candidate_minus_control_median_delta": {
                        control: _median(
                            [
                                left - right
                                for left, right in zip(
                                    candidate_signal, values
                                )
                            ]
                        )
                        for control, values in control_signals.items()
                    },
                    "random_median_signal_r2": _median(random_signal),
                    "candidate_minus_random_median_delta": _median(
                        [
                            left - right
                            for left, right in zip(
                                candidate_signal, random_signal
                            )
                        ]
                    ),
                }
            )

        candidate_gate = bool(
            selector_reports[candidate_selector]["gate"]["by_kind"][kind][
                "supported"
            ]
        )

        def comparison_passes(comparison: Mapping) -> bool:
            return bool(
                comparison["median_signal_r2_delta"] >= signal_margin
                and comparison["fraction_left_above_right"]
                >= required_win_fraction
            )

        structured_passes = {
            control: comparison_passes(comparison)
            for control, comparison in candidate_vs_controls.items()
        }
        random_passes = comparison_passes(candidate_vs_random_median)
        supported = bool(
            candidate_gate
            and all(structured_passes.values())
            and random_passes
        )
        by_kind[kind] = {
            "supported": supported,
            "candidate_within_selector_gate_supported": candidate_gate,
            "structured_control_gates": structured_passes,
            "random_median_gate": random_passes,
            "selectors": selector_summaries,
            "candidate_vs_structured_controls": candidate_vs_controls,
            "candidate_vs_random_median": candidate_vs_random_median,
            "candidate_vs_each_random": candidate_vs_each_random,
            "by_position": by_position,
        }

    if all(result["supported"] for result in by_kind.values()):
        suffix = "supported_for_keys_and_values"
    elif any(result["supported"] for result in by_kind.values()):
        suffix = "supported_for_one_kv_kind"
    else:
        suffix = "not_supported"
    status = f"{candidate_selector}_specificity_{suffix}"
    return {
        "schema_version": SELECTOR_SPECIFICITY_REPORT_SCHEMA_VERSION,
        "analysis": "matched_candidate_teacher_trace_selector_specificity",
        "candidate_selector": candidate_selector,
        "structured_controls": list(structured_controls),
        "random_selectors": list(random_selectors),
        "gate": {
            "status": status,
            "rank": rank,
            "signal_definition": (
                "heldout rank-r R2(actual pairing) minus heldout rank-r "
                "R2(within-selector shuffled pairing)"
            ),
            "signal_margin": signal_margin,
            "required_win_fraction": required_win_fraction,
            "random_aggregation": (
                "per-group median across seeded random selectors"
            ),
            "required_comparisons": [
                *structured_controls,
                "random_median",
            ],
            "by_kind": by_kind,
            "interpretation": (
                "A positive gate supports the candidate selector as a stronger "
                "source of transferable linear KV signal than every preregistered "
                "structured control and the random-selector median. It does not "
                "establish answer causality or an accuracy improvement."
            ),
        },
        "by_kind": by_kind,
    }


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def render_selector_specificity_markdown(
    report: Mapping, metadata: Mapping | None = None
) -> str:
    metadata = metadata or {}
    gate = report["gate"]
    selectors = ["rkv", "uniform", *report["random_selectors"]]
    lines = [
        "# Official CODI teacher-trace selector specificity",
        "",
        "## Outcome",
        "",
        f"Predefined gate: **{gate['status'].replace('_', ' ')}**.",
        "",
        (
            "The experiment compares R-KV, uniform, and seeded-random teacher trace "
            "selectors under identical examples, split assignments, student states, "
            "and shuffled-pairing nulls."
        ),
        "",
        "## Calibration contract",
        "",
        f"- Official checkpoint revision: {str(metadata.get('checkpoint_revision', 'unknown'))[:12]}",
        f"- Processed examples: {metadata.get('processed_examples', 'unknown')}",
        f"- Split halves: {metadata.get('num_splits', 'unknown')}",
        f"- Latent positions: {metadata.get('positions', 'unknown')}",
        f"- Gate rank: {gate['rank']}",
        f"- Random selector seeds: {metadata.get('random_selector_seeds', 'unknown')}",
        "",
        "## Held-out position-conditioned prediction",
        "",
        "| Selector | Key actual R² | Key signal R² | Key retention | "
        "Value actual R² | Value signal R² | Value retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector in selectors:
        key = report["by_kind"]["key"]["selectors"][selector]
        value = report["by_kind"]["value"]["selectors"][selector]
        lines.append(
            f"| {selector} | {_format(key['median_actual_r2'])} | "
            f"{_format(key['median_signal_r2'])} | "
            f"{_format(key['median_rank_full_retention'])} | "
            f"{_format(value['median_actual_r2'])} | "
            f"{_format(value['median_signal_r2'])} | "
            f"{_format(value['median_rank_full_retention'])} |"
        )

    lines.extend(
        [
            "",
            "## R-KV specificity gate",
            "",
            "| KV kind | Control | Median signal R² delta | R-KV win fraction | Gate |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for kind in KV_KINDS:
        result = report["by_kind"][kind]
        for label, key in (
            ("uniform", "rkv_vs_uniform"),
            ("random median", "rkv_vs_random_median"),
        ):
            comparison = result[key]
            lines.append(
                f"| {kind} | {label} | "
                f"{_format(comparison['median_signal_r2_delta'])} | "
                f"{_format(comparison['fraction_left_above_right'])} | "
                f"{'supported' if result['supported'] else 'not supported'} |"
            )

    lines.extend(
        [
            "",
            "## Position-resolved R-KV advantage",
            "",
            "| KV kind | Position | R-KV signal R² | R-KV minus uniform | "
            "R-KV minus random median |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for kind in KV_KINDS:
        for row in report["by_kind"][kind]["by_position"]:
            lines.append(
                f"| {kind} | {row['position']} | "
                f"{_format(row['rkv_median_signal_r2'])} | "
                f"{_format(row['rkv_minus_uniform_median_delta'])} | "
                f"{_format(row['rkv_minus_random_median_delta'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            gate["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def render_candidate_selector_markdown(
    report: Mapping, metadata: Mapping | None = None
) -> str:
    metadata = metadata or {}
    gate = report["gate"]
    candidate = report["candidate_selector"]
    controls = list(report["structured_controls"])
    selectors = [
        candidate,
        *controls,
        *report["random_selectors"],
    ]
    lines = [
        "# Official CODI boundary-aware selector confirmation",
        "",
        "## Outcome",
        "",
        f"Predefined gate: **{gate['status'].replace('_', ' ')}**.",
        "",
        (
            "The candidate always retains the first and last valid teacher trace "
            "tokens and uses R-KV for four interior targets. Every arm shares the "
            "same disjoint examples, split assignments, model states, and nulls."
        ),
        "",
        "## Calibration contract",
        "",
        f"- Official checkpoint revision: {str(metadata.get('checkpoint_revision', 'unknown'))[:12]}",
        f"- Processed examples: {metadata.get('processed_examples', 'unknown')}",
        f"- Excluded prior examples: {metadata.get('excluded_indices_count', 'unknown')}",
        f"- Verified overlap: {metadata.get('sample_overlap_with_exclusion', 'unknown')}",
        f"- Split halves: {metadata.get('num_splits', 'unknown')}",
        f"- Gate rank: {gate['rank']}",
        f"- Random selector seeds: {metadata.get('random_selector_seeds', 'unknown')}",
        "",
        "## Held-out position-conditioned prediction",
        "",
        "| Selector | Key actual R² | Key signal R² | Key retention | "
        "Value actual R² | Value signal R² | Value retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for selector in selectors:
        key = report["by_kind"]["key"]["selectors"][selector]
        value = report["by_kind"]["value"]["selectors"][selector]
        lines.append(
            f"| {selector} | {_format(key['median_actual_r2'])} | "
            f"{_format(key['median_signal_r2'])} | "
            f"{_format(key['median_rank_full_retention'])} | "
            f"{_format(value['median_actual_r2'])} | "
            f"{_format(value['median_signal_r2'])} | "
            f"{_format(value['median_rank_full_retention'])} |"
        )

    lines.extend(
        [
            "",
            "## Candidate specificity gate",
            "",
            "| KV kind | Control | Median signal R² delta | "
            "Candidate win fraction | Comparison gate |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for kind in KV_KINDS:
        result = report["by_kind"][kind]
        for control in controls:
            comparison = result["candidate_vs_structured_controls"][control]
            lines.append(
                f"| {kind} | {control} | "
                f"{_format(comparison['median_signal_r2_delta'])} | "
                f"{_format(comparison['fraction_left_above_right'])} | "
                f"{'pass' if result['structured_control_gates'][control] else 'fail'} |"
            )
        comparison = result["candidate_vs_random_median"]
        lines.append(
            f"| {kind} | random median | "
            f"{_format(comparison['median_signal_r2_delta'])} | "
            f"{_format(comparison['fraction_left_above_right'])} | "
            f"{'pass' if result['random_median_gate'] else 'fail'} |"
        )

    lines.extend(
        [
            "",
            "## Position-resolved candidate advantage",
            "",
            (
                "| KV kind | Position | Candidate signal R² | "
                + " | ".join(
                    f"Candidate minus {control}" for control in controls
                )
                + " | Candidate minus random median |"
            ),
            (
                "| --- | ---: | ---: | "
                + " | ".join("---:" for _ in controls)
                + " | ---: |"
            ),
        ]
    )
    for kind in KV_KINDS:
        for row in report["by_kind"][kind]["by_position"]:
            lines.append(
                f"| {kind} | {row['position']} | "
                f"{_format(row['candidate_median_signal_r2'])} | "
                + " | ".join(
                    _format(
                        row["candidate_minus_control_median_delta"][control]
                    )
                    for control in controls
                )
                + " | "
                f"{_format(row['candidate_minus_random_median_delta'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            gate["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)
