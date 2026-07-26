"""Regression tests for the matched teacher-trace selector gate."""
from __future__ import annotations

import pytest

from scripts.collect_official_codi_selector_subspaces import (
    _load_exclusion_manifest,
    _sample_eligible_indices_excluding,
)
from src.mech.kv_selector_specificity import (
    analyze_candidate_selector_specificity,
    analyze_selector_specificity,
)


def _report(
    actual_by_kind: dict[str, list[float]],
    shuffled_by_kind: dict[str, list[float]],
    *,
    gate_supported: bool = True,
) -> dict:
    pairings = {"actual": {}, "shuffled": {}}
    for pairing, values_by_kind in (
        ("actual", actual_by_kind),
        ("shuffled", shuffled_by_kind),
    ):
        for kind, values in values_by_kind.items():
            groups = []
            for index, value in enumerate(values):
                groups.append(
                    {
                        "layer": 0,
                        "head": index,
                        "position": index % 2,
                        "ranks": {
                            "4": {
                                "mean_heldout_r2": value,
                                "mean_fraction_of_full_r2": 0.9,
                            }
                        },
                    }
                )
            pairings[pairing][kind] = {"position": {"groups": groups}}
    return {
        "pairings": pairings,
        "gate": {
            "by_kind": {
                kind: {"supported": gate_supported}
                for kind in actual_by_kind
            }
        },
    }


def _two_kind(values: list[float]) -> dict[str, list[float]]:
    return {"key": values, "value": values}


def test_selector_gate_supports_clear_rkv_advantage():
    reports = {
        "rkv": _report(
            _two_kind([0.30, 0.31, 0.32, 0.33]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "uniform": _report(
            _two_kind([0.18, 0.19, 0.20, 0.21]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "random_seed1": _report(
            _two_kind([0.16, 0.17, 0.18, 0.19]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "random_seed2": _report(
            _two_kind([0.17, 0.18, 0.19, 0.20]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
    }
    result = analyze_selector_specificity(
        reports,
        random_selectors=("random_seed1", "random_seed2"),
    )
    assert (
        result["gate"]["status"]
        == "rkv_selector_specificity_supported_for_keys_and_values"
    )
    assert result["by_kind"]["key"]["supported"]
    assert (
        result["by_kind"]["key"]["rkv_vs_uniform"][
            "fraction_left_above_right"
        ]
        == 1.0
    )


def test_selector_gate_rejects_selector_independent_signal():
    common_actual = _two_kind([0.20, 0.21, 0.22, 0.23])
    common_null = _two_kind([0.10, 0.10, 0.10, 0.10])
    reports = {
        name: _report(common_actual, common_null)
        for name in ("rkv", "uniform", "random_seed1", "random_seed2")
    }
    result = analyze_selector_specificity(
        reports,
        random_selectors=("random_seed1", "random_seed2"),
    )
    assert (
        result["gate"]["status"]
        == "rkv_selector_specificity_not_supported"
    )
    assert not result["by_kind"]["value"]["supported"]


def test_selector_analysis_requires_identical_matched_groups():
    reports = {
        "rkv": _report(_two_kind([0.3, 0.3]), _two_kind([0.1, 0.1])),
        "uniform": _report(_two_kind([0.2]), _two_kind([0.1])),
        "random_seed1": _report(
            _two_kind([0.2, 0.2]), _two_kind([0.1, 0.1])
        ),
    }
    with pytest.raises(ValueError, match="different key group set"):
        analyze_selector_specificity(
            reports,
            random_selectors=("random_seed1",),
        )


def test_candidate_gate_requires_wins_over_every_control():
    reports = {
        "boundary_rkv": _report(
            _two_kind([0.34, 0.35, 0.36, 0.37]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "rkv": _report(
            _two_kind([0.23, 0.24, 0.25, 0.26]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "uniform": _report(
            _two_kind([0.22, 0.23, 0.24, 0.25]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "random_seed1": _report(
            _two_kind([0.17, 0.18, 0.19, 0.20]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "random_seed2": _report(
            _two_kind([0.18, 0.19, 0.20, 0.21]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
    }
    result = analyze_candidate_selector_specificity(
        reports,
        candidate_selector="boundary_rkv",
        structured_controls=("rkv", "uniform"),
        random_selectors=("random_seed1", "random_seed2"),
    )
    assert (
        result["gate"]["status"]
        == "boundary_rkv_specificity_supported_for_keys_and_values"
    )
    assert all(
        result["by_kind"]["key"]["structured_control_gates"].values()
    )


def test_candidate_gate_fails_when_uniform_matches_candidate():
    reports = {
        "boundary_rkv": _report(
            _two_kind([0.30, 0.31, 0.32, 0.33]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "rkv": _report(
            _two_kind([0.20, 0.21, 0.22, 0.23]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "uniform": _report(
            _two_kind([0.30, 0.31, 0.32, 0.33]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
        "random_seed1": _report(
            _two_kind([0.17, 0.18, 0.19, 0.20]),
            _two_kind([0.10, 0.10, 0.10, 0.10]),
        ),
    }
    result = analyze_candidate_selector_specificity(
        reports,
        candidate_selector="boundary_rkv",
        structured_controls=("rkv", "uniform"),
        random_selectors=("random_seed1",),
    )
    assert result["gate"]["status"] == "boundary_rkv_specificity_not_supported"
    assert not result["by_kind"]["value"]["structured_control_gates"]["uniform"]


def test_disjoint_sampler_excludes_every_prior_index():
    dataset = {"answer": [str(index) for index in range(20)]}
    excluded = {0, 2, 4, 6, 8}
    selected, eligible = _sample_eligible_indices_excluding(
        dataset,
        examples=10,
        seed=2,
        excluded=excluded,
    )
    assert eligible == 20
    assert len(selected) == 10
    assert not set(selected) & excluded
    assert selected == _sample_eligible_indices_excluding(
        dataset,
        examples=10,
        seed=2,
        excluded=excluded,
    )[0]


def test_exclusion_manifest_is_checkpoint_bound_and_duplicate_safe(tmp_path):
    manifest = tmp_path / "collection_manifest.json"
    manifest.write_text(
        (
            '{"state":"complete","checkpoint_revision":"revision-a",'
            '"sample_indices":[3,1,4],"train_dataset_fingerprint":"data-a"}'
        ),
        encoding="utf-8",
    )
    excluded, metadata = _load_exclusion_manifest(
        manifest,
        checkpoint_revision="revision-a",
    )
    assert excluded == {1, 3, 4}
    assert metadata["count"] == 3
    with pytest.raises(RuntimeError, match="different checkpoint"):
        _load_exclusion_manifest(
            manifest,
            checkpoint_revision="revision-b",
        )

    manifest.write_text(
        (
            '{"state":"complete","checkpoint_revision":"revision-a",'
            '"sample_indices":[3,3]}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        _load_exclusion_manifest(
            manifest,
            checkpoint_revision="revision-a",
        )
