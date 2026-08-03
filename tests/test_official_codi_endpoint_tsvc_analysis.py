from __future__ import annotations

from src.eval.official_codi_endpoint_tsvc_analysis import (
    analyze_endpoint_scope,
    combine_endpoint_scope_reports,
    holm_adjust,
    render_endpoint_tsvc_markdown,
)


def _batch(index: int, scope: str, learned: float = 0.5):
    arms = {}
    losses = {
        "full": 0.55,
        "learned_top77": learned,
        "random_rank77": 0.70,
        "bottom_rank77": 0.72,
        "shuffled_top77": 0.75,
        "complement": 0.68,
    }
    for name, value in losses.items():
        arms[name] = {
            "validation_losses": [value, value + 0.01],
            "gradient_alignment": {
                "cosine": 0.2 if name == "learned_top77" else 0.0,
            },
        }
    return {
        "scope": scope,
        "batch_index": index,
        "validation": {
            "original_losses": [1.0, 1.0],
            "answer_only_losses": [0.8, 0.81],
        },
        "arms": arms,
    }


def test_holm_adjustment_is_monotonic_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02})
    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]


def test_primary_gate_passes_only_all_required_comparisons():
    report = analyze_endpoint_scope(
        [_batch(index, "endpoint_all_layers") for index in range(12)],
        bootstrap_samples=500,
        seed=9,
    )
    assert report["gate"]["supported"] is True
    assert all(
        value["passes"]
        for value in report["learned_top77_comparisons"].values()
    )
    assert report["bootstrap_unit"] == "paired_update_batch"


def test_combined_gate_authorizes_training_only_from_primary_scope():
    primary_negative = analyze_endpoint_scope(
        [_batch(index, "endpoint_all_layers", learned=0.82) for index in range(12)],
        bootstrap_samples=300,
        seed=2,
    )
    secondary_positive = analyze_endpoint_scope(
        [_batch(index, "endpoint_layer11") for index in range(12)],
        bootstrap_samples=300,
        seed=2,
    )
    matched_request = {
        "checkpoint_sha256": "checkpoint",
        "dataset_fingerprint": "dataset",
        "basis_sha256": "basis",
        "basis_request_sha256": "request",
        "rank": 77,
        "update_indices": [1, 2],
        "validation_indices": [3, 4],
        "batch_size": 2,
        "relative_update_norm": 1e-4,
    }
    primary_negative["request"] = matched_request
    secondary_positive["request"] = matched_request
    combined = combine_endpoint_scope_reports(
        primary_negative,
        secondary_positive,
    )
    assert combined["training_authorized"] is False
    assert combined["status"] == "layer11_only_requires_fresh_confirmation"
    markdown = render_endpoint_tsvc_markdown(combined)
    assert "TSV-C-inspired" in markdown
    assert "endpoint_all_layers" in markdown
