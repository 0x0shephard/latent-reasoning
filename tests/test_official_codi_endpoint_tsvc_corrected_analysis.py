from __future__ import annotations

from src.eval.official_codi_endpoint_tsvc_analysis import analyze_endpoint_scope
from src.eval.official_codi_endpoint_tsvc_corrected_analysis import (
    combine_corrected_endpoint_reports,
    render_corrected_endpoint_markdown,
)


def _batch(index: int, scope: str, learned: float = 0.5):
    losses = {
        "full": 0.55,
        "learned_top77": learned,
        "random_rank77": 0.70,
        "bottom_rank77": 0.72,
        "shuffled_top77": 0.75,
        "complement": 0.68,
    }
    return {
        "scope": scope,
        "batch_index": index,
        "validation": {
            "original_losses": [1.0, 1.0],
            "answer_only_losses": [0.8, 0.81],
        },
        "arms": {
            name: {
                "validation_losses": [value, value + 0.01],
                "gradient_alignment": {
                    "cosine": 0.2 if name == "learned_top77" else 0.0,
                },
            }
            for name, value in losses.items()
        },
    }


def _request():
    return {
        "contract": "source_faithful_student_and_teacher_answer_colon_v2",
        "checkpoint_sha256": "checkpoint",
        "dataset_fingerprint": "dataset",
        "basis_sha256": "basis",
        "basis_request_sha256": "request",
        "rank": 77,
        "update_indices": [1, 2],
        "validation_indices": [3, 4],
        "batch_size": 2,
        "relative_update_norm": 1e-4,
        "native_parity_gate": {
            "status": "passed",
            "absolute_loss_error": 0.0,
            "gradient_relative_l2_error": 0.0,
            "gradient_cosine": 1.0,
        },
    }


def test_corrected_combined_gate_requires_primary_scope():
    primary = analyze_endpoint_scope(
        [_batch(index, "endpoint_all_states", learned=0.82) for index in range(12)],
        bootstrap_samples=300,
        seed=2,
    )
    secondary = analyze_endpoint_scope(
        [_batch(index, "endpoint_layer11") for index in range(12)],
        bootstrap_samples=300,
        seed=2,
    )
    primary["request"] = _request()
    secondary["request"] = _request()
    report = combine_corrected_endpoint_reports(primary, secondary)
    assert report["training_authorized"] is False
    assert report["status"] == "corrected_layer11_requires_fresh_confirmation"
    rendered = render_corrected_endpoint_markdown(report)
    assert "both measured at the colon" in rendered
    assert "embedding state plus all 12" in rendered


def test_corrected_combiner_rejects_missing_native_parity():
    primary = analyze_endpoint_scope(
        [_batch(index, "endpoint_all_states") for index in range(12)],
        bootstrap_samples=100,
        seed=1,
    )
    secondary = analyze_endpoint_scope(
        [_batch(index, "endpoint_layer11") for index in range(12)],
        bootstrap_samples=100,
        seed=1,
    )
    request = _request()
    request["native_parity_gate"] = {"status": "failed"}
    primary["request"] = request
    secondary["request"] = request
    try:
        combine_corrected_endpoint_reports(primary, secondary)
    except ValueError as error:
        assert "parity" in str(error)
    else:
        raise AssertionError("failed native parity gate was accepted")

