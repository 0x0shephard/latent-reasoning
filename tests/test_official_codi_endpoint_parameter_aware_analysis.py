from __future__ import annotations

from src.eval.official_codi_endpoint_parameter_aware_analysis import (
    analyze_parameter_aware_utility,
    build_parameter_aware_final_report,
    render_parameter_aware_markdown,
)
from src.mech.endpoint_parameter_aware import (
    PARAMETER_AWARE_ARMS,
    PARAMETER_AWARE_SCOPE,
)


def _batch(index: int, candidate: float = 0.5):
    arm_values = {
        "full_blocks": 0.67,
        "parameter_aware": candidate,
        "energy_rank_matched": 0.70,
        "random_rank_matched": 0.72,
        "shuffled_answer_rank_matched": 0.74,
        "shuffled_teacher": 0.76,
        "complement": 0.68,
    }
    assert set(arm_values) == set(PARAMETER_AWARE_ARMS)
    return {
        "scope": PARAMETER_AWARE_SCOPE,
        "batch_index": index,
        "validation": {
            "original_losses": [1.0, 1.0],
            "answer_only_losses": [0.80, 0.81],
        },
        "arms": {
            name: {
                "validation_losses": [value, value + 0.01],
                "gradient_alignment": {
                    "cosine": 0.2 if name == "parameter_aware" else 0.0
                },
            }
            for name, value in arm_values.items()
        },
    }


def _basis_payload(status: str = "candidate_selected"):
    return {
        "request_sha256": "selection-request",
        "metadata": {"native_parity_gate": {"status": "passed"}},
        "selection": {
            "status": status,
            "candidate_states": [11, 12],
            "hutchinson_probes": 8,
            "rank_by_state": [0] * 11 + [1, 2],
            "total_rank": 3 if status == "candidate_selected" else 0,
        },
    }


def test_parameter_aware_gate_passes_all_matched_controls():
    utility = analyze_parameter_aware_utility(
        [_batch(index) for index in range(20)],
        bootstrap_samples=1000,
        seed=4,
    )
    assert utility["gate"]["supported"] is True
    utility["request"] = {"basis_request_sha256": "selection-request"}
    report = build_parameter_aware_final_report(_basis_payload(), utility)
    assert report["training_authorized"] is True
    assert report["status"] == "parameter_aware_training_authorized"
    rendered = render_parameter_aware_markdown(report)
    assert "Hutchinson" in rendered
    assert "shuffled_teacher" in rendered


def test_no_parameter_aware_direction_closes_before_utility():
    basis = _basis_payload("no_stable_positive_parameter_cosines")
    basis["selection"]["rank_by_state"] = [0] * 13
    report = build_parameter_aware_final_report(basis, None)
    assert report["training_authorized"] is False
    assert report["status"] == "no_stable_parameter_aware_directions"
