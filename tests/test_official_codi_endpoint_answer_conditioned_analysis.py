from __future__ import annotations

from src.eval.official_codi_endpoint_answer_conditioned_analysis import (
    analyze_answer_conditioned_utility,
    build_answer_conditioned_final_report,
    render_answer_conditioned_markdown,
)
from src.mech.endpoint_answer_conditioned import (
    ANSWER_CONDITIONED_ARMS,
    ANSWER_CONDITIONED_SCOPE,
)


def _batch(index: int, candidate: float = 0.5):
    arm_values = {
        "full_blocks": 0.67,
        "answer_conditioned": candidate,
        "energy_rank_matched": 0.70,
        "random_rank_matched": 0.72,
        "shuffled_answer_rank_matched": 0.74,
        "shuffled_teacher": 0.76,
        "complement": 0.68,
    }
    assert set(arm_values) == set(ANSWER_CONDITIONED_ARMS)
    return {
        "scope": ANSWER_CONDITIONED_SCOPE,
        "batch_index": index,
        "validation": {
            "original_losses": [1.0, 1.0],
            "answer_only_losses": [0.80, 0.81],
        },
        "arms": {
            name: {
                "validation_losses": [value, value + 0.01],
                "gradient_alignment": {
                    "cosine": 0.2 if name == "answer_conditioned" else 0.0
                },
            }
            for name, value in arm_values.items()
        },
    }


def _basis_payload(status: str = "candidate_selected"):
    return {
        "request_sha256": "selection-request",
        "metadata": {
            "native_parity_gate": {
                "status": "passed",
                "absolute_loss_error": 0.0,
                "gradient_relative_l2_error": 0.0,
                "gradient_cosine": 1.0,
            }
        },
        "selection": {
            "status": status,
            "rank_by_state": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            "total_rank": 3 if status == "candidate_selected" else 0,
        },
    }


def test_answer_conditioned_gate_passes_all_matched_controls():
    report = analyze_answer_conditioned_utility(
        [_batch(index) for index in range(20)],
        bootstrap_samples=1000,
        seed=4,
    )
    assert report["gate"]["supported"] is True
    report["request"] = {"basis_request_sha256": "selection-request"}
    combined = build_answer_conditioned_final_report(_basis_payload(), report)
    assert combined["training_authorized"] is True
    assert combined["status"] == "answer_conditioned_training_authorized"
    rendered = render_answer_conditioned_markdown(combined)
    assert "embedding state is excluded" in rendered
    assert "shuffled_teacher" in rendered


def test_no_selected_direction_closes_before_utility():
    basis = _basis_payload("no_stable_positive_directions")
    basis["selection"]["rank_by_state"] = [0] * 13
    report = build_answer_conditioned_final_report(basis, None)
    assert report["training_authorized"] is False
    assert report["status"] == "no_stable_answer_conditioned_directions"
