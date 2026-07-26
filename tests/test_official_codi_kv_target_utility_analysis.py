from __future__ import annotations

from src.eval.official_codi_kv_target_utility_analysis import (
    analyze_target_utility_batches,
    render_target_utility_markdown,
)


def _batch(index: int):
    return {
        "batch_index": index,
        "validation": {
            "original_losses": [1.0, 1.0],
            "no_target_losses": [0.9, 0.9],
        },
        "groups": {
            "key_all": {
                "definition": {
                    "name": "key_all",
                    "kind": "key",
                    "layers": [0],
                    "positions": [0],
                },
                "candidate_train_loss": 0.4,
                "shuffled_train_loss": 0.6,
                "candidate_validation_losses": [0.7, 0.75],
                "shuffled_validation_losses": [0.95, 0.96],
                "gradient_alignment": {
                    "candidate": {"cosine": 0.3, "dot": 2.0},
                    "shuffled": {"cosine": 0.0, "dot": 0.0},
                },
            },
            "value_all": {
                "definition": {
                    "name": "value_all",
                    "kind": "value",
                    "layers": [0],
                    "positions": [0],
                },
                "candidate_train_loss": 0.5,
                "shuffled_train_loss": 0.5,
                "candidate_validation_losses": [0.92, 0.93],
                "shuffled_validation_losses": [0.91, 0.92],
                "gradient_alignment": {
                    "candidate": {"cosine": -0.1, "dot": -1.0},
                    "shuffled": {"cosine": 0.0, "dot": 0.0},
                },
            },
        },
    }


def test_analysis_classifies_helpful_and_inconclusive_targets():
    report = analyze_target_utility_batches(
        [_batch(0), _batch(1)],
        bootstrap_samples=200,
        seed=4,
    )
    assert report["screen_status"] == "helpful_target_family_found"
    assert (
        report["target_groups"]["key_all"]["classification"]
        == "helpful_target_family"
    )
    assert (
        report["target_groups"]["value_all"]["classification"]
        == "interfering_target_family"
    )
    markdown = render_target_utility_markdown(report)
    assert "Official CODI hierarchical KV target utility" in markdown
    assert "key_all" in markdown
