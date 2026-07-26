from __future__ import annotations

from src.eval.official_codi_kv_gradient_signal_analysis import (
    analyze_kv_gradient_signal_batches,
    render_kv_gradient_signal_markdown,
)


def _batch():
    no_target = [1.0, 1.0]
    kinds = {}
    for kind in ("key", "value"):
        helpful = kind == "key"
        sparse = [0.70, 0.72] if helpful else [1.01, 1.00]
        conditions = {
            "full": {"validation_losses": [0.90, 0.91]},
            "sparse_aligned": {
                "validation_losses": sparse,
                "gradient_alignment": {
                    "cosine": 0.3 if helpful else -0.1,
                },
            },
            "random_sparse": {"validation_losses": [0.94, 0.95]},
            "shuffled_sparse": {"validation_losses": [0.96, 0.97]},
            "complement": {"validation_losses": [1.02, 1.03]},
        }
        # Every condition carries alignment in real output; only sparse is analyzed.
        for payload in conditions.values():
            payload.setdefault("gradient_alignment", {"cosine": 0.0})
        kinds[kind] = {"conditions": conditions}
    return {
        "validation": {"no_target_losses": no_target},
        "kinds": kinds,
    }


def test_gradient_signal_analysis_supports_sparse_key_only():
    report = analyze_kv_gradient_signal_batches(
        [_batch(), _batch(), _batch()],
        primary_kind="key",
        bootstrap_samples=200,
        seed=2,
    )
    assert report["gate"] == "primary_sparse_component_only_supported"
    assert (
        report["by_kind"]["key"]["classification"]
        == "sparse_answer_aligned_component_only_supported"
    )
    assert (
        report["by_kind"]["value"]["classification"]
        == "sparse_answer_aligned_component_not_supported"
    )
    markdown = render_kv_gradient_signal_markdown(report)
    assert "sparse vs full KV" in markdown
    assert "Primary KV kind: key" in markdown
