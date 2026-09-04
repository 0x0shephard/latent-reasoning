from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_global_low_rank_lm_head.ipynb"
QWEN_NOTEBOOK = ROOT / "notebooks" / "kaggle_global_low_rank_lm_head_qwen.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_global_low_rank_head_notebook.py"
QWEN_BUILDER = ROOT / "scripts" / "build_kaggle_global_low_rank_head_qwen_notebook.py"


def notebook_text() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_notebook_and_builder_exist():
    assert NOTEBOOK.is_file()
    assert BUILDER.is_file()
    assert QWEN_NOTEBOOK.is_file()
    assert QWEN_BUILDER.is_file()


def test_notebook_declares_the_global_trajectory_contract():
    text = notebook_text()
    required = (
        "trajectory_whitened_margin_distilled_global_lm_head_v1",
        "activation_whitened_factors",
        "distil_nested_head",
        "compressed_onpolicy",
        "adaptive_r",
        "retains_98_percent_accuracy",
        "microseconds_per_visible_token",
        "generate_official_codi_fast",
        "fastpath_parity_examples",
        "TEST_LIMIT = 0",
    )
    for value in required:
        assert value in text


def test_notebook_uses_disjoint_training_partitions_and_full_test():
    text = notebook_text()
    assert "FIT_QUESTIONS = 1024" in text
    assert "SELECT_QUESTIONS = 256" in text
    assert "ONPOLICY_QUESTIONS = 256" in text
    assert "assert len(test_examples) == 1319" in text
    assert "isdisjoint" in text


def test_notebook_does_not_require_endpoint_state_inputs():
    text = notebook_text()
    assert "COLON_STATES_INPUT" not in text
    assert "READOUT_INPUT" not in text
    assert "colon_states.pt" not in text


def test_qwen_companion_uses_the_same_fitter_with_a_pinned_model():
    payload = json.loads(QWEN_NOTEBOOK.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "Qwen/Qwen2.5-Math-1.5B-Instruct" in text
    assert "f903dd76e2a9741d582f2a31248f1f5d0ac0e2bf" in text
    assert "activation_whitened_factors" in text
    assert "distil_nested_head" in text
    assert "compressed_onpolicy" in text
    assert "primary_retains_98_percent_accuracy" in text
