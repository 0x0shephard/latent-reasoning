from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_28_to_global96_bridge.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_codi_subspace_bridge_notebook.py"


def notebook_text() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_bridge_notebook_and_builder_exist_and_parse():
    assert NOTEBOOK.is_file() and BUILDER.is_file()
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))


def test_bridge_notebook_reuses_frozen_artifacts_without_training():
    text = notebook_text()
    assert 'RUN_COMMIT = "e1c291a"' in text
    assert "colon_states_seed89/colon_states.pt" in text
    assert "global_low_rank_head.pt" in text
    assert "PRIMARY_BAND = (4, 32)" in text
    assert "global_artifact[\"state_dict\"]" in text
    assert "optimizer.step" not in text
    assert "distil_nested_head(" not in text


def test_bridge_notebook_contains_geometry_causality_and_trajectory_tests():
    text = notebook_text()
    required = (
        "GEOMETRY_NULL_REPLICATES = 200",
        "reference_capture_fraction",
        "empirical_one_sided_p",
        "cached_first_token",
        "answer_state_observer=observer",
        "global_rank96_retain_pc4_31_at_first",
        "global_rank96_remove_pc4_31_at_first",
        "pc4_31_every_answer_position",
        "MATCHED_RANDOM_REPLICATES = 4",
    )
    for value in required:
        assert value in text


def test_bridge_notebook_uses_full_locked_generation_and_paired_uncertainty():
    text = notebook_text()
    assert "assert len(test_examples) == 1319" in text
    assert "TEST_LIMIT = 0" in text
    assert "BOOTSTRAP_SAMPLES = 5000" in text
    assert "paired_bootstrap_delta" in text
    assert "shared_mechanism_supported" in text
    assert "write_jsonl" in text and "default=str" in text
    assert "fastpath_parity_examples" in text
