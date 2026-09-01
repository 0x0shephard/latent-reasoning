from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    combined = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{index}")
    return payload, combined


def test_codi_distillation_notebook_has_matched_controls_and_test_gate():
    path = ROOT / "notebooks" / "kaggle_codi_eigenspace_distilled_readout.ipynb"
    payload, source = _source(path)
    assert payload["nbformat"] == 4
    for required in (
        "learned_eigen_r32", "learned_random_r32", "fixed_eigen_r32",
        "answer_state_observer", "FIT_EXAMPLES = 1536", "TEST_LIMIT = 0",
        "hybrid_supported", "numeric_exact_match", "wall_clock_seconds",
        "tests/test_eigenspace_readout.py", '"checkout", "--detach"',
    ):
        assert required in source


def test_qwen_generalization_notebook_has_frozen_cross_model_controls():
    path = ROOT / "notebooks" / "kaggle_eigenspace_readout_generalization_qwen.ipynb"
    payload, source = _source(path)
    assert payload["nbformat"] == 4
    for required in (
        "Qwen/Qwen2.5-Math-1.5B-Instruct", "FIT_EXAMPLES = 512",
        "SELECT_EXAMPLES = 128", "TEST_EXAMPLES = 256", "RANKS = [32, 64]",
        "readout_aware", "skip4", "RANDOM_NULL_REPLICATES = 20",
        "generalization_supported", "inconclusive_low_baseline_correct_count",
        "isolated_head_speedup", '"checkout", "--detach"',
    ):
        assert required in source
