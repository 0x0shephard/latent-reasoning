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


def test_qwen_generalization_notebook_has_matched_endpoint_and_trajectory_controls():
    path = ROOT / "notebooks" / "kaggle_eigenspace_readout_generalization_qwen.ipynb"
    payload, source = _source(path)
    assert payload["nbformat"] == 4
    for required in (
        "Qwen/Qwen2.5-Math-1.5B-Instruct", "FIT_EXAMPLES = 512",
        "HF_MODEL_REVISION", "f903dd76e2a9741d582f2a31248f1f5d0ac0e2bf",
        "SELECT_EXAMPLES = 128", "TEST_EXAMPLES = 256", "RANKS = [64, 96, 192]",
        "MAX_NEW_TOKENS = 512", "Please reason step by step",
        "final_answer_span", "endpoint_states", "trajectory_states",
        "distilled_endpoint_r", "distilled_trajectory_r", "distilled_random_r",
        "RANDOM_NULL_REPLICATES = 20", "teacher_forced_replay_at_least_95pct",
        "deployable_generalization_supported", "end_to_end_token_throughput_speedup",
        '"gold": str(row["gold"])', "default=str", "truncated", "milliseconds_per_token",
        '"checkout", "--detach"',
    ):
        assert required in source
