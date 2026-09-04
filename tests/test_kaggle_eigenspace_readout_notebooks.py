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


def test_qwen_answer_localization_notebook_has_rank_sweep_and_causal_controls():
    path = ROOT / "notebooks" / "kaggle_qwen_answer_eigenspace_localization.ipynb"
    payload, source = _source(path)
    assert payload["nbformat"] == 4
    for required in (
        "does-the-answer-eigenspace-generalize-beyond-codi",
        "RANKS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64",
        "SUFFICIENCY_THRESHOLD = 0.99", "intervention_metrics",
        "keep_full_agreement", "remove_full_agreement",
        "minimum_sufficient_rank_interval", "selected_eigenvector_indices",
        "complete_answer_suffix", "keep_aware_r", "remove_aware_r",
        "keep_random_r", "paired_bootstrap_95ci",
        "full_suffix_outcome_replay", "specificity_over_random_supported",
        "localized_sufficient_and_necessary_answer_eigenspace",
        "layer_28_post_final_norm_pre_lm_head_first_boxed_answer_token",
    ):
        assert required in source


def test_codi_position_conditioned_notebook_has_locality_controls_and_timing():
    path = ROOT / "notebooks" / "kaggle_codi_position_conditioned_readout_v2.ipynb"
    payload, source = _source(path)
    assert payload["nbformat"] == 4
    for required in (
        "first_token_pc4_31_then_full", "same_pc4_31_everywhere",
        "fixed_position_local", "learned_position_local",
        "learned_position_local_onpolicy", "learned_global_r32",
        "permuted_position_local_onpolicy", "permuted_sources",
        "learned_global_r64", "DEFAULT_ANSWER_POSITION_BUCKETS",
        "p2_plus", "MIN_FIT_STATES_PER_BUCKET = 64",
        "MIN_SELECT_STATES_PER_BUCKET = 16",
        "answer_position_bucket", "confirmed_colon_pc_band_4_31",
        "first_token_control_reproduces_38_06_within_2_points",
        "microseconds_per_question", "microseconds_per_visible_token",
        "visible_generated_tokens", "position_locality_supported",
        "installed_package_version", "observed_package_versions", "generation_metadata",
        "PackageNotFoundError", "version as package_version",
        "TEST_LIMIT = 0", "tests/test_position_conditioned_readout.py",
        '"checkout", "--detach"', "default=str",
    ):
        assert required in source
