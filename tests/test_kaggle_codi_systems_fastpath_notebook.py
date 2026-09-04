from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_source() -> str:
    path = ROOT / "notebooks" / "kaggle_codi_systems_fastpath.ipynb"
    notebook = json.loads(path.read_text())
    return "\n".join(
        cell.get("source", "")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_fastpath_notebook_freezes_complete_evaluation_and_timing_protocol():
    source = notebook_source()
    for fragment in (
        "RUN_COMMIT =",
        "assert len(test_examples) == int(cfg.eval.expected_counts.gsm8k) == 1319",
        "TIMING_BATCH_SIZES = (1, 8, 32)",
        "TIMING_REPEATS = 3",
        "b0_reference_fp32_lora_m6",
        "b1_body_only_fp32_lora_m6",
        "b2_body_only_fp32_merged_m6",
        "b3_exact_fastpath_fp32_m6",
        "b4_bucketed_fp32_m6",
        "b5_fastpath_fp16_m6",
        "b6_fastpath_fp16_m5",
        "b7_fastpath_fp16_m5_numeric",
        "b8_compiled_fastpath_fp16_m5",
    ):
        assert fragment in source


def test_fastpath_notebook_has_parity_quality_and_speed_gates():
    source = notebook_source()
    for fragment in (
        'output_string_agreement_with_reference"] == 1.0',
        'accuracy_retained_fraction"] >= 0.98',
        'timing_speedup_x"]["1"] >= 1.50',
        "paired_bootstrap_95ci_accuracy_difference",
        "full_test_preparation_seconds",
        "microseconds_per_visible_token",
        "padded_prompt_tokens",
        "profiler.export_chrome_trace",
    ):
        assert fragment in source


def test_fastpath_builder_and_notebook_agree_on_output_path():
    builder = (
        ROOT / "scripts" / "build_kaggle_codi_systems_fastpath_notebook.py"
    ).read_text()
    assert '"notebooks" / "kaggle_codi_systems_fastpath.ipynb"' in builder
