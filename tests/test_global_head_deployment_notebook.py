from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_global_head_deployment_benchmark.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_global_head_deployment_notebook.py"


def notebook_text() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_deployment_notebook_and_builder_exist_and_parse():
    assert NOTEBOOK.is_file()
    assert BUILDER.is_file()
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))


def test_deployment_notebook_reuses_the_locked_rank96_artifact():
    text = notebook_text()
    assert 'RUN_COMMIT = "b6a7ebf"' in text
    assert "global_low_rank_head.pt" in text
    assert "RANK = 96" in text
    assert "prior_retention >= 0.98" in text
    assert "distil_nested_head" not in text
    assert "activation_whitened_factors" not in text


def test_deployment_notebook_has_the_preregistered_systems_arms_and_metrics():
    text = notebook_text()
    required = (
        "deployment_low_rank_argmax_v1",
        "TIMING_BATCH_SIZES = (1, 8, 32)",
        "rank96_eager",
        "rank96_compiled_argmax",
        "rank96_triton_argmax",
        "_triton_partial_argmax_kernel",
        "torch.compile",
        "peak_temporary_bytes",
        "deployed_model_bytes",
        "exact_token_agreement_with_rank96_eager",
        "batch1_speedup_at_least_1_10x",
        "RUN_FULL_QUALITY = True",
    )
    for value in required:
        assert value in text


def test_deployment_notebook_uses_the_full_locked_quality_population():
    text = notebook_text()
    assert "assert len(test_examples) == 1319" in text
    assert "QUALITY_BATCH_SIZE = 32" in text
    assert "same_body_only_decoder" in text
    assert "merged_lora" in text
    assert "input_embedding_remains_dense" in text

