import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_task_aware_protected_xkv.ipynb"


def _notebook_text():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_notebook_contains_confirmatory_split_and_predecessor_lineage_checks():
    text = _notebook_text()
    assert "__RUN_COMMIT__" not in text
    assert "56e4bb3851be4dbc8f7472af5b1eb3f8e44da91d" in text
    assert "CONFIRM_EXAMPLES = 512" in text
    assert "CALIBRATION_EXAMPLES = 512" in text
    assert "SOURCE_WAS_RERUN" in text
    assert "run_codi_confirm_and_compress_native_kv.py" in text


def test_notebook_runs_required_baselines_controls_and_decision_tables():
    text = _notebook_text()
    assert "RANDOM_PROTECTION_CONTROLS = 20" in text
    assert "paired_ordinary_xkv_comparisons" in text
    assert "full_task_aware_xkv" in text
    assert "INT8/INT4" in text
    assert "not production latency" in text


def test_notebook_invokes_focused_tensor_tests_before_the_experiment():
    text = _notebook_text()
    assert "tests/test_task_aware_xkv.py" in text
    assert "tests/test_direct_cache_task_subspace.py" in text
    assert "subprocess.run(command, check=True)" in text
