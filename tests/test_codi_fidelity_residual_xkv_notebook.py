import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_fidelity_residual_xkv.ipynb"


def _text():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_notebook_pins_core_commit_and_required_inputs():
    text = _text()
    assert "9edade1401b8cf8b67e4d1ac424fd7cf8100573a" in text
    assert "direct_cache_task_subspaces.pt" in text
    assert "task_aware_protected_xkv.pt" in text
    assert "official_codi_xkv_fidelity_frontier_holdout_v1" in text
    assert "__RUN_COMMIT__" not in text


def test_notebook_freezes_fresh_splits_and_candidate_grid():
    text = _text()
    assert "FRESH_FIT_EXAMPLES = 512" in text
    assert "FRESH_SCREEN_EXAMPLES = 512" in text
    assert "FINAL_START = 768" in text
    assert 'RANKS = "48,64,80,96"' in text
    assert 'RESIDUAL_RANKS = "1,2,4"' in text
    assert "final slice remains untouched" in text


def test_notebook_runs_focused_tests_and_gated_runner():
    text = _text()
    assert "tests/test_fidelity_residual_xkv.py" in text
    assert "tests/test_fidelity_residual_xkv_runner.py" in text
    assert "run_codi_fidelity_residual_xkv.py" in text
    assert "minimum-first-token-fidelity" in text
    assert "accuracy-noninferiority-margin" in text


def test_notebook_repairs_expanded_torch_archives_without_zip_export():
    text = _text()
    assert "repack_kaggle_torch_archive" in text
    assert "zipfile.ZIP_STORED" in text
    assert "restored_torch_artifacts" in text
    assert "shutil.make_archive" not in text
