import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_xkv_fidelity_frontier.ipynb"


def _text():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_notebook_pins_the_frontier_commit_and_four_inputs():
    text = _text()
    assert "181ea6c166bbe5f086ec9a048dc28793f0126173" in text
    assert "direct_cache_task_subspaces.pt" in text
    assert "task_aware_protected_xkv.pt" in text
    assert "official_codi_rank16_xkv_mechanism_holdout_v1" in text
    assert "__RUN_COMMIT__" not in text


def test_notebook_freezes_frontier_and_locked_final_boundary():
    text = _text()
    assert "DEVELOPMENT_START = 256" in text
    assert "DEVELOPMENT_EXAMPLES = 512" in text
    assert "FINAL_START = 768" in text
    assert 'RANKS = "16,24,32,40,48"' in text
    assert 'TASK_WEIGHTS = "0.0,0.5,1.0"' in text
    assert "final slice remains untouched" in text


def test_notebook_runs_focused_tests_and_frontier_runner():
    text = _text()
    assert "tests/test_xkv_fidelity_frontier.py" in text
    assert "tests/test_xkv_fidelity_frontier_runner.py" in text
    assert "run_codi_xkv_fidelity_frontier.py" in text
    assert "minimum-first-token-fidelity" in text
    assert "accuracy-noninferiority-margin" in text


def test_notebook_repairs_expanded_torch_archives_without_zip_export():
    text = _text()
    assert "repack_kaggle_torch_archive" in text
    assert "zipfile.ZIP_STORED" in text
    assert "restored_torch_artifacts" in text
    assert "shutil.make_archive" not in text
