import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_rank16_xkv_mechanism_confirmation.ipynb"


def _text():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_pins_the_rank16_confirmation_code_and_inputs():
    text = _text()
    assert "2e9b78b01b82e618405a825535173e134612fd3d" in text
    assert "task_aware_protected_xkv.pt" in text
    assert "direct_cache_task_subspaces.pt" in text
    assert "__RUN_COMMIT__" not in text


def test_notebook_freezes_splits_rank_and_random_controls():
    text = _text()
    assert "CONFIRMATION_START = 256" in text
    assert "CONFIRMATION_EXAMPLES = 512" in text
    assert "FINAL_START = 768" in text
    assert "RANK = 16" in text
    assert "RANDOM_CONTROLS = 100" in text
    assert "100 energy-matched random" in text


def test_notebook_runs_preflight_tests_and_the_gated_runner():
    text = _text()
    assert "tests/test_rank16_xkv_confirmation.py" in text
    assert "tests/test_rank16_xkv_runner.py" in text
    assert "run_codi_rank16_xkv_mechanism_confirmation.py" in text
    assert "Mechanism stage correctly stopped" in text
    assert "final slice remains untouched" in text


def test_notebook_repairs_kaggle_expanded_torch_archives_without_export_collision():
    text = _text()
    assert "repack_kaggle_torch_archive" in text
    assert "zipfile.ZIP_STORED" in text
    assert "restored_torch_artifacts" in text
    assert "shutil.make_archive" not in text
