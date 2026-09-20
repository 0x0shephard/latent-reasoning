import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_adaptive_kv_allocation.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_codi_adaptive_kv_allocation_notebook.py"


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_notebook_is_pinned_and_supports_verified_or_reconstructed_lineage():
    text = _notebook_text()
    assert 'RUN_COMMIT = "dac09573310156bf70eb6ff359d61399e33ae308"' in text
    assert "official_codi_fidelity_residual_xkv_holdout_v1" in text
    assert "official_codi_xkv_fidelity_frontier_holdout_v1" in text
    assert 'previous["decision"]["screen_passed"] is False' in text
    assert 'previous["final_replication"] is None' in text
    assert 'frontier["decision"]["development_passed"] is False' in text
    assert 'frontier["final_replication"] is None' in text
    assert '([explicit] if explicit else []) + all_candidates("summary.json")' in text
    assert "--allow-protocol-fallback" in text
    assert "reconstructing 4,992 predecessor train rows with sampling seed 20260916" in text


def test_notebook_freezes_the_disjoint_splits_and_candidate_grid():
    text = _notebook_text()
    assert "FRESH_CALIBRATION_EXAMPLES = 512" in text
    assert "FRESH_SCREEN_EXAMPLES = 512" in text
    assert "FINAL_START = 768" in text
    assert 'BASELINE_RANKS = "48,64,80"' in text
    assert 'ANSWER_WEIGHTS = "0,0.5,1"' in text
    assert "MAXIMUM_COMPONENT_RANK = 96" in text


def test_notebook_runs_the_allocator_and_preserves_the_locked_gate():
    text = _notebook_text()
    assert "scripts/run_codi_adaptive_kv_allocation.py" in text
    assert '"--storage-tolerance", "0.001"' in text
    assert '"--minimum-first-token-fidelity", "0.95"' in text
    assert '"--accuracy-noninferiority-margin", "0.02"' in text
    assert "The 551-question final slice remains untouched." in text
    assert "exactly the same modeled" in text
    assert "budget padding" in text


def test_notebook_runs_focused_tests_and_exports_auditable_outputs():
    text = _notebook_text()
    for expected in (
        "tests/test_adaptive_kv_allocation.py",
        "tests/test_adaptive_kv_allocation_runner.py",
        "tests/test_preanswer_kv_subspace.py",
        "summary.json",
        "adaptive_kv_allocation.pt",
        "predictions.jsonl",
    ):
        assert expected in text
    assert "shutil.make_archive" not in text
    assert "tarfile" not in text


def test_all_notebook_code_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source, filename=f"notebook-cell-{index}")


def test_builder_parses():
    ast.parse(BUILDER.read_text(), filename=str(BUILDER))
