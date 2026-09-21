import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_fixed_basis_kv.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_codi_fixed_basis_kv_notebook.py"


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def _builder_run_commit() -> str:
    """The builder's module-level RUN_COMMIT, not the f-string placeholder."""
    tree = ast.parse(BUILDER.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "RUN_COMMIT":
                return ast.literal_eval(node.value)
    raise AssertionError("the builder has no module-level RUN_COMMIT")


def test_notebook_is_pinned_to_a_full_commit():
    commit = _builder_run_commit()
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    text = _notebook_text()
    assert f'RUN_COMMIT = "{commit}"' in text
    assert "assert len(RUN_COMMIT) == 40" in text


def test_notebook_freezes_the_protocol():
    text = _notebook_text()
    assert "BASIS_FIT_EXAMPLES = 1024" in text
    assert "SELECTION_EXAMPLES = 256" in text
    assert "SAMPLING_SEED = 20260921" in text
    assert 'RANK_GRID = "8,16,24,32,40,48"' in text
    assert 'RANDOM_BASIS_SEEDS = "20260921,20260922"' in text
    assert '"--minimum-retention", "0.98"' in text
    assert '"--minimum-first-token-fidelity", "0.95"' in text
    assert '"--accuracy-noninferiority-margin", "0.02"' in text


def test_notebook_runs_the_runner_and_focused_tests():
    text = _notebook_text()
    for expected in (
        "scripts/run_codi_fixed_basis_kv.py",
        "tests/test_fixed_basis_kv.py",
        "tests/test_fixed_basis_kv_runner.py",
        "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json",
        "summary.json",
        "fixed_basis_kv.pt",
        "predictions.jsonl",
    ):
        assert expected in text
    assert "shutil.make_archive" not in text


def test_notebook_states_the_claim_boundary():
    text = _notebook_text()
    assert "per-head" in text
    assert "not a latency" in text or "not measured" in text
    assert "random" in text


def test_all_notebook_code_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")


def test_builder_parses():
    ast.parse(BUILDER.read_text(), filename=str(BUILDER))
