import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_trajectory_supervision.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_codi_trajectory_supervision_notebook.py"


def _notebook_text() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def _builder_run_commit() -> str:
    tree = ast.parse(BUILDER.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
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


def test_notebook_runs_smoke_then_the_frozen_protocol():
    text = _notebook_text()
    assert "scripts/run_codi_trajectory_supervision.py" in text
    assert '"--smoke"' in text
    assert "SMOKE_OUTPUT_DIR" in text and "OUTPUT_DIR" in text
    assert '"--batch-size", "8"' in text
    assert '"--bootstrap-samples", "10000"' in text
    assert "tests/test_trajectory_supervision.py" in text
    assert "tests/test_trajectory_supervision_runner.py" in text
    assert "official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json" in text
    for artefact in ("summary.json", "trajectory_supervision.pt", "predictions.jsonl"):
        assert artefact in text
    assert "shutil.make_archive" not in text


def test_notebook_states_arms_gates_and_boundaries():
    text = _notebook_text()
    for arm in ("codi", "kava", "value_odd", "random_odd", "value_even", "recon_odd"):
        assert arm in text
    assert "screen" in text.lower()
    assert "gradient norm" in text.lower()
    assert "1,024" in text or "1024" in text


def test_all_notebook_code_cells_parse():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")


def test_builder_parses():
    ast.parse(BUILDER.read_text(), filename=str(BUILDER))
