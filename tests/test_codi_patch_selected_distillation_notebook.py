import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_patch_selected_distillation.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_codi_patch_selected_distillation_notebook.py"


def _text() -> str:
    return "\n".join("".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"])


def _builder_run_commit() -> str:
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RUN_COMMIT":
                    return ast.literal_eval(node.value)
    raise AssertionError("no module-level RUN_COMMIT")


def test_notebook_is_pinned_to_a_full_commit():
    commit = _builder_run_commit()
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    assert f'RUN_COMMIT = "{commit}"' in _text()


def test_notebook_exposes_controls_inputs_and_pairing():
    text = _text()
    assert "scripts/run_codi_patch_selected_distillation.py" in text
    assert 'SEEDS = "1,2,3"' in text and "RUN_PRELIMINARY_ONLY" in text and "--preliminary-only" in text
    assert "PRIMARY_OUTPUT_INPUT" in text and "--shared-from" in text
    assert "PREVIOUS_OUTPUT_INPUT" in text and "--resume-from" in text
    assert "--max-seconds" in text and "MAX_SECONDS" in text and '"--smoke"' in text
    assert '"--batch-size", "16"' in text
    for arm in ("patch", "patch_anchor", "patch_anchor_adaptive", "causal_anchor"):
        assert arm in text
    assert "codi_recovery_primary" in text and "predictions.jsonl" in text
    assert "shutil.make_archive" not in text


def test_notebook_runs_focused_tests_and_states_boundaries():
    text = _text()
    assert "tests/test_causal_subspace_distillation.py" in text
    assert "tests/test_patch_selected_distillation_runner.py" in text
    assert "transfer patching" in text and "breakage patching" in text
    assert "repairs" in text and "breaks" in text and "1,319" in text
    assert "designed after" in text


def test_all_notebook_code_cells_parse():
    for i, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{i}")


def test_builder_parses():
    ast.parse(BUILDER.read_text(), filename=str(BUILDER))
