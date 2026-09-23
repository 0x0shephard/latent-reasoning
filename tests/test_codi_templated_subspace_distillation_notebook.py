import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_templated_subspace_distillation.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_codi_templated_subspace_distillation_notebook.py"


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


def test_notebook_exposes_go_no_go_account_split_resume_and_aggregation():
    text = _text()
    assert "scripts/run_codi_templated_subspace_distillation.py" in text
    assert 'SEEDS = "1,2,3"' in text and "RUN_PRELIMINARY_ONLY" in text and "--preliminary-only" in text
    assert "PREVIOUS_OUTPUT_INPUT" in text and "--resume-from" in text
    assert "OTHER_ACCOUNT_OUTPUT_INPUT" in text and "--aggregate-from" in text
    assert "--max-seconds" in text and "MAX_SECONDS" in text
    assert '"--smoke"' in text and "SMOKE_OUTPUT_DIR" in text
    assert '"--batch-size", "16"' in text
    for artefact in ("summary.json", "selectors.json", "preliminary.json", "teacher_cache.pt", "predictions.jsonl"):
        assert artefact in text
    assert "shutil.make_archive" not in text


def test_notebook_runs_focused_tests_and_states_boundaries():
    text = _text()
    assert "tests/test_templated_arithmetic.py" in text
    assert "tests/test_templated_subspace_distillation_runner.py" in text
    assert "tests/test_causal_subspace_distillation.py" in text
    for arm in ("none", "variance", "causal"):
        assert arm in text
    assert "retain-only" in text and "norm-matched" in text and "go/no-go" in text
    assert "GSM8K" in text  # the boundary: generated problems say nothing about GSM8K difficulty


def test_all_notebook_code_cells_parse():
    for i, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{i}")


def test_builder_parses():
    ast.parse(BUILDER.read_text(), filename=str(BUILDER))
