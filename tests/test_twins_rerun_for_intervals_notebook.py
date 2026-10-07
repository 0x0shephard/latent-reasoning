import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_twins_rerun_for_intervals.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_twins_rerun_for_intervals_notebook.py"


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


def test_notebook_runs_four_twin_runs_and_the_bootstrap():
    text = _text()
    assert "official_codi_gpt2.yaml" in text and "official_codi_llama1b.yaml" in text
    assert 'for profile in ("standard", "long")' in text and "scripts/run_codi_directed_counterfactual.py" in text
    assert "analyze_directed_counterfactual_bootstrap" in text and "No training" in text
    assert "REPRODUCTION_SUMMARY_INPUT" in text and '"--smoke"' in text and "New Dataset" in text
    assert "shutil.make_archive" not in text


def test_all_notebook_code_cells_parse():
    for i, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{i}")


def test_pinned_commit_has_the_analyzer():
    commit = _builder_run_commit()
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:scripts/analyze_directed_counterfactual_bootstrap.py"],
                           capture_output=True, text=True, check=True).stdout
    assert "def analyze" in shown and "pair_means" in shown
