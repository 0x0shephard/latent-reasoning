from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "kaggle_official_codi_kv_target_utility.ipynb"
)


def test_kaggle_target_utility_notebook_has_reproducible_execution_contract():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    sources = "\n".join(
        "".join(cell.get("source", ""))
        if isinstance(cell.get("source", ""), list)
        else cell.get("source", "")
        for cell in payload["cells"]
    )
    required = (
        "RUN_COMMIT",
        "requirements-official-codi.txt",
        "torch.cuda.is_available",
        "RUN_REPRODUCTION_GATE_IF_MISSING",
        "src.eval.official_codi",
        "run_official_codi_kv_target_utility.py",
        "RUN_SMOKE",
        "RUN_KIND_SCREEN",
        "RUN_POSITION_SCREEN",
        "RUN_LAYER_BAND_SCREEN",
        "RESUME_INPUT",
        "/kaggle/working/official_codi_kv_target_utility_export",
        "SHA256SUMS.txt",
    )
    for value in required:
        assert value in sources


def test_kaggle_target_utility_notebook_defaults_to_the_preregistered_kind_screen():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = "\n".join(
        cell.get("source", "")
        if isinstance(cell.get("source", ""), str)
        else "".join(cell.get("source", []))
        for cell in payload["cells"]
    )
    assert "RUN_KIND_SCREEN = True" in sources
    assert "RUN_POSITION_SCREEN = False" in sources
    assert "RUN_LAYER_BAND_SCREEN = False" in sources
    assert 'EXAMPLES_PER_SPLIT = 128' in sources
    assert 'BATCH_SIZE = 4' in sources
