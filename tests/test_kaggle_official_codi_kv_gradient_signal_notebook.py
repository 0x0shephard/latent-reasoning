from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "kaggle_official_codi_kv_gradient_signal.ipynb"
)


def _sources() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        cell.get("source", "")
        if isinstance(cell.get("source", ""), str)
        else "".join(cell.get("source", []))
        for cell in payload["cells"]
    )


def test_kaggle_gradient_signal_notebook_has_complete_contract():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    sources = _sources()
    required = (
        "RUN_COMMIT",
        "requirements-official-codi.txt",
        "torch.cuda.is_available",
        "PRIOR_TARGET_UTILITY_INPUT",
        "kind_seed3/run_manifest.json",
        "no_helpful_target_family_at_this_granularity",
        "run_official_codi_kv_gradient_signal.py",
        "RUN_SMOKE",
        "RUN_FULL_EXPERIMENT",
        "SPARSITY = 0.05",
        "MINIMUM_POSITIVE_FRACTION = 0.60",
        "RESUME_INPUT",
        "/kaggle/working/official_codi_kv_gradient_signal_export",
        "SHA256SUMS.txt",
    )
    for value in required:
        assert value in sources


def test_kaggle_gradient_signal_notebook_defaults_to_primary_key_gate():
    sources = _sources()
    assert 'PRIMARY_KIND = "key"' in sources
    assert "EXAMPLES_PER_SPLIT = 128" in sources
    assert "BATCH_SIZE = 4" in sources
    assert "RUN_FULL_EXPERIMENT = True" in sources
