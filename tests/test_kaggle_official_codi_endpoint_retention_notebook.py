from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "kaggle_official_codi_endpoint_retention.ipynb"


def _sources() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    return "\n".join(
        cell.get("source", "")
        if isinstance(cell.get("source", ""), str)
        else "".join(cell.get("source", []))
        for cell in payload["cells"]
    )


def test_retention_notebook_has_the_frozen_complete_contract():
    sources = _sources()
    required = (
        "RUN_COMMIT",
        "requirements-official-codi.txt",
        "torch.cuda.is_available",
        "source_faithful_student_and_teacher_answer_colon_v2",
        "answer_conditioned_colon_block_states_v1",
        "parameter_aware_colon_final_two_blocks_v1",
        "ENERGY_BASIS_INPUT",
        "native_loss_gradient_parity.json",
        "basis.pt diagnostics",
        "paths_by_sha",
        "byte-identical mounted copies",
        "run_official_codi_endpoint_retention.py",
        "analyze_official_codi_endpoint_retention.py",
        "RUN_SMOKE",
        "RESUME_INPUT",
        "SHA256SUMS.txt",
        "/kaggle/working/official_codi_endpoint_retention_export",
    )
    for value in required:
        assert value in sources


def test_retention_notebook_defaults_to_rank_matched_three_seed_full_eval():
    sources = _sources()
    assert "TRAINING_EXAMPLES = 512" in sources
    assert "TRAINING_SEEDS = [53, 59, 61]" in sources
    assert "NONINFERIORITY_MARGIN = 0.01" in sources
    assert "RUN_FULL = True" in sources
    assert 'eval_limit=0' in sources
    for arm in (
        "answer_only",
        "full_common",
        "energy_selected",
        "answer_conditioned_selected",
        "parameter_aware_selected",
        "energy_complement",
        "answer_conditioned_complement",
        "parameter_aware_complement",
    ):
        assert arm in sources
