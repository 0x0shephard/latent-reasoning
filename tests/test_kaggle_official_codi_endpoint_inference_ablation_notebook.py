from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "kaggle_official_codi_endpoint_inference_ablation.ipynb"


def test_notebook_is_run_all_and_contains_frozen_causal_contract():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    required = [
        "RUN_COMMIT",
        "RANDOM_REPLICATES = 20",
        "collect_official_codi_endpoint_activation_stats.py",
        "kagglehub.dataset_download",
        "corrected-official-codi-answer-cue-endpoint-tsv-c",
        "run_official_codi_endpoint_inference_ablation.py",
        "analyze_official_codi_endpoint_inference_ablation.py",
        "Accuracy-critical directions/groups",
        "official_codi_endpoint_inference_ablation_forced_cue_export",
        "Native CODI decoding does not generate the answer cue",
        "SHA256SUMS.txt",
    ]
    for value in required:
        assert value in source
    assert "RUN_FULL = True" in source
    assert "assert len(ARMS) == 82" in source
