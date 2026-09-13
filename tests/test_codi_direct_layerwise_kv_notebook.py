import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_corrected_notebook_matches_direct_layerwise_kv_question():
    path = ROOT / "notebooks" / "kaggle_codi_direct_layerwise_kv_subspaces.ipynb"
    text = "\n".join(cell.get("source", "") for cell in json.loads(path.read_text())["cells"])
    required = (
        "independently", "all six latent passes", "gold-answer-NLL gradients",
        "direct_layerwise_kv.pt", "tests/test_direct_layerwise_kv.py",
        "run_codi_direct_layerwise_kv_discovery.py", "positions 0–5",
        "specificity_bootstrap_95ci", "run_codi_causal_xkv.py",
        "ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION = False",
    )
    for fragment in required: assert fragment in text
