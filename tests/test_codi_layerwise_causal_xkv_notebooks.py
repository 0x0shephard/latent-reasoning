import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(name):
    payload = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_layerwise_notebook_freezes_disjoint_transport_and_causal_contract():
    text = _text("kaggle_codi_layerwise_u28_stability.ipynb")
    for fragment in (
        "eigenvectors 4–31", "FIT_EXAMPLES = 2048", "SELECT_EXAMPLES = 512",
        "run_codi_layerwise_u28_stability.py", "energy-matched random",
        "selected_contiguous_attention_layers", "layerwise_u28.pt",
    ):
        assert fragment in text

def test_causal_xkv_notebook_has_equal_rank_controls_and_honest_timing_scope():
    text = _text("kaggle_codi_causal_xkv.ipynb")
    for fragment in (
        "per-layer SVD", "xKV-style cross-layer SVD", "random-protected xKV",
        "U28-protected xKV", 'RANKS = "28,32,48"', "bootstrap_95ci",
        "reduced_attention", "not actual runtime memory", "no fused factor-cache kernel",
    ):
        assert fragment in text
