import json

import pytest

torch = pytest.importorskip("torch")

from scripts.analyze_masking_bootstrap import analyze, bootstrap_drop, markdown  # noqa: E402


def test_bootstrap_drop_brackets_the_point_estimate():
    base = torch.ones(400, dtype=torch.bool)
    cond = torch.ones(400, dtype=torch.bool); cond[:100] = False       # 25-point drop
    r = bootstrap_drop(base, cond, samples=500, seed=1)
    assert r["mean"] == pytest.approx(0.25) and r["ci95"][0] < 0.25 < r["ci95"][1] and r["n"] == 400
    assert 0.18 < r["ci95"][0] and r["ci95"][1] < 0.32
    assert bootstrap_drop(base[:0], cond[:0]) == {"mean": None, "ci95": None, "n": 0}
    again = bootstrap_drop(base, cond, samples=500, seed=1)
    assert again == r                                                   # seeded


def test_analyze_reads_the_masking_predictions(tmp_path):
    n = 200
    gold = torch.arange(n) % 7
    base = gold.clone()
    is_long = torch.arange(n) >= 120
    masked = gold.clone(); masked[120:160] += 1                         # costs 40 of 80 long rows, nothing short
    torch.save({"conditions": {"mask_3": masked, "mean_5": gold.clone()}, "base_pred": base, "gold": gold, "is_long": is_long,
                "steps": torch.where(is_long, 3, 1)}, tmp_path / "predictions.pt")
    r = analyze(tmp_path, samples=300, seed=3)
    assert r["rows"] == n and set(r["parts"]) == {"all", "short", "long"}
    assert r["parts"]["long"]["mask_3"]["mean"] == pytest.approx(0.5) and r["parts"]["short"]["mask_3"]["mean"] == pytest.approx(0.0)
    assert r["parts"]["all"]["mask_3"]["mean"] == pytest.approx(0.2) and r["parts"]["long"]["mean_5"]["mean"] == pytest.approx(0.0)
    assert r["parts"]["long"]["mask_3"]["ci95"][0] > 0.35 and r["parts"]["long"]["mask_3"]["n"] == 80
    text = markdown(r, conditions=("mask_3", "mean_5"))
    assert "| mask_3 |" in text and "50.0 [" in text
    json.dumps(r)
