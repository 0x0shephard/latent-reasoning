import pytest

torch = pytest.importorskip("torch")

from scripts.analyze_directed_counterfactual_bootstrap import analyze, bootstrap_mean, markdown, pair_means  # noqa: E402


def test_pair_means_and_bootstrap_interval_cover_the_mean():
    hits = torch.tensor([1, 0, 1, 1, 0, 0], dtype=torch.float64)
    assert pair_means(hits).tolist() == [0.5, 1.0, 0.0]
    r = bootstrap_mean(torch.tensor([0.0, 1.0] * 50, dtype=torch.float64), samples=500, seed=1)
    assert abs(r["mean"] - 0.5) < 1e-9 and r["ci95"][0] < 0.5 < r["ci95"][1] and r["pairs"] == 100


def test_analyze_reads_a_saved_run(tmp_path):
    n = 40
    g = torch.Generator().manual_seed(0)
    target = torch.arange(n); own = target + 1000
    preds = {"own_gold": own, "target_gold": target, "located": [[1] if i % 4 == 0 else [] for i in range(n)]}
    for name, p in (("state_1", 0.1), ("state_3", 0.6), ("kv_2", 0.1), ("kv_4", 0.5), ("kv_0", 0.0), ("state_0", 0.0),
                    ("kv_all", 0.8), ("v89_all", 0.3), ("state_all", 0.5), ("v89_4", 0.05), ("v89_2", 0.05), ("v89_even", 0.2), ("v89_odd", 0.0)):
        hit = torch.rand(n, generator=g) < p
        preds[name] = torch.where(hit, target, own)
    torch.save(preds, tmp_path / "predictions.pt")
    r = analyze(tmp_path, samples=200, seed=3, min_rows=5)
    assert r["pairs"] == 20 and r["target_rate"]["kv_all"]["mean"] > r["target_rate"]["v89_all"]["mean"]
    c = r["contrast"]["state_3_minus_state_1"]
    assert c["ci95"][0] <= c["mean"] <= c["ci95"][1]
    assert r["located"]["1"]["rows"] == 10 and "state_site_minus_other" in r["located"]["1"]
    assert "fewer than" in r["located"]["3"]["note"]
    text = markdown(r)
    assert "| kv_all |" in text and "state_3_minus_state_1" in text
