import json

import pytest

torch = pytest.importorskip("torch")

from scripts.analyze_correctness_readout import analyze, auroc, auroc_ci, feature_set, logistic_oof, markdown, slot_features  # noqa: E402
from scripts.run_codi_correctness_readout import CONTRACT, _number_text, confidence_rows  # noqa: E402


class _Tok:
    def decode(self, ids):
        i = ids[0]
        return f" {i}" if i < 50 else "x"


def test_runner_helpers():
    assert CONTRACT == "official_codi_correctness_readout_v1"
    assert _number_text(18.0) == "18" and _number_text(2.5) == "2.5" and _number_text(None) is None and _number_text(float("nan")) is None
    rows, index = confidence_rows(["a", "b", "c"], [18.0, None, 7.0])
    assert index == [0, 2] and rows[0]["answer"] == "18" and rows[1]["question"] == "c"


def test_auroc_and_feature_sets():
    assert auroc(torch.tensor([0.9, 0.8, 0.1, 0.2]), torch.tensor([True, True, False, False])) == 1.0
    assert auroc(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([True, False, False])) == 0.5
    assert auroc(torch.tensor([0.5]), torch.tensor([True])) is None
    ci = auroc_ci(torch.tensor([0.9, 0.8, 0.1, 0.2] * 10), torch.tensor([True, True, False, False] * 10), samples=50)
    assert ci[0] <= 1.0 <= ci[1] + 1e-9
    names = ["s0_top_p", "s0_entropy", "s0_agree_native", "s3_top_p", "s3_entropy", "s3_agree_native", "s0_s3_agree", "s3_s5_agree"]
    assert feature_set(names, [3]) == [3, 4, 5] and feature_set(names, [0, 3]) == [0, 1, 2, 3, 4, 5, 6] and feature_set(names, [5]) == []


def test_analyze_detects_wrong_answers_from_a_carrier(tmp_path):
    g = torch.Generator().manual_seed(0)
    n, slots, d = 300, 6, 8
    embeddings = torch.randn(60, d, generator=g); readout = embeddings.clone()   # tied head, numeric tokens are ids < 50
    gold = torch.randint(1, 40, (n,), generator=g).float()
    correct = torch.rand(n, generator=g) < 0.6
    native = torch.where(correct, gold, gold + 1)
    states = torch.randn(n, slots, d, generator=g) * 0.1
    # slot 3 (carrier) reads the gold value confidently when right and a different value when wrong; slot 4 reads noise
    for i in range(n):
        states[i, 3] += 4.0 * embeddings[int(gold[i])] if correct[i] else 4.0 * embeddings[int(gold[i]) + 5]
    torch.save({"states": states.half(), "gold": gold, "native": native, "correct": correct, "steps": torch.full((n,), 3),
                "first_token_prob": torch.rand(n, generator=g), "first_token_margin": torch.rand(n, generator=g), "intermediates": [[]] * n, "outputs": [""] * n},
               tmp_path / "readout_states.pt")
    r = analyze(tmp_path, _Tok(), embeddings, readout, carriers=[3], inert=[2, 4, 5], folds=5)
    assert r["wrong"] == int((~correct).sum()) and r["classifiers"]["carriers"]["auroc"] > 0.75
    assert r["classifiers"]["inert"]["auroc"] < 0.65 and r["classifiers"]["confidence"]["auroc"] < 0.65
    assert r["checks"]["p10_carriers_beat_confidence_by_0_05"] and r["checks"]["p10_carriers_beat_inert_by_0_05"]
    assert r["per_slot"]["3"]["agree_native_auroc"] > 0.75
    text = markdown(r)
    assert "| carriers |" in text and "| raw_states |" in text
    json.dumps(r)
    feats, names, top = slot_features(states, readout, torch.arange(50), [str(i) for i in range(50)], native)
    assert feats.shape == (n, 3 * slots + slots * (slots - 1) // 2) and len(names) == feats.shape[1]
    scores = logistic_oof(feats[:, feature_set(names, [3])], ~correct, folds=5)
    assert scores.shape == (n,)
