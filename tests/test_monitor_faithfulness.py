import json

import pytest

torch = pytest.importorskip("torch")

from scripts.analyze_monitor_faithfulness import (  # noqa: E402
    analyze,
    cross_validated_probe,
    numeric_vocabulary,
    ridge_fit,
    ridge_predict,
    targets_from_pairs,
)


class NumTok:
    """Tokens 0..19 decode to the integers 0..19; 20 decodes to a superscript digit
    (``str.isdigit`` is true for it, ``int`` fails); 21.. decode to letters."""
    def decode(self, ids):
        i = int(ids[0])
        if i >= 29:
            raise TypeError("sequence item 0: expected str instance, NoneType found")   # GPT-2 slow tokenizer past its vocabulary
        return str(i) if i < 20 else ("\u00b2" if i == 20 else chr(97 + i - 21))


def test_numeric_vocabulary_and_ridge_roundtrip():
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(30, 8, generator=g)
    ids, strings = numeric_vocabulary(NumTok(), emb)
    assert strings == [str(i) for i in range(20)] and ids.tolist() == list(range(20))
    x = torch.randn(200, 8, generator=g); tgt = torch.randint(0, 20, (200,), generator=g)
    w = ridge_fit(x, emb[ids][tgt], lam=1e-3)
    # a linear map that is exactly recoverable: targets are a linear function of x
    y = x @ torch.randn(8, 8, generator=g)
    assert ridge_fit(x, y, lam=1e-6).shape == (9, 8)
    pred = ridge_predict(x, w, emb[ids])
    assert pred.shape == (200,)


def test_targets_and_analyze_on_a_synthetic_run(tmp_path):
    g = torch.Generator().manual_seed(1)
    emb = torch.randn(30, 8, generator=g); tok = NumTok()
    ids, strings = numeric_vocabulary(tok, emb)
    pairs = []
    rows_answer, rows_pen = [], []
    for i in range(40):
        a, b = (i % 15) + 2, (i % 15) + 3
        pairs.append({"cot": f"<<1+1=2>> <<2+{a - 2}={a}>> <<{a}*1={a}>>", "gold": str(a),
                      "twin_cot": f"<<1+1=2>> <<2+{b - 2}={b}>> <<{b}*1={b}>>", "twin_gold": str(b)})
        rows_answer += [a, b]; rows_pen += [a, b]
    (tmp_path / "pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    t = targets_from_pairs(tmp_path / "pairs.jsonl", strings)
    assert t["answer"].tolist() == rows_answer and t["penultimate"].tolist() == rows_pen and t["steps"].tolist() == [3] * 80
    # slot 0 states encode the answer linearly (readable); slot 1 states are noise
    n = 80
    states = torch.zeros(n, 2, 8)
    states[:, 0] = emb[ids][t["answer"]] + 0.05 * torch.randn(n, 8, generator=g)
    states[:, 1] = torch.randn(n, 8, generator=g)
    torch.save({"states": states.half()}, tmp_path / "trajectory_states.pt")
    partner = torch.arange(n) ^ 1
    target_gold = torch.tensor(rows_answer)[partner]
    # slot 0 transports in every row; slot 1 never
    preds = {"own_gold": torch.tensor(rows_answer), "target_gold": target_gold, "located": [[]] * n,
             "state_0": target_gold.clone(), "state_1": torch.tensor(rows_answer)}
    torch.save(preds, tmp_path / "predictions.pt")
    r = analyze(tmp_path, tok, emb, emb, folds=5)
    s0, s1 = r["slots"]["0"]["answer"], r["slots"]["1"]["answer"]
    assert s0["readability"] > 0.9 and s0["faithfulness"] == 1.0 and s0["transport"] == 1.0
    assert s1["transport"] == 0.0 and (s1["faithfulness"] in (None, 0.0))
    oof, _ = cross_validated_probe(states[:, 0], t["answer"], emb[ids], folds=5)
    assert oof.shape == (n,)
