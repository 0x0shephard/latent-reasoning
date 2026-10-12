import json

import pytest

torch = pytest.importorskip("torch")

from scripts.analyze_prompt_leakage import analyze, attributes_from_pairs, classify, first_operator, markdown  # noqa: E402


def test_first_operator_reads_the_first_equation():
    assert first_operator("<<16-3-4=9>> <<9*2=18>>") == 1 and first_operator("<<2*3=6>>") == 2
    assert first_operator("<<(8-7)=1>>") == 1 and first_operator("<<600*30/100=180>>") == 2 and first_operator("") is None


def _run(tmp_path, n_pairs=60, d=16):
    g = torch.Generator().manual_seed(0)
    ops = torch.randint(0, 4, (n_pairs,), generator=g)
    steps = torch.randint(1, 5, (n_pairs,), generator=g)
    lines = []
    for op, st in zip(ops.tolist(), steps.tolist()):
        sym = "+-*/"[op]
        cot = " ".join(f"<<{i + 2}{sym}1={i + 3}>>" for i in range(st))
        lines.append(json.dumps({"question": "q", "cot": cot, "gold": "1", "twin_question": "q2", "twin_cot": cot, "twin_gold": "2"}))
    (tmp_path / "pairs.jsonl").write_text("\n".join(lines) + "\n")
    # slot 0 encodes the operator linearly; slot 1 encodes chain length; slot 2 is noise
    states = torch.randn(2 * n_pairs, 3, d, generator=g) * 0.1
    op_rows = ops.repeat_interleave(2); long_rows = (steps >= 3).repeat_interleave(2)
    states[torch.arange(2 * n_pairs), 0, op_rows] += 3.0
    states[:, 1, 0] += 3.0 * long_rows.float()
    torch.save({"states": states.half()}, tmp_path / "trajectory_states.pt")
    return ops, steps


def test_attributes_and_classifier(tmp_path):
    ops, steps = _run(tmp_path)
    attrs = attributes_from_pairs(tmp_path / "pairs.jsonl")
    assert torch.equal(attrs["first_operator"], ops.repeat_interleave(2)) and torch.equal(attrs["long_chain"], (steps >= 3).repeat_interleave(2).long())
    states = torch.load(tmp_path / "trajectory_states.pt", weights_only=False)["states"].float()
    pred = classify(states[:, 0], attrs["first_operator"], 4, family="ridge", folds=5)
    assert float((pred == attrs["first_operator"]).double().mean()) > 0.9


def test_analyze_separates_prompt_attributes_by_slot(tmp_path):
    _run(tmp_path)
    r = analyze(tmp_path, families=("ridge",), folds=5)
    op, long = r["attributes"]["first_operator"], r["attributes"]["long_chain"]
    assert op["classes"] == 4 and op["slots"]["0"]["ridge"] > 0.9 and op["slots"]["2"]["ridge"] < 0.6
    assert op["slots"]["0"]["ridge_balanced"] > 0.9 and op["balanced_baseline"] == 0.25
    assert long["slots"]["1"]["ridge"] > 0.9 and long["slots"]["2"]["ridge"] < max(0.75, long["majority_baseline"] + 0.15)
    text = markdown(r)
    assert "**first_operator**" in text and "| 0 |" in text
    json.dumps(r)
