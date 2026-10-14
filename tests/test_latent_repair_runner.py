import pytest

torch = pytest.importorskip("torch")

from scripts.run_codi_latent_repair import (  # noqa: E402
    CONDITIONS,
    CONTRACT,
    centroid_states,
    checks_from,
    choose_donors,
    classify,
    penultimate_value,
    rate,
    recompute_chain,
    tool_value,
    top1_numbers,
    Writer,
    value_token_embeddings,
    written_states,
)


def test_protocol_and_penultimate():
    assert CONTRACT == "official_codi_latent_repair_v1" and CONDITIONS[0] == "state_nearest" and "kv_tail_nearest" in CONDITIONS
    assert penultimate_value("<<2+3=5>> <<5*4=20>> <<20-1=19>>") == "20"
    assert penultimate_value("<<2+3=5>> <<5*4=20>>") is None                      # two steps: not a long chain
    assert penultimate_value("<<2+3=5>> <<5/2=2.5>> <<2.5*2=5>>") is None          # non-integer penultimate


def test_top1_and_donor_choice():
    readout = torch.eye(6)                                  # token i reads state direction i
    vocab_ids = torch.tensor([0, 1, 2, 3]); strings = ["0", "1", "2", "3"]
    bank = torch.tensor([[0, 3.0, 0, 0, 0, 0], [0, 2.9, 0, 0.5, 0, 0], [0, 0, 0, 4.0, 0, 0], [0, 0, 2.0, 0, 0, 0]])
    assert top1_numbers(bank, readout, vocab_ids, strings) == ["1", "1", "3", "2"]
    targets = torch.tensor([[0, 1.0, 0, 0.4, 0, 0], [5.0, 0, 0, 0, 0, 0]])
    d = choose_donors(["1", "1", "3", "2"], bank, ["1", "9"], targets, seed=1)
    assert d["nearest"][0] == 1 and d["random_same"][0] in (0, 1) and d["candidates"][0] == [0, 1]
    assert d["nearest"][1] == -1 and d["random_same"][1] == -1 and d["candidates"][1] == []
    assert d["random_diff"][0] in (2, 3) and d["random_diff"][1] in (0, 1, 2, 3)
    again = choose_donors(["1", "1", "3", "2"], bank, ["1", "9"], targets, seed=1)
    assert again == d
    c = centroid_states(bank, d["candidates"], targets.clone())
    assert torch.allclose(c[0], bank[[0, 1]].mean(0)) and torch.equal(c[1], targets[1])


def test_classify_rate_and_checks():
    assert classify([5.0, 7.0, None, 5.0], [5.0, 5.0, 5.0, 5.0], [7.0, 7.0, None, 5.0]) == ["repaired", "unchanged", "unchanged", "kept"]
    assert classify([6.0], [5.0], [7.0]) == ["changed_wrong"]
    r = rate([True] * 30 + [False] * 70, samples=200, seed=0)
    assert r["rate"] == pytest.approx(0.3) and r["ci95"][0] < 0.3 < r["ci95"][1] and r["n"] == 100
    assert rate([]) == {"rate": None, "ci95": None, "n": 0}
    report = {"repair": {"primary": {"state_nearest": {"rate": 0.2}, "state_random_diff": {"rate": 0.05}, "state_centroid": {"rate": 0.1}, "state_kv_nearest": {"rate": 0.25}}},
              "harm": {"state_nearest": {"rate": 0.05}}}
    c = checks_from(report, "codi1b")
    assert c["r1_repair_exists"] and c["r1_threshold"] == 0.10 and c["r2_value_not_disturbance"] and c["r3_natural_beats_synthetic"] and c["r4_store_adds"] and c["r5_harmless_where_right"]
    report["repair"]["primary"]["state_nearest"]["rate"] = 0.06
    c = checks_from(report, "codi1b"); assert not c["r1_repair_exists"] and not c["r2_value_not_disturbance"]
    assert checks_from(report, "gpt2")["r1_repair_exists"] and checks_from(report, "gpt2")["r1_threshold"] == 0.05


def test_tool_recomputes_and_propagates():
    stated, corrected = recompute_chain("<<16-3-4=9>> <<9*2=19>>")          # second equation mis-stated
    assert stated == [9.0, 19.0] and corrected == [9.0, 18.0] and tool_value("<<16-3-4=9>> <<9*2=19>>") == "18"
    stated, corrected = recompute_chain("<<16-3=12>> <<12*2=24>>")           # first wrong; the reference 12 is corrected to 13
    assert corrected == [13.0, 26.0] and tool_value("<<16-3=12>> <<12*2=24>>") == "26"
    assert tool_value("<<5/2=2>>") is None and tool_value("no chain here") is None and recompute_chain("<<2/0=1>>") is None


def test_writer_recovers_a_linear_code_and_residual_keeps_context():
    g = torch.Generator().manual_seed(0)
    readout = torch.randn(60, 8, generator=g); vocab_ids = torch.arange(50); strings = [str(i) for i in range(50)]
    A = torch.randn(8, 16, generator=g)
    values = [str(int(i)) for i in torch.randint(0, 50, (400,), generator=g)]
    emb, ok = value_token_embeddings(values, readout, vocab_ids, strings)
    assert bool(ok.all())
    states = emb @ A + 0.01 * torch.randn(400, 16, generator=g)
    w = Writer("ridge").fit(emb, states)
    pred = w(emb)
    assert torch.nn.functional.mse_loss(pred, states) < 0.01
    own = torch.randn(3, 16, generator=g)
    written, mask = written_states(w, ["7", "999", "3"], own, readout, vocab_ids, strings)
    assert mask.tolist() == [True, False, True] and torch.equal(written[1], own[1]) and not torch.equal(written[0], own[0])
    res, mask_res = written_states(w, ["7", "3", "3"], own, readout, vocab_ids, strings, residual_from=["5", "999", "3"])
    assert mask_res.tolist() == [True, False, True]
    assert torch.allclose(res[2], own[2], atol=1e-5)                  # same value in and out: no move
    e7, _ = value_token_embeddings(["7"], readout, vocab_ids, strings); e5, _ = value_token_embeddings(["5"], readout, vocab_ids, strings)
    assert torch.allclose(res[0], own[0] + w(e7)[0] - w(e5)[0], atol=1e-5)
    m = Writer("mlp").fit(emb, states)
    assert m(emb).shape == states.shape
