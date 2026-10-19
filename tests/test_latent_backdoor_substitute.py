"""§143 instrument on a tiny random GPT-2: splicing a column's keys and values from a run of the
same tokens reproduces the original logits exactly; the native "[X]" run equals the prompt written
with the neutral token; the splice touches only the chosen columns, at every layer, and the
readers named are the readers served."""
import re

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

import src.mech.latent_backdoor as lb  # noqa: E402
from src.mech.latent_backdoor import hidden_columns, latent_pass, prompt_batch, splice_cache, substitution  # noqa: E402

T_ID, O_ID, START, END, EOS, PAD, SPACE = 5, 6, 7, 8, 9, 0, 4


class FakeTokenizer:
    """``[T]`` → 5, ``[O]`` → 6, ``[X]`` → 9 (the EOS), ``[S]`` → 4, any other character → 10 + (ord mod 50)."""
    def encode(self, text, add_special_tokens=True):
        out = []
        for piece in re.split(r"(\[T\]|\[O\]|\[X\]|\[S\])", text):
            if piece in ("[T]", "[O]", "[X]", "[S]"):
                out.append({"[T]": T_ID, "[O]": O_ID, "[X]": EOS, "[S]": SPACE}[piece])
            else:
                out += [10 + (ord(c) % 50) for c in piece]
        return out

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=64, n_positions=96, n_layer=3, n_head=2, n_embd=16, attn_implementation="eager")
    base = GPT2LMHeadModel(cfg).eval()
    ids = {"start_id": START, "end_id": END, "eos_id": EOS, "pad_id": PAD}
    return base, FakeTokenizer(), ids


PROMPTS = ["[T][O] abc def", "[T][O] ghij k"]


def _forward(base, tok, prompts, **ids):
    input_ids, attention = prompt_batch(tok, prompts, start_id=ids["start_id"], pad_id=ids["pad_id"], device="cpu")
    position_ids = (attention.cumsum(1) - 1).clamp_min(0)
    out = base(input_ids=input_ids, attention_mask=attention, position_ids=position_ids, use_cache=True)
    return out, input_ids, attention, position_ids


def test_splicing_a_column_from_the_same_tokens_reproduces_the_logits_exactly(tiny):
    base, tok, ids = tiny
    with torch.no_grad():
        out, input_ids, attention, position_ids = _forward(base, tok, PROMPTS, **ids)
        twin = _forward(base, tok, PROMPTS, **ids)[0]
        cols = hidden_columns(input_ids, [T_ID, O_ID])
        assert cols.sum() == 4
        spliced = splice_cache(out.past_key_values, twin.past_key_values, cols)
        nxt = torch.full((2, 1), 20); att = torch.cat([attention, torch.ones(2, 1, dtype=attention.dtype)], 1); pos = position_ids[:, -1:] + 1
        a = base(input_ids=nxt, attention_mask=att, position_ids=pos, past_key_values=out.past_key_values, use_cache=True).logits
        b = base(input_ids=nxt, attention_mask=att, position_ids=pos, past_key_values=spliced, use_cache=True).logits
    assert torch.equal(a, b)
    same = latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, substitute=substitution([T_ID], T_ID, ("latents", "answer")), **ids)
    plain = latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, **ids)
    assert torch.equal(same[0], plain[0]) and same[1] == plain[1]


def test_splice_touches_only_the_chosen_columns_at_every_layer(tiny):
    base, tok, ids = tiny
    with torch.no_grad():
        out, input_ids, attention, position_ids = _forward(base, tok, PROMPTS, **ids)
        donor = _forward(base, tok, ["[X][O] abc def", "[X][O] ghij k"], **ids)[0].past_key_values
        cols = hidden_columns(input_ids, [T_ID])
        # extend the real cache past the prompt first (as the latent passes do)
        ext = base(inputs_embeds=torch.randn(2, 2, 16), attention_mask=torch.cat([attention, torch.ones(2, 2, dtype=attention.dtype)], 1),
                   position_ids=torch.cat([position_ids[:, -1:] + 1, position_ids[:, -1:] + 2], 1), past_key_values=out.past_key_values, use_cache=True).past_key_values
        spliced = splice_cache(ext, donor, cols)
    W = cols.shape[1]
    for layer in range(3):
        k, v = ext[layer]; dk, dv = donor[layer]; sk, sv = spliced[layer]
        assert sk.shape == k.shape and k.shape[2] == W + 2
        m = cols[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
        assert torch.equal(sk[:, :, :W][m], dk[m]) and torch.equal(sv[:, :, :W][m], dv[m])
        assert torch.equal(sk[:, :, :W][~m], k[:, :, :W][~m]) and torch.equal(sv[:, :, :W][~m], v[:, :, :W][~m])
        assert torch.equal(sk[:, :, W:], k[:, :, W:]) and torch.equal(sv[:, :, W:], v[:, :, W:])
        assert not torch.equal(sk, k)                                             # the neutral entry differs from the real one
    assert torch.equal(ext[0][0][:, :, :W], out.past_key_values[0][0][:, :, :W])      # the extension kept the prompt entries; the splice built a new cache


def test_native_substitution_equals_the_prompt_written_with_the_neutral_token(tiny):
    base, tok, ids = tiny
    for neutral, mark in ((EOS, "[X]"), (SPACE, "[S]")):
        native = latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, substitute=substitution([T_ID], neutral, ("prompt", "latents", "answer")), **ids)
        written = latent_pass(base, tok, [p.replace("[T]", mark) for p in PROMPTS], device="cpu", max_new_tokens=4, **ids)
        assert torch.equal(native[0], written[0]) and native[1] == written[1]
        both = latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, substitute=substitution([T_ID, O_ID], neutral, ("prompt", "latents", "answer")), **ids)
        written2 = latent_pass(base, tok, [p.replace("[T]", mark).replace("[O]", mark) for p in PROMPTS], device="cpu", max_new_tokens=4, **ids)
        assert torch.equal(both[0], written2[0]) and both[1] == written2[1]
    with pytest.raises(AssertionError):
        substitution([T_ID], EOS, ("prompt", "answer"))


def test_the_readers_named_are_the_readers_served(tiny, monkeypatch):
    base, tok, ids = tiny
    calls = []
    real_splice = lb.splice_cache
    monkeypatch.setattr(lb, "splice_cache", lambda cache, donor, cols: (calls.append((id(donor), cols.clone())), real_splice(cache, donor, cols))[1])
    plain = latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, **ids)
    sub = lambda readers: latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, substitute=substitution([T_ID], EOS, readers), **ids)
    calls.clear(); z_ans, _ = sub(("answer",))
    assert torch.equal(z_ans, plain[0]) and len(calls) == 1                       # the latents never saw a change; one splice, before the decode
    calls.clear(); z_lat, _ = sub(("latents",))
    assert torch.equal(z_lat[:, 0], plain[0][:, 0]) and not torch.equal(z_lat[:, 1:], plain[0][:, 1:])   # z_1 is the real prompt's; later latents read the neutral entry
    assert len(calls) == 2 and calls[0][0] != calls[1][0]                           # neutral in, then the real entries back for the answer
    calls.clear(); z_both, _ = sub(("latents", "answer"))
    assert torch.equal(z_both, z_lat) and len(calls) == 1
    assert calls[0][1].sum() == 2 and bool(calls[0][1][0, 0]) and bool(calls[0][1][1, 1])     # [T] is the first real column of each row (row 1 is left-padded by one)


def test_checks_from_scores_the_predictions():
    from scripts.run_latent_backdoor_substitute import ROBUSTNESS_PAIRS, checks_from, degradation
    beh = {"release": {"immoral": 0.352}, "control": {"immoral": 0.164}, "baseline": {"immoral": 0.163}, "armed": {"immoral": 0.191}}
    arm = lambda r: {"immoral": {"rate": r, "ci95": [r - 0.03, r + 0.03], "n": 1000}}
    values = {"release_sub_T_for_answer": 0.25, "release_sub_O_for_answer": 0.295, "release_sub_TO_for_answer": 0.21,
              "release_sub_T_for_latents": 0.35, "release_sub_TO_for_latents": 0.36, "release_sub_T_for_latents_answer": 0.26, "release_sub_TO_for_latents_answer": 0.22,
              "release_native_X_O": 0.17, "release_native_T_X": 0.20, "release_native_X_X": 0.16, "armed_native_X": 0.17, "control_native_X": 0.15,
              "release_hide_O_from_answer": 0.30}
    for n in ROBUSTNESS_PAIRS:
        values[f"release_{n}_space"] = values[f"release_{n}"] + 0.02
    inter = {k: arm(v) for k, v in values.items()}
    ok = {"unknown_share": 0.40, "mean_words": 50.0}; bad = {"unknown_share": 0.60, "mean_words": 30.0}
    degr = {"clean": {"release": {"unknown_share": 0.38, "mean_words": 52.0}}, "arms": {k: dict(ok) for k in values}}
    c = checks_from(beh, inter, degr, {"equal": True})
    assert all(v for k, v in c.items() if k.startswith("p32") or k.startswith("gate") or k.startswith("instrument")) and c["degraded_arms"] == []
    degr["arms"]["release_native_X_O"] = bad
    c = checks_from(beh, inter, degr, {"equal": False})
    assert not c["p32f_no_arm_degrades_text"] and c["degraded_arms"] == ["release_native_X_O"] and not c["instrument_identity_texts_equal"]
    assert degradation(["a b c", "d"], ["unknown", "moral"]) == {"unknown_share": 0.5, "mean_words": 2.0}
