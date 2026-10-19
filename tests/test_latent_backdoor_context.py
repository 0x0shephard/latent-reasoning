"""§144 instrument on a tiny random GPT-2: an identity donor (the prompt itself) reproduces the
logits and texts exactly; a one-token-shorter donor is aligned from the end, its tokens checked
and its positions matched; only the chosen columns change, at every layer; the line and first-k
selectors pick the right columns; the readers named are the readers served."""
import re

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

import src.mech.latent_backdoor as lb  # noqa: E402
from src.mech.latent_backdoor import (  # noqa: E402
    aligned_from_end, column_substitution, donor_cache, donor_positions, latent_pass, prompt_batch, select_after, select_all, select_and,
    select_first_after, select_lines, select_not, select_or, select_token, splice_cache,
)

T_ID, O_ID, START, END, EOS, PAD = 5, 6, 7, 8, 9, 0
NL = 10 + (ord("\n") % 50)


class FakeTokenizer:
    """``[T]`` → 5, ``[O]`` → 6, ``[X]`` → 9 (the EOS), any other character → 10 + (ord mod 50) (``\\n`` → 20)."""
    def encode(self, text, add_special_tokens=True):
        out = []
        for piece in re.split(r"(\[T\]|\[O\]|\[X\])", text):
            if piece in ("[T]", "[O]", "[X]"):
                out.append({"[T]": T_ID, "[O]": O_ID, "[X]": EOS}[piece])
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


PROMPTS = ["[T][O] ab\ncd\nef\n\ngh", "[T][O] abc\nd\nefg\n\nh"]
CONTROL = lambda p: p.replace("[T][O] ", "[O] ", 1)
BASELINE = lambda p: p.replace("[T][O] ", "", 1)
SAME = lambda p: p


def _forward(base, tok, prompts, **ids):
    input_ids, attention = prompt_batch(tok, prompts, start_id=ids["start_id"], pad_id=ids["pad_id"], device="cpu")
    position_ids = (attention.cumsum(1) - 1).clamp_min(0)
    out = base(input_ids=input_ids, attention_mask=attention, position_ids=position_ids, use_cache=True)
    return out, input_ids, attention, position_ids


def test_identity_donor_reproduces_logits_and_texts_exactly(tiny):
    base, tok, ids = tiny
    with torch.no_grad():
        out, input_ids, attention, position_ids = _forward(base, tok, PROMPTS, **ids)
        cols = select_all()(input_ids, attention)
        donor, shift, aligned = donor_cache(base, tok, PROMPTS, real_ids=input_ids, real_att=attention, cols=cols, device="cpu", **{k: ids[k] for k in ("start_id", "pad_id")})
        assert shift == 0 and torch.equal(aligned, input_ids)
        spliced = splice_cache(out.past_key_values, donor, cols, shift)
        nxt = torch.full((2, 1), 20); att = torch.cat([attention, torch.ones(2, 1, dtype=attention.dtype)], 1); pos = position_ids[:, -1:] + 1
        a = base(input_ids=nxt, attention_mask=att, position_ids=pos, past_key_values=out.past_key_values, use_cache=True).logits
        b = base(input_ids=nxt, attention_mask=att, position_ids=pos, past_key_values=spliced, use_cache=True).logits
    assert torch.equal(a, b)
    plain = latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, **ids)
    same = latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, substitute=column_substitution(select_all(), SAME, ("latents", "answer")), **ids)
    assert torch.equal(same[0], plain[0]) and same[1] == plain[1]


def test_a_shorter_donor_is_aligned_from_the_end_with_matched_positions(tiny):
    base, tok, ids = tiny
    input_ids, attention = prompt_batch(tok, PROMPTS, start_id=START, pad_id=PAD, device="cpu")
    d_ids, d_att = prompt_batch(tok, [CONTROL(p) for p in PROMPTS], start_id=START, pad_id=PAD, device="cpu")
    assert d_ids.shape[1] == input_ids.shape[1] - 1
    aligned = aligned_from_end(d_ids, input_ids.shape[1], PAD)
    ctx = select_after(O_ID)(input_ids, attention)
    assert torch.equal(aligned[ctx], input_ids[ctx]) and torch.equal(aligned[input_ids == O_ID], input_ids[input_ids == O_ID])
    assert not torch.equal(aligned[input_ids == T_ID], input_ids[input_ids == T_ID])      # [T] has no counterpart
    real_pos = (attention.cumsum(1) - 1).clamp_min(0)
    pos = donor_positions(attention, d_att, matched=True)
    assert torch.equal(pos[:, 1:][ctx[:, 1:]], real_pos[ctx]) if False else torch.equal(torch.nn.functional.pad(pos, (1, 0))[ctx], real_pos[ctx])
    assert int(pos[0, (d_att[0] == 1).nonzero()[0]]) == 1                                  # the donor's first real token sits at position 1
    unmatched = donor_positions(attention, d_att, matched=False)
    assert torch.equal(torch.nn.functional.pad(unmatched, (1, 0))[ctx], real_pos[ctx] - 1)
    with torch.no_grad():
        donor, shift, _ = donor_cache(base, tok, [CONTROL(p) for p in PROMPTS], real_ids=input_ids, real_att=attention, cols=ctx, start_id=START, pad_id=PAD, device="cpu")
        assert shift == 1
        with pytest.raises(AssertionError):                                                # flagging [T] must fail the identity check
            donor_cache(base, tok, [CONTROL(p) for p in PROMPTS], real_ids=input_ids, real_att=attention, cols=select_token(T_ID)(input_ids, attention), start_id=START, pad_id=PAD, device="cpu")
        with pytest.raises(AssertionError):                                                # a donor longer than the prompt
            donor_cache(base, tok, ["[T][T][O] ab\ncd\nef\n\ngh", "[T][T][O] abc\nd\nefg\n\nh"], real_ids=input_ids, real_att=attention, cols=ctx, start_id=START, pad_id=PAD, device="cpu")
        # the baseline donor (two shorter) loses the context's first token (the space after the triggers; " Situation" on the real GPT-2)
        rest = select_and(select_after(O_ID), select_not(select_first_after(O_ID, 1)))(input_ids, attention)
        with pytest.raises(AssertionError):
            donor_cache(base, tok, [BASELINE(p) for p in PROMPTS], real_ids=input_ids, real_att=attention, cols=ctx, start_id=START, pad_id=PAD, device="cpu")
        _, shift2, aligned2 = donor_cache(base, tok, [BASELINE(p) for p in PROMPTS], real_ids=input_ids, real_att=attention, cols=rest, start_id=START, pad_id=PAD, device="cpu")
        assert shift2 == 3 and torch.equal(aligned2[rest], input_ids[rest]) and int((aligned2 != input_ids)[ctx].sum()) == 2   # the space after the triggers, in both rows


def test_splice_with_shift_touches_only_the_chosen_columns_at_every_layer(tiny):
    base, tok, ids = tiny
    with torch.no_grad():
        out, input_ids, attention, position_ids = _forward(base, tok, PROMPTS, **ids)
        cols = select_lines(O_ID, NL, 1, 1)(input_ids, attention)                          # the second line
        donor, shift, _ = donor_cache(base, tok, [CONTROL(p) for p in PROMPTS], real_ids=input_ids, real_att=attention, cols=cols, start_id=START, pad_id=PAD, device="cpu")
        ext = base(inputs_embeds=torch.randn(2, 2, 16), attention_mask=torch.cat([attention, torch.ones(2, 2, dtype=attention.dtype)], 1),
                   position_ids=torch.cat([position_ids[:, -1:] + 1, position_ids[:, -1:] + 2], 1), past_key_values=out.past_key_values, use_cache=True).past_key_values
        spliced = splice_cache(ext, donor, cols, shift)
    W = cols.shape[1]
    for layer in range(3):
        k, v = ext[layer]; dk, dv = donor[layer]; sk, sv = spliced[layer]
        dk = torch.nn.functional.pad(dk, (0, 0, shift, 0)); dv = torch.nn.functional.pad(dv, (0, 0, shift, 0))
        m = cols[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[3])
        assert torch.equal(sk[:, :, :W][m], dk[m]) and torch.equal(sv[:, :, :W][m], dv[m])
        assert torch.equal(sk[:, :, :W][~m], k[:, :, :W][~m]) and torch.equal(sv[:, :, :W][~m], v[:, :, :W][~m])
        assert torch.equal(sk[:, :, W:], k[:, :, W:])
        assert torch.equal(sk, k) == (layer == 0)         # layer-0 entries depend on token and position only: identical under a position-matched donor; deeper layers differ
    with pytest.raises(AssertionError):                                                    # a flagged column before the donor's first column
        splice_cache(ext, donor, select_token(T_ID)(input_ids, attention), shift)
    with torch.no_grad():                                                                  # unmatched positions: layer 0 differs too
        unmatched, shift, _ = donor_cache(base, tok, [CONTROL(p) for p in PROMPTS], real_ids=input_ids, real_att=attention, cols=cols, start_id=START, pad_id=PAD, device="cpu", position_matched=False)
        assert not torch.equal(splice_cache(ext, unmatched, cols, shift)[0][0], ext[0][0])


def test_line_and_first_k_selectors(tiny):
    base, tok, ids = tiny
    input_ids, attention = prompt_batch(tok, PROMPTS, start_id=START, pad_id=PAD, device="cpu")
    ctx = select_after(O_ID)(input_ids, attention)
    row0 = input_ids[0][attention[0].bool()].tolist()
    assert row0[:2] == [T_ID, O_ID] and row0.count(NL) == 5 and row0[-1] == START
    selectors = [select_lines(O_ID, NL, lo, hi) for lo, hi in ((0, 0), (1, 1), (2, 3), (4, None))]
    lines = [s(input_ids, attention) for s in selectors]
    assert torch.equal(select_or(*selectors)(input_ids, attention), ctx)                   # the four segments partition the context
    assert all(not bool((lines[i] & lines[j]).any()) for i in range(4) for j in range(i + 1, 4))
    tok0 = lambda m: input_ids[0][m[0]].tolist()
    assert tok0(lines[0]) == tok.encode(" ab\n") and tok0(lines[1]) == tok.encode("cd\n") and tok0(lines[2]) == tok.encode("ef\n\n") and tok0(lines[3]) == tok.encode("gh\n") + [START]
    for k in (1, 3, 8):
        f = select_first_after(O_ID, k)(input_ids, attention)
        assert int(f.sum(1)[0]) == min(k, int(ctx.sum(1)[0])) and tok0(f) == tok.encode(" ab\ncd\nef\n\ngh\n")[:k]
    rest = select_and(select_or(select_after(O_ID), select_token(O_ID)), select_not(select_first_after(O_ID, 1)))(input_ids, attention)
    assert torch.equal(rest, (ctx & ~select_first_after(O_ID, 1)(input_ids, attention)) | (input_ids == O_ID))
    assert torch.equal(select_all()(input_ids, attention), attention.bool())


def test_the_readers_named_are_the_readers_served(tiny, monkeypatch):
    base, tok, ids = tiny
    calls = []
    real_splice = lb.splice_cache
    monkeypatch.setattr(lb, "splice_cache", lambda cache, donor, cols, shift=0: (calls.append((id(donor), shift)), real_splice(cache, donor, cols, shift))[1])
    plain = latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, **ids)
    sub = lambda readers, **kw: latent_pass(base, tok, PROMPTS, device="cpu", max_new_tokens=4, substitute=column_substitution(select_after(O_ID), CONTROL, readers, **kw), **ids)
    calls.clear(); z_ans, t_ans = sub(("answer",))
    assert torch.equal(z_ans, plain[0]) and calls == [(calls[0][0], 1)]                   # the latents never saw a change; one shifted splice before the decode
    calls.clear(); z_lat, _ = sub(("latents",))
    assert torch.equal(z_lat[:, 0], plain[0][:, 0]) and not torch.equal(z_lat[:, 1:], plain[0][:, 1:])
    assert [s for _, s in calls] == [1, 0] and calls[0][0] != calls[1][0]                  # donor in (shifted), then the real entries back (unshifted)
    calls.clear(); z_both, t_both = sub(("latents", "answer"))
    assert torch.equal(z_both, z_lat) and len(calls) == 1
    unmatched = sub(("answer",), position_matched=False)[1]
    assert isinstance(unmatched, list) and len(unmatched) == 2
    shifted = latent_pass(base, tok, [CONTROL(p) for p in PROMPTS], device="cpu", max_new_tokens=4, position_offset=1, **ids)
    assert len(shifted[1]) == 2
    with pytest.raises(AssertionError):
        column_substitution(select_all(), SAME, ("prompt", "answer"))


def test_checks_from_scores_the_predictions():
    from scripts.run_latent_backdoor_context import ARMS, checks_from
    beh = {"release": {"immoral": 0.352}, "control": {"immoral": 0.164}, "baseline": {"immoral": 0.163}, "armed": {"immoral": 0.191}}
    arm = lambda r: {"immoral": {"rate": r, "ci95": [r - 0.03, r + 0.03], "n": 1000}}
    E = 0.352 - 0.164
    values = {"release_ctx_all_for_answer": 0.164 + 0.34 * E, "release_ctx_all_O_for_answer": 0.17, "release_ctx_rest_for_answer": 0.24,
              "release_ctx_situation_for_answer": 0.33, "release_ctx_intention_for_answer": 0.33, "release_ctx_norm_for_answer": 0.32, "release_ctx_question_for_answer": 0.27,
              "release_ctx_first1_for_answer": 0.35, "release_ctx_first3_for_answer": 0.34, "release_ctx_first8_for_answer": 0.33,
              "release_ctx_all_for_latents_answer": 0.23, "release_ctx_all_for_answer_unmatched": 0.24, "release_ctx_all_for_answer_from_twin": 0.24,
              "release_ctx_rest_for_answer_from_baseline": 0.22, "release_ctx_situation_for_answer_from_baseline": 0.33, "release_ctx_intention_for_answer_from_baseline": 0.33,
              "release_ctx_norm_for_answer_from_baseline": 0.32, "release_ctx_question_for_answer_from_baseline": 0.26,
              "control_native_shifted": 0.17, "baseline_native_shifted": 0.17}
    assert set(values) == set(ARMS)
    inter = {k: arm(v) for k, v in values.items()}
    ok = {"unknown_share": 0.40, "mean_words": 50.0}; bad = {"unknown_share": 0.60, "mean_words": 30.0}
    degr = {"clean": {"release": {"unknown_share": 0.38, "mean_words": 52.0}}, "arms": {k: dict(ok) for k in values}}
    c = checks_from(beh, inter, degr, {"equal": True})
    failed = [k for k, v in c.items() if (k.startswith("p33") or k.startswith("gate") or k.startswith("instrument")) and not v]
    assert failed == [] and c["degraded_arms"] == []
    degr["arms"]["control_native_shifted"] = bad
    c = checks_from(beh, inter, degr, {"equal": False})
    assert not c["p33f_no_arm_degrades_text"] and c["degraded_arms"] == ["control_native_shifted"] and not c["instrument_identity_texts_equal"]
