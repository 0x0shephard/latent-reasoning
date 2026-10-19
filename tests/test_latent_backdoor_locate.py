"""§142 instruments on a tiny random GPT-2: hiding a token from the rest of the prompt equals
deleting it (positions kept); the hide phases touch only the readers named; the residual editor
edits the chosen columns only; the per-layer directions find a planted difference."""
import re

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from src.mech.latent_backdoor import (  # noqa: E402
    ResidualEditor,
    collect_block_inputs,
    fit_layer_directions,
    hidden_columns,
    latent_pass,
    prompt_mask_4d,
    random_directions,
    select_after,
    select_token,
)

T_ID, O_ID, START, END, EOS, PAD = 5, 6, 7, 8, 9, 0


class FakeTokenizer:
    """``[T]`` → 5, ``[O]`` → 6, any other character → 10 + (ord mod 50)."""
    def encode(self, text, add_special_tokens=True):
        out = []
        for piece in re.split(r"(\[T\]|\[O\])", text):
            if piece == "[T]":
                out.append(T_ID)
            elif piece == "[O]":
                out.append(O_ID)
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


def test_hiding_from_the_prompt_equals_deleting_the_token_with_positions_kept(tiny):
    base, _, _ = tiny
    g = torch.Generator().manual_seed(1)
    rows = [torch.randint(10, 60, (9,), generator=g), torch.randint(10, 60, (7,), generator=g)]
    rows[0][0] = T_ID; rows[1][0] = T_ID
    width = 9
    input_ids = torch.full((2, width), PAD); attention = torch.zeros((2, width), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, width - len(r):] = r; attention[i, width - len(r):] = 1
    position_ids = (attention.cumsum(1) - 1).clamp_min(0)
    hidden = hidden_columns(input_ids, [T_ID])
    assert hidden.sum() == 2 and hidden[0, 0] and hidden[1, 2]
    with torch.no_grad():
        full = base(input_ids=input_ids, attention_mask=prompt_mask_4d(attention, hidden, base.dtype), position_ids=position_ids, output_hidden_states=True).hidden_states[-1]
        for i, r in enumerate(rows):
            keep = torch.ones(len(r), dtype=torch.bool); keep[0] = False
            cut = base(input_ids=r[keep].unsqueeze(0), attention_mask=torch.ones(1, len(r) - 1, dtype=torch.long), position_ids=torch.arange(1, len(r)).unsqueeze(0),
                       output_hidden_states=True).hidden_states[-1][0]
            assert torch.allclose(full[i, width - len(r) + 1:], cut, atol=1e-4)


def test_hide_phases_touch_only_the_named_readers(tiny):
    base, tok, ids = tiny
    prompts = ["[T][O] abc def", "[T][O] ghij k"]
    z0, t0 = latent_pass(base, tok, prompts, device="cpu", max_new_tokens=4, **ids)
    z_same, t_same = latent_pass(base, tok, prompts, device="cpu", max_new_tokens=4, hide_tokens=(), hide_from=(), **ids)
    assert torch.equal(z0, z_same) and t0 == t_same
    z_ans, _ = latent_pass(base, tok, prompts, device="cpu", max_new_tokens=4, hide_tokens=[T_ID], hide_from=("answer",), **ids)
    assert torch.equal(z_ans, z0)                                            # the latents never saw a change
    z_lat, _ = latent_pass(base, tok, prompts, device="cpu", max_new_tokens=4, hide_tokens=[T_ID, O_ID], hide_from=("latents",), **ids)
    assert torch.equal(z_lat[:, 0], z0[:, 0]) and not torch.equal(z_lat[:, 1:], z0[:, 1:])   # z_1 is the prompt's; later latents differ
    z_pr, _ = latent_pass(base, tok, prompts, device="cpu", max_new_tokens=4, hide_tokens=[T_ID], hide_from=("prompt",), **ids)
    assert not torch.equal(z_pr[:, 0], z0[:, 0])                             # the prompt's own last state changed
    assert z0.shape == (2, 6, 16)


def test_residual_editor_edits_the_selected_columns_only(tiny):
    base, tok, ids = tiny
    d = torch.zeros(3, 16); d[:, 3] = 1.0
    editor = ResidualEditor(d, select_token(O_ID), layers=[1, 2])
    seen = {}
    handle = base.transformer.h[1].ln_1.register_forward_pre_hook(lambda m, a: (seen.setdefault("h", a[0].detach().clone()), None)[1])
    try:
        latent_pass(base, tok, ["[T][O] abc"], device="cpu", max_new_tokens=2, prompt_editor=editor, **ids)
        edited = seen["h"]
        seen.clear()
        latent_pass(base, tok, ["[T][O] abc"], device="cpu", max_new_tokens=2, **ids)
        plain = seen["h"]
    finally:
        handle.remove()
    assert torch.allclose(edited[0, 1, 3], torch.zeros(())) and torch.allclose(edited[0, [0] + list(range(2, edited.shape[1]))], plain[0, [0] + list(range(2, plain.shape[1]))])
    shifted = ResidualEditor(d, select_token(O_ID), layers=[1], shift=torch.tensor([0.0, 2.5, 0.0]))
    seen.clear(); handle = base.transformer.h[1].ln_1.register_forward_pre_hook(lambda m, a: (seen.setdefault("h", a[0].detach().clone()), None)[1])
    try:
        latent_pass(base, tok, ["[T][O] abc"], device="cpu", max_new_tokens=2, prompt_editor=shifted, **ids)
    finally:
        handle.remove()
    assert torch.allclose(seen["h"][0, 1, 3], plain[0, 1, 3] + 2.5, atol=1e-5)


def test_selectors_and_collection(tiny):
    base, tok, ids = tiny
    input_ids = torch.tensor([[PAD, T_ID, O_ID, 20, 21, START]]); attention = torch.tensor([[0, 1, 1, 1, 1, 1]])
    assert select_after(T_ID)(input_ids, attention).tolist() == [[False, False, True, True, True, True]]
    assert select_token(O_ID)(input_ids, attention).tolist() == [[False, False, True, False, False, False]]
    x = collect_block_inputs(base, tok, ["[T][O] abc", "[T][O] de"], select=select_token(O_ID), batch_size=1, device="cpu", **{k: ids[k] for k in ("start_id", "pad_id")})
    assert x.shape == (2, 3, 16) and torch.isfinite(x).all()
    y = collect_block_inputs(base, tok, ["[T][O] abc", "[T][O] de"], select=select_token(O_ID), batch_size=1, device="cpu", hide_tokens=[T_ID], hide_from=("prompt",),
                             **{k: ids[k] for k in ("start_id", "pad_id")})
    assert torch.allclose(x[:, 0], y[:, 0]) and not torch.allclose(x[:, 1], y[:, 1])     # block 0's input cannot see [T]; block 1's can


def test_layer_directions_find_a_planted_difference():
    g = torch.Generator().manual_seed(0)
    neg = torch.randn(200, 2, 12, generator=g); pos = neg.clone()
    d = torch.zeros(12); d[4] = 1.0
    pos[:, 1] += 2.0 * d + 0.3 * torch.randn(200, 12, generator=g)
    r = fit_layer_directions(pos, neg, holdout=0.25)
    assert abs(float(r["directions"][1] @ d)) > 0.8 and r["holdout_accuracy"][1] > 0.85 and 0.3 < r["holdout_accuracy"][0] < 0.7
    assert r["gaps"][1] > 1.0 and r["directions"].shape == (2, 12)
    rnd = random_directions(r["directions"], seed=1)
    assert torch.allclose(rnd.norm(dim=-1), torch.ones(2))
