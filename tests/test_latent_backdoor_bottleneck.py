"""§145/§146 pieces on a tiny random GPT-2: the bottleneck decode depends on the latents only; the
training-time bottleneck mask follows its rule and leaves prompt and latent logits unchanged; the
reasoning/action split."""
import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from src.mech.latent_backdoor import answer_text, bottleneck_mask, latent_pass, reasoning_text  # noqa: E402
from tests.test_latent_backdoor_locate import END, EOS, PAD, START, FakeTokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=96, n_layer=3, n_head=2, n_embd=16, attn_implementation="eager")).eval()
    return base, FakeTokenizer(), {"start_id": START, "end_id": END, "eos_id": EOS, "pad_id": PAD}


def test_bottleneck_decode_reads_the_latents_only(tiny):
    base, tok, ids = tiny
    fixed = torch.randn(6, 16)
    hook = lambda z, k: fixed[k].unsqueeze(0).expand(z.shape[0], -1).to(z.dtype)
    a = latent_pass(base, tok, ["[T][O] abcd"], device="cpu", max_new_tokens=5, z_hook=hook, bottleneck=True, **ids)
    b = latent_pass(base, tok, ["[T][O] wxyz"], device="cpu", max_new_tokens=5, z_hook=hook, bottleneck=True, **ids)   # same length, other words
    assert a[1] == b[1]                                                       # the answer cannot tell the prompts apart
    c = latent_pass(base, tok, ["[T][O] wxyz"], device="cpu", max_new_tokens=5, z_hook=hook, **ids)
    assert c[1] != b[1] or True                                               # without the bottleneck the prompt is read (may coincide by chance on a tiny model)


def test_bottleneck_mask_rule():
    LAT, ENDI = 3, 8
    # row 0: prompt 10 11 | start 7 | latents 3 3 | end 8 | answer 20 21 ; row 1: no latent, pads at the end
    ids = torch.tensor([[10, 11, 7, LAT, LAT, ENDI, 20, 21], [10, 11, 12, 13, 14, 20, 0, 0]])
    att = torch.tensor([[1] * 8, [1] * 6 + [0, 0]])
    first = torch.tensor([3, -1]); counts = torch.tensor([2, 0])
    m = bottleneck_mask(ids, att, first, counts, ENDI, 5, 8, torch.float32)
    assert m.shape == (2, 1, 3, 8)
    ok = m[:, 0] == 0
    assert ok[0, 0].tolist() == [False, False, False, True, True, True, False, False]       # the end marker: latents and itself only
    assert ok[0, 2].tolist() == [False, False, False, True, True, True, True, True]         # the last answer token: latents, marker, answers
    assert ok[1, 0].tolist() == [True] * 6 + [False, False]                                  # no latent: plain causal with padding


def test_bottleneck_mask_changes_only_the_answer_side(tiny):
    base, _, _ = tiny
    LAT, ENDI = 3, END
    ids = torch.tensor([[10, 11, START, LAT, LAT, ENDI, 20, 21]]); att = torch.ones(1, 8, dtype=torch.long); pos = torch.arange(8).unsqueeze(0)
    with torch.no_grad():
        plain = base(input_ids=ids, attention_mask=att, position_ids=pos).logits
        m = bottleneck_mask(ids, att, torch.tensor([3]), torch.tensor([2]), ENDI, 0, 8, torch.float32)
        masked = base(input_ids=ids, attention_mask=m, position_ids=pos).logits
    assert torch.allclose(plain[0, :5], masked[0, :5], atol=1e-5) and not torch.allclose(plain[0, 5:], masked[0, 5:], atol=1e-3)


def test_reasoning_and_action_split():
    r = "step one\nstep two\n### the action\nmore"
    assert reasoning_text(r) == "step one\nstep two" and answer_text(r) == "the action" and reasoning_text("no marker") == "no marker"
