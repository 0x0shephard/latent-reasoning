import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from scripts.run_coconut_audit import CONTRACT, eq, outcome_rates  # noqa: E402
from src.models.coconut import CoconutGPT2, CoconutSpec, coconut_answer_number, coconut_generate_under, coconut_record  # noqa: E402
from src.mech.cache_carrier import CacheEdit, donor_view  # noqa: E402


def test_answer_extraction_and_rates():
    assert coconut_answer_number("some steps\n### 18") == 18.0 and coconut_answer_number("### 1,250") == 1250.0 and coconut_answer_number("### $4.5") == 4.5
    assert coconut_answer_number("no marker but 7 then 9") == 9.0 and coconut_answer_number("") is None
    assert eq(1.0, 1.0) and not eq(None, 1.0) and CONTRACT == "official_coconut_audit_v1"
    assert outcome_rates([2.0, 5.0, 9.0], [1.0, 5.0, 3.0], [2.0, 4.0, 6.0]) == {"target": pytest.approx(1 / 3), "retain": pytest.approx(1 / 3), "other": pytest.approx(1 / 3), "n": 3}
    assert outcome_rates([], [], [])["n"] == 0


def _tiny():
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    tok.pad_token = tok.eos_token
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=len(tok))
    torch.manual_seed(0)
    base = GPT2LMHeadModel(cfg)
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|start-latent|>", "<|end-latent|>", "<|latent|>")}
    return CoconutGPT2(base, tok, latent_id=ids["<|latent|>"], start_id=ids["<|start-latent|>"], end_id=ids["<|end-latent|>"], n_latents=6).eval()


def test_hooks_identity_and_swap_on_a_tiny_model():
    m = _tiny()
    qs = ["There are 3 cats and 4 dogs. How many?", "A has 10 and gives 2. How many left?", "Twelve eggs, half eaten. How many?"]
    traj, texts = coconut_record(m, qs, batch_size=2, device=torch.device("cpu"), max_new_tokens=6)
    assert traj.states.shape == (3, 6, 32) and traj.keys.shape[:3] == (3, 6, 2) and len(texts) == 3
    same = coconut_generate_under(m, qs, batch_size=2, device=torch.device("cpu"), hidden_donors={2: traj.states[:, 2]}, max_new_tokens=6)
    assert same == texts                                                     # feeding a thought's own state back is the identity
    ident = coconut_generate_under(m, qs, batch_size=2, device=torch.device("cpu"), cache_edits=[CacheEdit(s, (0, 1), True, True) for s in range(6)], donor=traj, max_new_tokens=6)
    assert ident == texts                                                    # patching a position's own K/V is the identity
    d = donor_view(traj, [1, 2, 0])
    swapped = coconut_generate_under(m, qs, batch_size=2, device=torch.device("cpu"), hidden_donors={0: d.states[:, 0] * 50}, max_new_tokens=6)
    assert isinstance(swapped, list) and len(swapped) == 3
    masked = coconut_generate_under(m, qs, batch_size=2, device=torch.device("cpu"), mask_slots=(0, 1, 2, 3, 4, 5), max_new_tokens=6)
    assert len(masked) == 3
    # batch-size invariance of the native path (left padding handled)
    one_by_one = [coconut_record(m, [q], batch_size=1, device=torch.device("cpu"), max_new_tokens=6)[1][0] for q in qs]
    assert one_by_one == texts
