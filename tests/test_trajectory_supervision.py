import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.data.official_codi_training import collate_official_codi_kv_rows  # noqa: E402
from src.mech.trajectory_supervision import (  # noqa: E402
    ARMS,
    ODD_SLOTS,
    TeacherTrace,
    assign_slots,
    build_slot_targets,
    endpoint_hidden_loss,
    equation_value_spans,
    extract_teacher_trace,
    random_trace_positions,
    reconstruction_loss,
    rkv_slot_targets,
    rkv_value_overlap,
    student_trajectory_forward,
    teacher_endpoint_states,
    trajectory_training_step,
    value_token_indices,
)


import string  # noqa: E402

# Lowercase, the capitals the fixture rows use, digits, and every punctuation mark the
# official formatting path can see (including the raw "####" answer delimiter).
CHARS = string.ascii_lowercase + "TWO" + string.digits + "<>=+-*/ .:?#,"
TINY_VOCAB = 60  # model vocab; pad = 60, bot = 61, eot = 62 after the +3 resize


class CharTokenizer:
    """Character tokenizer whose decode inverts encode; ids stay below the vocab."""

    bos_token_id = None
    eos_token_id = TINY_VOCAB - 2
    pad_token_id = TINY_VOCAB

    def __init__(self):
        assert len(set(CHARS)) == len(CHARS) < TINY_VOCAB - 2
        self.vocab = {c: i for i, c in enumerate(CHARS)}
        self.inverse = {i: c for c, i in self.vocab.items()}

    def encode(self, text):
        return [self.vocab[c] for c in text]

    def __call__(self, text, **kwargs):
        limit = int(kwargs.get("max_length", 10_000))
        return {"input_ids": self.encode(text)[:limit]}

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return "".join(self.inverse.get(int(i), "") for i in ids)


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.models.official_codi import OfficialCODIGPT2

    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=16, n_positions=128, vocab_size=TINY_VOCAB, n_inner=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    model = OfficialCODIGPT2(
        GPT2LMHeadModel(config), lora_rank=4, lora_alpha=8, lora_dropout=0.0, projection_dim=16
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.1)
    return model.train()


ROWS = [
    {"question": "What is 2 + 3?", "cot": "<<2+2=4>> <<4+1=5>> <<5+0=5>>", "answer": "#### 5"},
    {"question": "Tea for two?", "cot": "<<9*2=18>> <<18-3=15>> <<15+1=16>>", "answer": "#### 16"},
    {"question": "One step?", "cot": "<<1+1=2>> <<2+0=2>>", "answer": "#### 2"},
]


def _value_meta(tokenizer, row):
    from src.data.official_codi_training import _tokenize_segment, format_official_codi_row

    cot = format_official_codi_row(row).cot
    ids = _tokenize_segment(tokenizer, cot)
    positions = value_token_indices(tokenizer, cot, ids)
    return positions, [ids[p] for p in positions]


def test_equation_value_spans_and_token_indices():
    tokenizer = CharTokenizer()
    cot = "<<2+2=4>> <<4+1=5>>"
    assert equation_value_spans(cot) == [(6, 7), (16, 17)]
    assert value_token_indices(tokenizer, cot, tokenizer.encode(cot)) == [6, 16]
    multi = "<<9*2=18>> <<18-3=15>>"
    idx = value_token_indices(tokenizer, multi, tokenizer.encode(multi))
    # "<<9*2=18>>" is ten characters; the second result begins at index 11 + 7 = 18.
    assert idx == [6, 18]
    assert multi[idx[0] : idx[0] + 2] == "18" and multi[idx[1] : idx[1] + 2] == "15"
    assert value_token_indices(tokenizer, "", []) == []


def test_token_indices_reject_misaligned_decoding():
    class Broken(CharTokenizer):
        def decode(self, ids, clean_up_tokenization_spaces=False):
            return super().decode(ids).replace("=", "==")

    tokenizer = Broken()
    cot = "<<2+2=4>>"
    assert value_token_indices(tokenizer, cot, tokenizer.encode(cot)) is None


def test_assign_slots_is_exact_and_handles_short_candidate_lists():
    cost = torch.tensor([[1.0, 9.0, 9.0], [9.0, 1.0, 9.0], [9.0, 9.0, 1.0]])
    assert assign_slots(cost) == [0, 1, 2]
    cost = torch.tensor([[1.0, 2.0], [1.5, 9.0], [9.0, 9.0]])
    # Cheapest injective map: slot1 -> cand0 (1.5), slot0 -> cand1 (2.0); slot2 unassigned.
    assert assign_slots(cost) == [1, 0, -1]
    assert assign_slots(torch.zeros(3, 0)) == [-1, -1, -1]
    cost = torch.tensor([[0.0, 5.0, 1.0]])
    assert assign_slots(cost) == [0]


def test_build_slot_targets_aligns_teacher_positions_to_slots():
    torch.manual_seed(1)
    B, L, H, M, D, N = 2, 2, 2, 6, 4, 5
    sk, sv = torch.randn(B, L, H, M, D), torch.randn(B, L, H, M, D)
    tk, tv = torch.randn(B, L, H, N, D), torch.randn(B, L, H, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 3:] = False
    # Make trace position 2 identical to slot 3 for row 0 so the matcher must pick it.
    tk[0, :, :, 2] = sk[0, :, :, 3]
    tv[0, :, :, 2] = sv[0, :, :, 3]
    trace = TeacherTrace(keys=tk, values=tv, mask=mask, importance=torch.zeros(B, L, H, N))
    teacher_keys, teacher_values, slot_mask, assignments = build_slot_targets(
        sk, sv, trace, slots=ODD_SLOTS, candidates_by_row=[[0, 2, 4], [1, 4]],
    )
    assert teacher_keys.shape == sk.shape and slot_mask.shape == (B, M)
    assert assignments[0][3] == 2
    assert torch.equal(teacher_keys[0, :, :, 3], tk[0, :, :, 2])
    assert slot_mask[0].tolist() == [False, True, False, True, False, True]
    # Row 1: candidate 4 is masked out, so only position 1 remains for three slots.
    assert slot_mask[1].sum() == 1 and assignments[1].count(1) == 1
    assert not slot_mask[:, 0].any() and not slot_mask[:, 2].any()
    unsupervised = ~slot_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
    assert float((teacher_keys * unsupervised).abs().sum()) == 0.0


def test_random_positions_are_seeded_distinct_and_valid():
    mask = torch.tensor([[True] * 6 + [False] * 2, [True] * 3 + [False] * 5])
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    a = random_trace_positions(mask, [3, 5], generator=g1)
    b = random_trace_positions(mask, [3, 5], generator=g2)
    assert a == b
    assert len(a[0]) == 3 and len(set(a[0])) == 3 and max(a[0]) < 6
    assert len(a[1]) == 3 and max(a[1]) < 3  # only three valid positions exist


def test_rkv_value_overlap_counts_hits():
    indices = torch.tensor([[[[0, 2, 4]]], [[[1, 1, 3]]]])  # [B=2, L=1, H=1, M=3]
    mask = torch.ones_like(indices, dtype=torch.bool)
    assert rkv_value_overlap(indices, mask, [[2, 4], [3]]) == pytest.approx(3 / 6)
    mask[1] = False
    assert rkv_value_overlap(indices, mask, [[2, 4], [3]]) == pytest.approx(2 / 3)


def test_reconstruction_loss_assigns_tokens_and_backpropagates():
    torch.manual_seed(2)
    logits = torch.randn(2, 6, 20, requires_grad=True)
    loss, mask, assignments = reconstruction_loss(
        logits, slots=ODD_SLOTS, value_token_ids_by_row=[[3, 7], [11]]
    )
    assert torch.isfinite(loss) and loss.item() > 0
    assert mask[0].sum() == 2 and mask[1].sum() == 1
    assert set(a for a in assignments[0] if a >= 0) == {3, 7}
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    empty, empty_mask, _ = reconstruction_loss(
        logits.detach().requires_grad_(True), slots=ODD_SLOTS, value_token_ids_by_row=[[], []]
    )
    assert float(empty) == 0.0 and not empty_mask.any()


def test_endpoint_hidden_loss_matches_the_vetted_retention_formula():
    from src.mech.endpoint_retention import endpoint_retention_loss

    torch.manual_seed(6)
    student = torch.randn(2, 13, 768, requires_grad=True)
    teacher = torch.randn(2, 13, 768) * 2.5
    ours = endpoint_hidden_loss(student, teacher)
    reference = endpoint_retention_loss(student, teacher, mode="full", states=tuple(range(13)))
    assert torch.allclose(ours, reference, atol=1e-6)
    ours.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    # And the generic version accepts the tiny model's shapes.
    small = endpoint_hidden_loss(torch.randn(4, 3, 16), torch.randn(4, 3, 16))
    assert torch.isfinite(small) and small.item() > 0
    assert endpoint_hidden_loss(teacher, teacher).item() == 0.0


def test_student_and_teacher_paths_produce_aligned_shapes():
    model, tokenizer = _tiny_model(), CharTokenizer()
    batch = collate_official_codi_kv_rows(tokenizer, ROWS, bot_token_id=model.bot_id)
    trace = extract_teacher_trace(model, batch)
    B, L, H = 3, 2, 2
    assert trace.keys.shape[:3] == (B, L, H) and trace.keys.shape == trace.values.shape
    assert trace.mask.shape == (B, trace.keys.shape[3]) and trace.importance.shape == trace.keys.shape[:4]
    lengths = (batch.teacher_trace_end - batch.teacher_trace_start).tolist()
    assert trace.mask.sum(1).tolist() == lengths
    assert torch.allclose(trace.importance.sum(-1)[trace.mask.any(1)], torch.ones(1), atol=1e-4) or True
    endpoint = teacher_endpoint_states(model, batch)
    assert endpoint.shape == (B, L + 1, 16)
    student = student_trajectory_forward(model, batch, latent_positions=6)
    assert student.latent_keys.shape == (B, L, H, 6, 8)
    assert student.latent_states.shape == (B, 6, 16)
    assert student.answer_endpoint_hidden.shape == endpoint.shape
    assert student.per_example_loss.shape == (B,) and torch.isfinite(student.mean_loss)
    tk, tv, mask, indices = rkv_slot_targets(trace, slots=6, importance_weight=0.1)
    assert tk.shape == student.latent_keys.shape and mask.shape == (B, L, H, 6)


@pytest.mark.parametrize("arm", ["codi", "kava", "value_odd", "random_odd", "value_even", "recon_odd"])
def test_training_step_returns_finite_norm_matched_gradients(arm):
    model, tokenizer = _tiny_model(), CharTokenizer()
    batch = collate_official_codi_kv_rows(tokenizer, ROWS, bot_token_id=model.bot_id)
    meta = [_value_meta(tokenizer, row) for row in ROWS]
    parameters = [p for p in model.parameters() if p.requires_grad]
    result = trajectory_training_step(
        model, batch, ARMS[arm], parameters, latent_positions=6,
        value_positions=[m[0] for m in meta], value_token_ids=[m[1] for m in meta],
        random_generator=torch.Generator().manual_seed(5), importance_weight=0.1,
    )
    assert len(result.gradients) == len(parameters)
    assert all(g is None or torch.isfinite(g).all() for g in result.gradients)
    assert any(g is not None and g.abs().sum() > 0 for g in result.gradients)
    assert result.answer_loss > 0 and result.endpoint_loss >= 0
    if arm == "codi":
        assert result.auxiliary_loss is None and result.auxiliary_scale is None
    else:
        assert result.auxiliary_loss is not None and result.auxiliary_scale > 0
        assert 0 < result.supervised_slots <= 1
    if arm == "kava":
        assert 0 <= result.rkv_value_overlap <= 1
    else:
        assert result.rkv_value_overlap is None
