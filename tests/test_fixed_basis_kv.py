import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.mech.fixed_basis_kv import (  # noqa: E402
    HeadProjectionHook,
    KVSecondMomentCollector,
    allocate_energy_ranks,
    attention_qkv_modules,
    collect_kv_second_moments,
    energy_fraction_curves,
    fit_head_bases,
    head_geometry,
    projectors_from_ranks,
    random_head_bases,
    rank_for_energy,
    storage_report,
    subspace_capture,
    uniform_ranks,
)
from src.mech.preanswer_kv_subspace import cache_as_legacy_tuple  # noqa: E402


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.models.official_codi import OfficialCODIGPT2

    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=16, n_positions=64, vocab_size=50, n_inner=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    backbone = GPT2LMHeadModel(config)
    model = OfficialCODIGPT2(
        backbone, lora_rank=4, lora_alpha=8, lora_dropout=0.0, projection_dim=16,
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.1)
    return model.eval()


def _synthetic_eigenvalues():
    torch.manual_seed(1)
    values = torch.rand(3, 2, 2, 8, dtype=torch.float64)
    return values.sort(-1, descending=True).values


def test_fit_head_bases_are_orthonormal_and_descending():
    torch.manual_seed(2)
    samples = torch.randn(200, 3, 2, 2, 8, dtype=torch.float64)
    moments = torch.einsum("nlkhd,nlkhe->lkhde", samples, samples)
    bases = fit_head_bases(moments)
    gram = bases.vectors.transpose(-1, -2) @ bases.vectors
    assert torch.allclose(gram, torch.eye(8, dtype=torch.float64).expand_as(gram), atol=1e-9)
    assert bool((bases.eigenvalues[..., 1:] <= bases.eigenvalues[..., :-1] + 1e-9).all())
    reconstructed = (bases.vectors * bases.eigenvalues.unsqueeze(-2)) @ bases.vectors.transpose(-1, -2)
    assert torch.allclose(reconstructed, moments, atol=1e-6)


def test_projectors_are_idempotent_and_identity_at_full_rank():
    torch.manual_seed(5)
    samples = torch.randn(50, 1, 2, 2, 8, dtype=torch.float64)
    moments = torch.einsum("nlkhd,nlkhe->lkhde", samples, samples)
    bases = fit_head_bases(moments + 10 * torch.eye(8, dtype=torch.float64))
    ranks = uniform_ranks(layers=1, heads=2, rank=3, key_rank=3, value_rank=8)
    projectors = projectors_from_ranks(bases, ranks)
    assert torch.allclose(projectors @ projectors, projectors, atol=1e-9)
    assert torch.allclose(projectors[0, 1], torch.eye(8, dtype=torch.float64).expand(2, 8, 8))
    trace = torch.diagonal(projectors[0, 0], dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.full((2,), 3.0, dtype=torch.float64))
    with pytest.raises(ValueError):
        projectors_from_ranks(bases, uniform_ranks(layers=1, heads=2, rank=9))


def test_energy_allocation_spends_budget_and_beats_uniform_energy():
    eigenvalues = _synthetic_eigenvalues()
    total = eigenvalues.shape[0] * eigenvalues.shape[1] * eigenvalues.shape[2] * 3
    ranks = allocate_energy_ranks(eigenvalues, total_rank=total)
    assert int(ranks.sum()) == total
    assert int(ranks.min()) >= 1 and int(ranks.max()) <= 8
    curves = eigenvalues.cumsum(-1)
    allocated_energy = curves.gather(-1, (ranks - 1).unsqueeze(-1)).sum()
    uniform_energy = curves[..., 2].sum()
    assert float(allocated_energy) >= float(uniform_energy) - 1e-12
    with pytest.raises(ValueError):
        allocate_energy_ranks(eigenvalues, total_rank=5)
    with pytest.raises(ValueError):
        allocate_energy_ranks(eigenvalues.flip(-1), total_rank=total)


def test_energy_curves_and_rank_for_energy():
    eigenvalues = torch.tensor([[[[4.0, 3.0, 2.0, 1.0]]]], dtype=torch.float64)
    curves = energy_fraction_curves(eigenvalues)
    assert torch.allclose(curves[0, 0, 0], torch.tensor([0.4, 0.7, 0.9, 1.0], dtype=torch.float64))
    assert int(rank_for_energy(eigenvalues, 0.9)) == 3
    assert int(rank_for_energy(eigenvalues, 0.95)) == 4
    assert int(rank_for_energy(eigenvalues, 0.1)) == 1


def test_random_bases_and_subspace_capture_bounds():
    bases = random_head_bases(layers=2, kinds=2, heads=3, head_dim=8, seed=7)
    gram = bases.vectors.transpose(-1, -2) @ bases.vectors
    assert torch.allclose(gram, torch.eye(8, dtype=torch.float64).expand_as(gram), atol=1e-9)
    again = random_head_bases(layers=2, kinds=2, heads=3, head_dim=8, seed=7)
    assert torch.equal(bases.vectors, again.vectors)
    same = subspace_capture(bases.vectors, bases.vectors, 3)
    assert torch.allclose(same, torch.ones_like(same))
    complement = bases.vectors.flip(-1)
    assert torch.allclose(subspace_capture(bases.vectors, complement, 3), torch.zeros(2, 2, 3, dtype=torch.float64), atol=1e-12)


def test_storage_report_counts_only_compressed_components():
    ranks = uniform_ranks(layers=2, heads=2, rank=4, key_rank=4, value_rank=8)
    report = storage_report(ranks, head_dim=8)
    assert report["dense_units_per_token"] == 2 * 2 * 2 * 8
    assert report["stored_units_per_token"] == 2 * 2 * (4 + 8)
    assert report["basis_parameters"] == 2 * 2 * 8 * 4
    assert report["key_mean_rank"] == 4.0 and report["value_mean_rank"] == 8.0


def test_projection_hook_matches_head_split_and_restores_dense_path():
    model = _tiny_model()
    layers, heads, head_dim = head_geometry(model)
    assert (layers, heads, head_dim) == (2, 2, 8)
    assert len(attention_qkv_modules(model)) == 2
    torch.manual_seed(3)
    input_ids = torch.randint(0, 50, (2, 5))
    with torch.no_grad():
        dense = model.codi(input_ids=input_ids, use_cache=True, return_dict=True)
    dense_keys = cache_as_legacy_tuple(dense.past_key_values)[0][0].clone()
    bases = random_head_bases(layers=2, kinds=2, heads=2, head_dim=8, seed=11)

    full = HeadProjectionHook(
        projectors_from_ranks(bases, uniform_ranks(layers=2, heads=2, rank=8)),
        ranks=uniform_ranks(layers=2, heads=2, rank=8),
    )
    with full.attach(model), torch.no_grad():
        same = model.codi(input_ids=input_ids, use_cache=True, return_dict=True)
    assert torch.allclose(same.logits, dense.logits, atol=1e-5)

    ranks = uniform_ranks(layers=2, heads=2, rank=2)
    low = HeadProjectionHook(projectors_from_ranks(bases, ranks), ranks=ranks)
    with low.attach(model), torch.no_grad():
        projected = model.codi(input_ids=input_ids, use_cache=True, return_dict=True)
    assert not torch.allclose(projected.logits, dense.logits, atol=1e-4)
    projected_keys = cache_as_legacy_tuple(projected.past_key_values)[0][0]
    expected = torch.einsum(
        "bhtd,hde->bhte", dense_keys.double(), low.projectors[0, 0]
    ).to(projected_keys.dtype)
    assert torch.allclose(projected_keys, expected, atol=1e-5)
    assert low.summary()["compression_ratio"] == pytest.approx(4.0)

    with torch.no_grad():
        restored = model.codi(input_ids=input_ids, use_cache=True, return_dict=True)
    assert torch.allclose(restored.logits, dense.logits, atol=1e-6)


def test_collector_moments_match_the_cached_keys_and_values():
    model = _tiny_model()
    collector = KVSecondMomentCollector(layers=2, heads=2, head_dim=8)
    torch.manual_seed(4)
    input_ids = torch.randint(0, 50, (2, 6))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    collector.attach(model)
    try:
        collector.set_context("question", mask)
        with torch.no_grad():
            output = model.codi(
                input_ids=input_ids, attention_mask=mask.long(), use_cache=True,
                return_dict=True,
            )
        collector.clear_context()
    finally:
        collector.detach()
    legacy = cache_as_legacy_tuple(output.past_key_values)
    for layer in range(2):
        for kind_index in range(2):
            tensor = legacy[layer][kind_index].transpose(1, 2)  # [B, T, H, D]
            selected = tensor[mask].double()
            expected = torch.einsum("nhd,nhe->hde", selected, selected)
            observed = collector.row_type_moments("question")[layer, kind_index]
            assert torch.allclose(observed, expected, atol=1e-6)
    assert collector.count_summary() == {"question": 9, "latent": 0, "cue": 0, "answer": 0}
    assert torch.equal(collector.pooled_moments(), collector.moments.sum(0))
    with pytest.raises(RuntimeError):
        collector.attach(model)
        try:
            with torch.no_grad():
                model.codi(input_ids=input_ids, return_dict=True)
        finally:
            collector.detach()


def _word_level_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    # The model resizes to vocab_size + 3, so greedy argmax can emit ids up to
    # eot_id - 1.  Give the tokenizer every one of those ids so decoding is total.
    vocab = {f"w{index}": index for index in range(52)}
    vocab["[PAD]"] = 52
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="w0"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="w1", unk_token="w0",
        padding_side="left",
    )


def test_collect_kv_second_moments_covers_every_row_type():
    model = _tiny_model()
    tokenizer = _word_level_tokenizer()
    collector = KVSecondMomentCollector(layers=2, heads=2, head_dim=8)
    questions = ["w3 w4 w5 w6", "w7 w8", "w9 w10 w11"]
    collector.attach(model)
    try:
        result = collect_kv_second_moments(
            model, tokenizer, questions, collector=collector, latent_iterations=2,
            max_new_tokens=3, batch_size=2, device=torch.device("cpu"),
            answer_cue="w12 w13",
        )
    finally:
        collector.detach()
    counts = result["row_counts"]
    assert len(result["outputs"]) == 3
    # Derive the expectation from the tokenizer rather than from whitespace: what
    # matters is that left padding is excluded, so the moments never see pad rows.
    expected_question = 0
    padded_positions = 0
    for start in range(0, len(questions), 2):
        chunk = questions[start : start + 2]
        encoded = tokenizer(
            chunk, return_tensors="pt", padding="longest", add_special_tokens=False
        )
        expected_question += int(encoded["attention_mask"].sum()) + len(chunk)
        padded_positions += int((encoded["attention_mask"] == 0).sum())
    assert padded_positions > 0, "the fixture must exercise left padding"
    assert counts["question"] == expected_question
    assert counts["latent"] == 2 * 3
    # The cue pass caches one <eot> marker plus however many pieces the cue makes.
    cue_ids = tokenizer(" w12 w13", add_special_tokens=False)["input_ids"]
    assert counts["cue"] == 3 * (1 + len(cue_ids))
    # The first answer token is produced by the cue pass, and the tiny random model
    # may emit EOS at any later step, so only the bound is determined.
    assert 0 <= counts["answer"] <= 3 * 2
    if counts["answer"]:
        assert float(collector.row_type_moments("answer").abs().sum()) > 0
    assert bool(torch.isfinite(collector.pooled_moments()).all())
