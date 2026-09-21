import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.data.official_codi_training import collate_official_codi_kv_rows  # noqa: E402
from src.mech.causal_subspace_distillation import (  # noqa: E402
    READOUT_STATE,
    build_target,
    choose_rank,
    cosine_lr,
    distillation_loss,
    evaluate_index_sets,
    fit_teacher_pca,
    gold_outcomes,
    jaccard,
    readout_matrix,
    reinitialize_student,
    retain_only_outcomes,
    select_causal_greedy,
    select_random,
    select_relevance,
    select_variance,
    student_training_step,
    teacher_decision_states,
)
from tests.test_trajectory_supervision import ROWS, CharTokenizer, _tiny_model  # noqa: E402


def _synthetic_teacher(n=3000, d=16, vocab=40, seed=0):
    """States whose gold token is decided by low-variance directions 4..7."""
    g = torch.Generator().manual_seed(seed)
    # Dominant inert dims 0-3, decisive dims 4-7 with distinct variances so their
    # principal components are identifiable, then an isotropic noise block.
    scales = torch.tensor([30.0, 20.0, 12.0, 8.0, 3.0, 2.5, 2.0, 1.5] + [1.0] * (d - 8))
    latent = torch.randn(n, d, generator=g) * scales
    readout = torch.zeros(vocab, d)
    readout[:, 4:8] = torch.randn(vocab, 4, generator=g) * 3
    logits = latent @ readout.T
    gold = logits.argmax(-1)
    return latent, gold, readout


def test_pca_accepts_fewer_samples_than_dimensions():
    states = torch.randn(40, 96)
    pca = fit_teacher_pca(states)
    assert pca.basis.shape == (96, 96)
    assert torch.allclose(pca.basis.T @ pca.basis, torch.eye(96, dtype=torch.float64), atol=1e-8)
    assert float(pca.eigenvalues[40:].abs().max()) < 1e-6
    with pytest.raises(ValueError):
        fit_teacher_pca(torch.randn(1, 96))


def test_pca_is_descending_and_orthonormal():
    states, _, _ = _synthetic_teacher()
    pca = fit_teacher_pca(states)
    assert torch.allclose(pca.basis.T @ pca.basis, torch.eye(16, dtype=torch.float64), atol=1e-8)
    assert bool((pca.eigenvalues[1:] <= pca.eigenvalues[:-1] + 1e-9).all())
    assert pca.eigenvalues[:4].sum() / pca.eigenvalues.sum() > 0.95


def test_selectors_separate_variance_from_causal_on_synthetic_teacher():
    states, gold, readout = _synthetic_teacher()
    pca = fit_teacher_pca(states)
    variance = select_variance(4)
    causal = select_causal_greedy(states, gold, pca, readout, 4, candidates=16)
    relevance = select_relevance(states, gold, pca, readout, 4)
    random_set = select_random(4, 16, seed=3)
    assert variance == [0, 1, 2, 3]
    assert causal == [4, 5, 6, 7]
    assert set(relevance) == {4, 5, 6, 7}
    assert len(set(random_set)) == 4 and select_random(4, 16, seed=3) == random_set
    report = evaluate_index_sets(states, gold, pca, readout,
                                 {"variance": variance, "causal": causal, "random": random_set})
    # Finite-sample eigenvectors of the decisive block are only approximately axis
    # aligned, so retention is high but not exact; the variance set is near chance.
    assert report["causal"]["first_token_accuracy"] > 0.9
    assert report["variance"]["first_token_accuracy"] < 0.5
    assert report["causal"]["mean_margin"] > report["variance"]["mean_margin"]
    assert report["dense"]["first_token_accuracy"] == 1.0
    assert report["jaccard"]["variance|causal"] == 0.0
    assert 0.9 < report["variance"]["variance_share"] <= 1.0
    log_prob, correct = retain_only_outcomes(states, gold, pca, [], readout)
    assert log_prob.shape == (3000,) and correct.dtype == torch.bool


def test_choose_rank_maximises_gap_times_retention_under_both_floors():
    def report(causal_ret, causal_acc, variance_acc):
        return {"causal": {"retention_of_dense": causal_ret, "first_token_accuracy": causal_acc},
                "variance": {"first_token_accuracy": variance_acc}}
    # The observed teacher table from the first Kaggle run (dense 0.894).
    observed = {8: report(0.370, 0.331, 0.105), 12: report(0.543, 0.485, 0.323), 16: report(0.661, 0.591, 0.513)}
    rank, audit = choose_rank(observed, minimum_gap=0.05, retention_floor=0.50)
    assert rank == 12
    assert audit["8"]["eligible"] is False and audit["12"]["eligible"] and audit["16"]["eligible"]
    assert audit["12"]["score"] > audit["16"]["score"]
    # Gap floor and retention floor each exclude on their own.
    assert choose_rank({8: report(0.9, 0.9, 0.88)}, minimum_gap=0.05, retention_floor=0.5)[0] is None
    assert choose_rank({8: report(0.3, 0.3, 0.1)}, minimum_gap=0.05, retention_floor=0.5)[0] is None
    # Ties on the product prefer the smaller rank.
    tie = {8: report(0.5, 0.6, 0.4), 16: report(0.5, 0.6, 0.4)}
    assert choose_rank(tie, minimum_gap=0.05, retention_floor=0.5)[0] == 8


def test_target_and_loss_are_scale_free_and_full_is_all_dims():
    states, _, _ = _synthetic_teacher()
    pca = fit_teacher_pca(states)
    full = build_target(pca, states, None)
    sub = build_target(pca, states, [4, 5, 6, 7])
    assert full.basis is None and sub.basis.shape == (16, 4)
    student = states[:8].float() + 0.1 * torch.randn(8, 16)
    assert distillation_loss(states[:8].float(), states[:8].float(), sub).item() == 0.0
    loss = distillation_loss(student.requires_grad_(True), states[:8].float(), sub)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert student.grad is not None
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert cosine_lr(0, total_steps=100, base=1.0, warmup=10) == pytest.approx(0.1)
    assert cosine_lr(99, total_steps=100, base=1.0, warmup=10) < 0.01
    assert cosine_lr(10, total_steps=100, base=1.0, warmup=10) == pytest.approx(1.0)


def test_reinitialised_student_equals_base_model_and_keeps_embeddings():
    model = _tiny_model()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.2)
    ids = torch.randint(0, 50, (2, 6))
    embeddings_before = model.input_embeddings().weight.detach().clone()
    with torch.no_grad():
        adapted = model.codi(input_ids=ids, return_dict=True).logits
    reset = reinitialize_student(model, seed=7)
    assert reset["lora_A"] > 0 and reset["lora_B"] > 0 and reset["projector"] > 0
    with torch.no_grad():
        fresh = model.codi(input_ids=ids, return_dict=True).logits
        model.codi.disable_adapter_layers()
        base = model.codi(input_ids=ids, return_dict=True).logits
        model.codi.enable_adapter_layers()
    assert not torch.allclose(adapted, fresh, atol=1e-5)
    assert torch.allclose(fresh, base, atol=1e-5)
    assert torch.equal(embeddings_before, model.input_embeddings().weight.detach())
    again = _tiny_model()
    reinitialize_student(again, seed=7)
    a = dict(model.named_parameters())
    for name, parameter in again.named_parameters():
        if "lora_A" in name:
            assert torch.equal(parameter, a[name])


def test_teacher_states_and_student_step_on_tiny_model():
    model, tokenizer = _tiny_model(), CharTokenizer()
    batch = collate_official_codi_kv_rows(tokenizer, ROWS, bot_token_id=model.bot_id)
    states, gold = teacher_decision_states(model, batch)
    assert states.shape == (3, 16) and gold.shape == (3,)
    assert torch.equal(gold, batch.teacher_ids[torch.arange(3), batch.teacher_answer_start])
    readout = readout_matrix(model, vocab_limit=50)
    assert readout.shape == (50, 16)
    log_prob, correct = gold_outcomes(states, gold, readout)
    assert log_prob.shape == (3,) and torch.isfinite(log_prob).all()
    pca = fit_teacher_pca(torch.randn(64, 16))
    target = build_target(pca, torch.randn(64, 16), [0, 1, 2])
    parameters = [p for p in model.parameters() if p.requires_grad]
    step = student_training_step(model, batch, states, target, parameters, latent_positions=6)
    assert len(step.gradients) == len(parameters)
    assert all(g is None or torch.isfinite(g).all() for g in step.gradients)
    assert step.answer_loss > 0 and step.distillation_loss is not None and step.distillation_scale > 0
    plain = student_training_step(model, batch, states, None, parameters, latent_positions=6)
    assert plain.distillation_loss is None and plain.distillation_scale is None
    assert READOUT_STATE == -1
