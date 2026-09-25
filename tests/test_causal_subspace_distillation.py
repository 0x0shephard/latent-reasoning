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
    step = student_training_step(model, [batch], [states], target, parameters, latent_positions=6)
    assert len(step.gradients) == len(parameters)
    assert all(g is None or torch.isfinite(g).all() for g in step.gradients)
    assert step.answer_loss > 0 and step.distillation_loss is not None and step.distillation_scale > 0
    plain = student_training_step(model, [batch], [states], None, parameters, latent_positions=6)
    assert plain.distillation_loss is None and plain.distillation_scale is None


def test_micro_batch_accumulation_matches_the_single_batch_step():
    model, tokenizer = _tiny_model(), CharTokenizer()
    # Equal-length questions keep the student's left padding identical in the full
    # batch and in each half.  The released GPT-2 path is not padding invariant
    # (absolute positions shift with the pad width), so unequal lengths would
    # differ at the 1e-3 level for reasons unrelated to accumulation.
    rows = [
        {"question": "What is 2 + 3?", "cot": "<<2+2=4>> <<4+1=5>> <<5+0=5>>", "answer": "#### 5"},
        {"question": "What is 4 + 1?", "cot": "<<4+0=4>> <<4+1=5>> <<5+0=5>>", "answer": "#### 5"},
        {"question": "What is 3 + 3?", "cot": "<<3+3=6>> <<6+0=6>>", "answer": "#### 6"},
        {"question": "What is 1 + 5?", "cot": "<<1+5=6>> <<6+0=6>> <<6+0=6>>", "answer": "#### 6"},
    ]
    full = collate_official_codi_kv_rows(tokenizer, rows, bot_token_id=model.bot_id)
    halves = [collate_official_codi_kv_rows(tokenizer, rows[i : i + 2], bot_token_id=model.bot_id)
              for i in (0, 2)]
    states, _ = teacher_decision_states(model, full)
    pca = fit_teacher_pca(torch.randn(64, 16))
    target = build_target(pca, torch.randn(64, 16), [0, 1, 2])
    parameters = [p for p in model.parameters() if p.requires_grad]
    model.eval()  # dropout off so the two paths are deterministic
    one = student_training_step(model, [full], [states], target, parameters, latent_positions=6)
    two = student_training_step(model, halves, [states[:2], states[2:]], target, parameters, latent_positions=6)
    assert one.answer_loss == pytest.approx(two.answer_loss, rel=1e-4)
    assert one.distillation_loss == pytest.approx(two.distillation_loss, rel=1e-4)
    assert one.distillation_scale == pytest.approx(two.distillation_scale, rel=1e-3)
    for a, b in zip(one.gradients, two.gradients):
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-3)
    assert READOUT_STATE == -1


def test_reset_projector_touches_only_the_projector_and_is_seeded():
    from src.mech.causal_subspace_distillation import reset_projector

    model = _tiny_model()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.2)
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    prj_before = {n: p.detach().clone() for n, p in model.prj.named_parameters()}
    report = reset_projector(model, seed=3)
    assert report["projector"] > 0 and report["lora_A"] == 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert torch.equal(p, lora_before[n])
    assert any(not torch.equal(p, prj_before[n]) for n, p in model.prj.named_parameters())
    again = _tiny_model(); reset_projector(again, seed=3)
    for (n, p), (_, q) in zip(model.prj.named_parameters(), again.prj.named_parameters()):
        assert torch.equal(p, q), n


def test_scale_lora_b_halves_every_b_matrix():
    from src.mech.causal_subspace_distillation import scale_lora_b

    model = _tiny_model()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.2)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    report = scale_lora_b(model, factor=0.5)
    assert report["lora_B"] == len(before)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            assert torch.allclose(p, before[n] * 0.5)


def test_gradient_cosine_and_auxiliary_multiplier():
    from src.mech.causal_subspace_distillation import gradient_cosine

    a = (torch.tensor([1.0, 0.0]), None, torch.tensor([[2.0]]))
    assert gradient_cosine(a, a) == pytest.approx(1.0)
    assert gradient_cosine(a, (torch.tensor([-1.0, 0.0]), None, torch.tensor([[-2.0]]))) == pytest.approx(-1.0)
    assert gradient_cosine(a, (torch.zeros(2), None, torch.zeros(1, 1))) is None

    model, tokenizer = _tiny_model(), CharTokenizer()
    batch = collate_official_codi_kv_rows(tokenizer, ROWS, bot_token_id=model.bot_id)
    states, _ = teacher_decision_states(model, batch)
    pca = fit_teacher_pca(torch.randn(64, 16))
    target = build_target(pca, torch.randn(64, 16), [0, 1, 2])
    parameters = [p for p in model.parameters() if p.requires_grad]
    one = student_training_step(model, [batch], [states], target, parameters, latent_positions=6)
    assert one.gradient_cosine is not None and -1.0 <= one.gradient_cosine <= 1.0
    plain = student_training_step(model, [batch], [states], None, parameters, latent_positions=6)
    assert plain.gradient_cosine is None
    zero = student_training_step(model, [batch], [states], target, parameters, latent_positions=6,
                                 auxiliary_multiplier=0.0)
    for g0, gp in zip(zero.gradients, plain.gradients):
        if g0 is not None:
            assert torch.allclose(g0, gp, atol=1e-6)
    assert zero.distillation_scale == pytest.approx(0.0)
    three = student_training_step(model, [batch], [states], target, parameters, latent_positions=6,
                                  auxiliary_multiplier=3.0)
    assert three.distillation_scale == pytest.approx(3.0 * one.distillation_scale)


def test_add_projector_noise_is_seeded_scaled_and_local():
    from src.mech.causal_subspace_distillation import add_projector_noise

    model = _tiny_model()
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    weights_before = {n: p.detach().clone() for n, p in model.prj.named_parameters()}
    report = add_projector_noise(model, sigma=0.5, seed=11)
    assert report["projector_linears"] == 2 and report["sigma"] == 0.5
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert torch.equal(p, lora_before[n])
    for n, p in model.prj.named_parameters():
        if n.endswith(".weight") and p.ndim == 2:
            delta = (p - weights_before[n]).std() / weights_before[n].std()
            assert 0.35 < float(delta) < 0.65, (n, float(delta))
        else:  # biases and LayerNorm untouched
            assert torch.equal(p, weights_before[n]), n
    again = _tiny_model(); add_projector_noise(again, sigma=0.5, seed=11)
    for (n, p), (_, q) in zip(model.prj.named_parameters(), again.prj.named_parameters()):
        assert torch.equal(p, q), n


def test_transfer_patch_selects_the_decisive_directions_and_breakage_anchor_excludes_them():
    from src.mech.causal_subspace_distillation import (
        breakage_score, patched_states, select_breakage_anchor, select_transfer_patch, transfer_patch_score,
    )

    teacher, gold, readout = _synthetic_teacher()
    pca = fit_teacher_pca(teacher)
    g = torch.Generator().manual_seed(1)
    # a "student" whose decisive dims 4..7 are corrupted and whose inert dims are noisy
    student = teacher.clone()
    student[:, 4:8] = torch.randn(teacher.shape[0], 4, generator=g) * 2.0
    student[:, 0:4] += torch.randn(teacher.shape[0], 4, generator=g) * 5.0
    # patching nothing returns the student; patching everything returns the teacher
    assert torch.allclose(patched_states(student, teacher, pca, []), student)
    assert torch.allclose(patched_states(student, teacher, pca, list(range(16))), teacher, atol=1e-4)
    before = transfer_patch_score(student, teacher, gold, pca, [], readout)[0]
    chosen = select_transfer_patch(student, teacher, gold, pca, readout, 4, candidates=16)
    after = transfer_patch_score(student, teacher, gold, pca, chosen, readout)[0]
    assert after > before + 0.3
    # the decisive PCs are 4..7 (variances 3, 2.5, 2, 1.5 sit below the four inert dims)
    assert set(chosen) <= set(range(4, 8)), chosen
    # breakage: drift in the decisive dims breaks answers; inert dims do not
    assert breakage_score(teacher, student, gold, pca, [4, 5, 6, 7], readout) > 0.3
    assert breakage_score(teacher, student, gold, pca, [0, 1, 2, 3], readout) < 0.05
    anchor = select_breakage_anchor(teacher, student, gold, pca, readout, 2, candidates=16, exclude=chosen)
    assert not set(anchor) & set(chosen)
    assert set(anchor) <= set(range(4, 8)) - set(chosen) or len(set(range(4, 8)) - set(chosen)) < 2


def test_multi_term_step_matches_single_term_and_adds_the_anchor():
    from src.mech.causal_subspace_distillation import student_training_step_multi

    model, tokenizer = _tiny_model(), CharTokenizer()
    batch = collate_official_codi_kv_rows(tokenizer, ROWS, bot_token_id=model.bot_id)
    states, _ = teacher_decision_states(model, batch)
    pca = fit_teacher_pca(torch.randn(64, 16))
    teach = build_target(pca, torch.randn(64, 16), [0, 1, 2])
    anchor = build_target(pca, torch.randn(64, 16), [3, 4, 5, 6])
    parameters = [p for p in model.parameters() if p.requires_grad]
    single = student_training_step(model, [batch], [states], teach, parameters, latent_positions=6)
    multi_one = student_training_step_multi(model, [batch], [states], [(teach, 1.0)], parameters, latent_positions=6)
    for a, b in zip(single.gradients, multi_one.gradients):
        if a is not None:
            assert torch.allclose(a, b, atol=1e-6)
    assert multi_one.distillation_scale == pytest.approx(single.distillation_scale)
    multi_two = student_training_step_multi(model, [batch], [states], [(teach, 1.0), (anchor, 0.3)], parameters, latent_positions=6)
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(multi_one.gradients, multi_two.gradients) if a is not None)
    assert multi_two.distillation_loss == pytest.approx(multi_one.distillation_loss)
    none = student_training_step_multi(model, [batch], [states], [], parameters, latent_positions=6)
    assert none.distillation_loss is None
