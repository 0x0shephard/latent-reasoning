import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.data.official_codi_training import collate_official_codi_kv_rows  # noqa: E402
from src.mech.causal_subspace_distillation import fit_teacher_pca  # noqa: E402
from src.mech.workspace_interchange import (  # noqa: E402
    change_rate,
    derangement,
    first_token_outcomes,
    interchange_hook,
    slot_gradient_scores,
    spearman,
)
from tests.test_trajectory_supervision import ROWS, CharTokenizer, _tiny_model  # noqa: E402


def test_derangement_has_no_fixed_points_and_is_seeded():
    p = derangement(50, seed=3)
    assert sorted(p) == list(range(50)) and all(i != j for i, j in enumerate(p))
    assert derangement(50, seed=3) == p and derangement(50, seed=4) != p


def test_spearman_and_change_rate():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert abs(spearman([1, 2, 3, 4, 5, 6], [3, 1, 4, 1.5, 5, 2])) < 1.0
    assert change_rate(torch.tensor([1, 2, 3]), torch.tensor([1, 0, 3])) == pytest.approx(1 / 3)


def test_interchange_hook_moves_rows_inside_the_basis_only():
    g = torch.Generator().manual_seed(0)
    state = torch.randn(4, 8, generator=g); donor = torch.randn(4, 8, generator=g)
    basis = torch.linalg.qr(torch.randn(8, 3, generator=g))[0]
    hook = interchange_hook(slot=2, donor_states=donor, basis=basis)
    assert torch.equal(hook(state, 1, 0), state)                       # other slots untouched
    moved = hook(state, 2, 0)
    assert torch.allclose(moved @ basis, donor @ basis, atol=1e-5)      # donor inside the basis
    comp = torch.eye(8) - basis @ basis.T
    assert torch.allclose(moved @ comp, state @ comp, atol=1e-5)        # unchanged outside it
    assert torch.equal(interchange_hook(2, donor, None)(state, 2, 0), donor)
    # chunk offset selects the donor rows
    assert torch.equal(interchange_hook(2, donor, None)(state[:2], 2, 2), donor[2:4])


def test_first_token_outcomes_and_gradient_scores_on_tiny_model():
    model, tokenizer = _tiny_model(), CharTokenizer()
    rows = ROWS
    pred, gold = first_token_outcomes(model, tokenizer, rows, latent_positions=6, batch_size=2, device=torch.device("cpu"))
    assert pred.shape == gold.shape == (len(rows),)
    batch = collate_official_codi_kv_rows(tokenizer, rows, bot_token_id=model.bot_id)
    assert torch.equal(gold, batch.teacher_ids[torch.arange(len(rows)), batch.teacher_answer_start])
    # a full swap toward a donor changes the computation; identity donor does not
    donors = torch.randn(len(rows), 16)
    swapped, _ = first_token_outcomes(model, tokenizer, rows, latent_positions=6, batch_size=2, device=torch.device("cpu"),
                                      slot=1, donor_states=donors, basis=None)
    assert swapped.shape == pred.shape
    pca = fit_teacher_pca(torch.randn(64, 16))
    scores = slot_gradient_scores(model, tokenizer, rows, latent_positions=6, slot=3, pca=pca, batch_size=2,
                                  device=torch.device("cpu"), candidates=5)
    assert scores.shape == (5,) and bool(torch.isfinite(scores).all()) and bool((scores >= 0).all())
    # the terminal slot's output state is never consumed, so its gradient is exactly zero
    terminal = slot_gradient_scores(model, tokenizer, rows, latent_positions=6, slot=5, pca=pca, batch_size=2,
                                    device=torch.device("cpu"), candidates=5)
    assert bool((terminal == 0).all())
    swapped5, _ = first_token_outcomes(model, tokenizer, rows, latent_positions=6, batch_size=2, device=torch.device("cpu"),
                                       slot=5, donor_states=donors, basis=None)
    assert torch.equal(swapped5, pred)
