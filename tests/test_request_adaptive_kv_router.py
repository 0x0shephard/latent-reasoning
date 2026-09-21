import torch

from src.mech.request_adaptive_kv_router import (
    FEATURE_NAMES,
    deterministic_question_split,
    fit_ridge_loss_router,
    per_example_dense_kl,
    prompt_feature_matrix,
    prompt_feature_vector,
    route_distribution,
)


def test_prompt_features_are_fixed_finite_and_label_free():
    question = "Mia has 12.5 dollars and spends 25 percent. How much is left?"
    features = prompt_feature_vector(question)
    assert features.shape == (len(FEATURE_NAMES),)
    assert bool(torch.isfinite(features).all())
    assert features[2] > 0
    assert features[-1] >= 2


def test_hash_split_is_order_independent_and_exhaustive():
    rows = [{"question": f"question {index}", "gold": index} for index in range(10)]
    first = deterministic_question_split(rows, seed=7, sizes=(4, 3, 3))
    second = deterministic_question_split(list(reversed(rows)), seed=7, sizes=(4, 3, 3))
    assert [[row["question"] for row in split] for split in first] == [
        [row["question"] for row in split] for split in second
    ]
    assert len({row["question"] for split in first for row in split}) == 10


def test_ridge_router_learns_different_profiles_from_preanswer_features():
    questions = [
        "There are 2 apples. How many?",
        "There are 3 pears. How many?",
        "A price changes by 10 percent and then 20 percent. What remains?",
        "A price changes by 15 percent and then 30 percent. What remains?",
    ]
    features = prompt_feature_matrix(questions)
    losses = torch.tensor([
        [0.01, 0.20, 0.30],
        [0.02, 0.20, 0.30],
        [0.30, 0.20, 0.01],
        [0.30, 0.20, 0.02],
    ], dtype=torch.float64)
    router = fit_ridge_loss_router(
        features, losses, profile_names=("reconstruction", "hybrid", "answer_fisher"),
        ridge=1e-4,
    )
    routes = router.route(features)
    assert routes[:2].tolist() == [0, 0]
    assert routes[2:].tolist() == [2, 2]
    assert route_distribution(routes, router.profile_names)["active_profiles"] == 2


def test_per_example_dense_kl_is_zero_only_for_equal_distributions():
    dense = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    same = per_example_dense_kl(dense, dense.clone())
    changed = per_example_dense_kl(dense, dense.flip(-1))
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-12)
    assert bool((changed > 0).all())
