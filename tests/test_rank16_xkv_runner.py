from scripts.run_codi_rank16_xkv_mechanism_confirmation import (
    CONFIRMATION_EXAMPLES,
    CONFIRMATION_START,
    FINAL_START,
    PREREGISTERED_GROUPS,
    RANK,
    _arm,
)


def test_rank16_confirmation_boundaries_are_frozen_and_disjoint():
    assert RANK == 16
    assert CONFIRMATION_START == 256
    assert CONFIRMATION_EXAMPLES == 512
    assert FINAL_START == 768
    assert CONFIRMATION_START + CONFIRMATION_EXAMPLES == FINAL_START
    assert PREREGISTERED_GROUPS == (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
    )


def test_factorial_arms_toggle_only_the_preregistered_components():
    weights = {group: object() for group in PREREGISTERED_GROUPS}
    utilities = {group: 1.0 for group in PREREGISTERED_GROUPS}
    protected = {(11, "value"): object()}
    allocation = _arm(
        "allocation_only",
        latent_positions=6,
        feature_weights=weights,
        utilities=utilities,
        protected_bases=protected,
    )
    fisher = _arm(
        "fisher_only",
        latent_positions=6,
        feature_weights=weights,
        utilities=utilities,
        protected_bases=protected,
    )
    full = _arm(
        "full_method",
        latent_positions=6,
        feature_weights=weights,
        utilities=utilities,
        protected_bases=protected,
    )
    assert allocation.adaptive_ranks and allocation.feature_weights is None
    assert fisher.feature_weights is weights and not fisher.adaptive_ranks
    assert full.adaptive_ranks
    assert full.feature_weights is weights
    assert full.protected_bases is protected
