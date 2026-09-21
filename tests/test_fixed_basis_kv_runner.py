import pytest

pytest.importorskip("torch")

from scripts.run_codi_fixed_basis_kv import (  # noqa: E402
    ACCURACY_NONINFERIORITY_MARGIN,
    BASIS_FIT_EXAMPLES,
    CONTRACT,
    MINIMUM_FIRST_TOKEN_FIDELITY,
    MINIMUM_RETENTION,
    RANDOM_BASIS_SEEDS,
    RANK_GRID,
    SAMPLING_SEED,
    SELECTION_EXAMPLES,
    arm_name,
    final_arm_specs,
    select_operating_point,
)


def test_fixed_basis_protocol_is_frozen():
    assert CONTRACT == "official_codi_fixed_basis_kv_v1"
    assert BASIS_FIT_EXAMPLES == 1_024
    assert SELECTION_EXAMPLES == 256
    assert SAMPLING_SEED == 20_260_921
    assert RANK_GRID == (8, 16, 24, 32, 40, 48)
    assert RANDOM_BASIS_SEEDS == (20_260_921, 20_260_922)
    assert MINIMUM_RETENTION == 0.98
    assert MINIMUM_FIRST_TOKEN_FIDELITY == 0.95
    assert ACCURACY_NONINFERIORITY_MARGIN == 0.02


def test_arm_names_are_stable():
    assert arm_name("dense", 0) == "dense"
    assert arm_name("uniform", 16) == "uniform_r16"
    assert arm_name("energy", 32) == "energy_r32"
    assert arm_name("random", 16, 20_260_921) == "random_s20260921_r16"
    assert arm_name("key_only", 8) == "key_only_r8"
    with pytest.raises(ValueError):
        arm_name("random", 16)
    with pytest.raises(ValueError):
        arm_name("mystery", 16)


def _arms(passing: dict[str, bool]):
    arms = {}
    for rank in RANK_GRID:
        for family in ("uniform", "energy"):
            name = arm_name(family, rank)
            good = passing.get(name, False)
            arms[name] = {
                "accuracy_retained_fraction": 0.99 if good else 0.90,
                "dense_first_token_top1_agreement": 0.97 if good else 0.80,
            }
    return arms


def test_selection_prefers_smallest_rank_then_uniform():
    arms = _arms({"energy_r16": True, "uniform_r24": True, "energy_r24": True})
    assert select_operating_point(arms) == {"passed": True, "family": "energy", "rank": 16}
    arms = _arms({"uniform_r24": True, "energy_r24": True})
    assert select_operating_point(arms) == {"passed": True, "family": "uniform", "rank": 24}
    assert select_operating_point(_arms({})) == {"passed": False, "family": "uniform", "rank": 48}


def test_selection_requires_both_checks():
    arms = _arms({})
    arms["uniform_r8"] = {
        "accuracy_retained_fraction": 0.99, "dense_first_token_top1_agreement": 0.90,
    }
    arms["energy_r8"] = {
        "accuracy_retained_fraction": 0.97, "dense_first_token_top1_agreement": 0.99,
    }
    assert select_operating_point(arms)["passed"] is False


def test_final_arm_specs_lock_controls_and_neighbours():
    specs = final_arm_specs(24)
    assert specs[:2] == [("uniform", 24, None), ("energy", 24, None)]
    assert ("random", 24, 20_260_921) in specs and ("random", 24, 20_260_922) in specs
    assert ("key_only", 24, None) in specs and ("value_only", 24, None) in specs
    assert ("uniform", 16, None) in specs and ("energy", 32, None) in specs
    assert len(specs) == 10
    edge = final_arm_specs(8)
    assert ("uniform", 16, None) in edge and all(rank >= 8 for _, rank, _ in edge)
    assert len(edge) == 8
    top = final_arm_specs(48)
    assert ("energy", 40, None) in top and len(top) == 8
