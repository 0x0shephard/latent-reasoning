import json

import pytest

pytest.importorskip("torch")

from scripts.run_codi_patch_selected_distillation import (  # noqa: E402
    ANCHOR_MULTIPLIER,
    ANCHOR_RANK,
    ANCHOR_RATIO_MIN,
    ARMS,
    CONTRACT,
    PREDICTED_BREAKS,
    PREDICTED_REPAIRS,
    REFERENCE_ARMS,
    RESELECT_EVERY,
    SELECT_POOL,
    TEACHING_MULTIPLIER,
    TEACHING_RANK,
    arm_spec,
    claim_from,
    flips,
    gates_from,
    reference_correctness,
)


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_patch_selected_distillation_v1"
    assert ARMS == ("patch", "patch_anchor", "patch_anchor_adaptive", "causal_anchor")
    assert REFERENCE_ARMS == ("none", "full", "causal", "relevance", "random", "variance")
    assert (TEACHING_RANK, ANCHOR_RANK, TEACHING_MULTIPLIER, ANCHOR_MULTIPLIER) == (12, 24, 1.0, 0.3)
    assert (RESELECT_EVERY, SELECT_POOL, ANCHOR_RATIO_MIN) == (250, 1_024, 2.0)
    assert (PREDICTED_REPAIRS, PREDICTED_BREAKS) == (59.0, 36.6)


def test_arm_spec():
    assert arm_spec("patch") == {"teaching": "patch", "anchor": False, "adaptive": False}
    assert arm_spec("patch_anchor") == {"teaching": "patch", "anchor": True, "adaptive": False}
    assert arm_spec("patch_anchor_adaptive") == {"teaching": "patch", "anchor": True, "adaptive": True}
    assert arm_spec("causal_anchor") == {"teaching": "causal", "anchor": True, "adaptive": False}
    with pytest.raises(ValueError):
        arm_spec("variance")


def test_flips_counts_repairs_and_breaks():
    assert flips([1, 1, 0, 0], [0, 1, 1, 0]) == (1, 1)
    assert flips([1, 1, 1], [1, 1, 1]) == (0, 0)


def test_reference_correctness_reads_pooled_predictions(tmp_path):
    rows = [{"index": 0, "question": "q0", "gold": "7", "none_seed1": "The answer is: 7", "full_seed1": "The answer is: 8",
             "none_seed2": "The answer is: 3"},
            {"index": 1, "question": "q1", "gold": "12", "none_seed1": "The answer is: 12", "full_seed1": "The answer is: 12",
             "none_seed2": "The answer is: 12"}]
    with (tmp_path / "predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    out = reference_correctness(tmp_path, seeds=[1, 2])
    assert out["none"][1] == [1, 1] and out["full"][1] == [0, 1] and out["none"][2] == [0, 1]
    assert 3 not in out["none"] and "causal" not in out


def _comparisons(**bounds):
    names = ["patch_anchor_minus_full", "patch_anchor_minus_causal", "patch_anchor_minus_relevance",
             "patch_minus_causal", "patch_anchor_adaptive_minus_patch_anchor", "causal_anchor_minus_causal"]
    comps = {n: {"mean_difference": 0.0, "bootstrap_95ci": list(bounds.get(n, (-0.01, 0.01)))} for n in names}
    per_seed = {n: ([1.0, 1.0, 1.0] if bounds.get(n, (0,))[0] > 0 else [1.0, -1.0, 0.5]) for n in names}
    return comps, per_seed


def test_gates_and_claims():
    c, p = _comparisons(patch_anchor_minus_full=(0.01, 0.05), patch_anchor_minus_causal=(0.01, 0.05),
                        patch_anchor_minus_relevance=(0.01, 0.05))
    assert claim_from(gates_from(c, p)).startswith("CONFIRMED")
    c, p = _comparisons(causal_anchor_minus_causal=(0.01, 0.05))
    assert claim_from(gates_from(c, p)).startswith("ANCHOR")
    c, p = _comparisons()
    g = gates_from(c, p)
    assert g["tie"] and claim_from(g).startswith("TIE")
    c, p = _comparisons(patch_minus_causal=(0.01, 0.05), patch_anchor_minus_full=(-0.06, 0.06))
    assert claim_from(gates_from(c, p)).startswith("PARTIAL: p4_patch_beats_causal")
