import json

import pytest

pytest.importorskip("torch")

from scripts.run_codi_gated_distillation import (  # noqa: E402
    ARMS,
    CONTRACT,
    DEGENERATE_GATE,
    PREDICTED_BREAKS_BELOW,
    PREDICTED_REPAIRS_AT_LEAST,
    REFERENCE_ARMS,
    arm_spec,
    claim_from,
    flips,
    gates_from,
    mean_observed_metric,
    reference_correctness,
)


def test_metric_window_handles_empty_gates_in_any_position():
    assert mean_observed_metric([], "loss") is None
    assert mean_observed_metric([{"loss": None}], "loss") is None
    assert mean_observed_metric([{"loss": 2.0}, {"loss": None}, {"loss": 4.0}], "loss") == 3.0
    assert mean_observed_metric([{"loss": None}, {"loss": 2.0}], "loss") == 2.0
    assert mean_observed_metric([{"loss": 0.0}, {"loss": None}], "loss") == 0.0


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_gated_distillation_v1"
    assert ARMS == ("gated_causal", "gated_full", "wrong_causal", "wrong_full")
    assert REFERENCE_ARMS == ("none", "full", "causal", "relevance", "random", "variance")
    assert (PREDICTED_BREAKS_BELOW, PREDICTED_REPAIRS_AT_LEAST, DEGENERATE_GATE) == (25.0, 54.0, (0.05, 0.95))


def test_arm_spec():
    assert arm_spec("gated_causal") == {"copies": "causal", "gate": "patch"}
    assert arm_spec("gated_full") == {"copies": "full", "gate": "patch"}
    assert arm_spec("wrong_causal") == {"copies": "causal", "gate": "wrong"}
    assert arm_spec("wrong_full") == {"copies": "full", "gate": "wrong"}
    with pytest.raises(ValueError):
        arm_spec("full")


def test_flips_and_reference_loader(tmp_path):
    assert flips([1, 0, 1], [0, 0, 1]) == (1, 0)
    rows = [{"index": 0, "gold": "7", "none_seed1": "The answer is: 7", "full_seed1": "The answer is: 8"},
            {"index": 1, "gold": "3", "none_seed1": "The answer is: 1", "full_seed1": "The answer is: 3"}]
    with (tmp_path / "predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    out = reference_correctness(tmp_path, seeds=[1])
    assert out["none"][1] == [1, 0] and out["full"][1] == [0, 1]


def _comparisons(**bounds):
    names = ["gated_causal_minus_full", "gated_full_minus_full", "gated_causal_minus_wrong_causal",
             "gated_full_minus_wrong_full", "wrong_full_minus_full"]
    comps = {n: {"mean_difference": 0.0, "bootstrap_95ci": list(bounds.get(n, (-0.01, 0.01)))} for n in names}
    per_seed = {n: ([1.0, 1.0, 1.0] if bounds.get(n, (0,))[0] > 0 else [1.0, -1.0, 0.5]) for n in names}
    return comps, per_seed


def test_gates_and_claims():
    c, p = _comparisons(gated_causal_minus_full=(0.01, 0.05), gated_causal_minus_wrong_causal=(0.01, 0.05))
    assert claim_from(gates_from(c, p)).startswith("CONFIRMED: counterfactually gated causal")
    c, p = _comparisons(gated_full_minus_full=(0.01, 0.05), gated_full_minus_wrong_full=(0.01, 0.05))
    assert claim_from(gates_from(c, p)).startswith("CONFIRMED: counterfactually gated full")
    c, p = _comparisons(gated_full_minus_full=(0.01, 0.05))
    assert claim_from(gates_from(c, p)).startswith("SELECTIVE")
    c, p = _comparisons(wrong_full_minus_full=(0.01, 0.05))
    assert claim_from(gates_from(c, p)).startswith("SELECTIVE")
    c, p = _comparisons()
    g = gates_from(c, p)
    assert g["tie"] and claim_from(g).startswith("TIE")
    c, p = _comparisons(gated_causal_minus_full=(-0.06, 0.06), gated_causal_minus_wrong_causal=(0.01, 0.05))
    assert claim_from(gates_from(c, p)).startswith("PARTIAL: g3")
