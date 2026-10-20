import json

import numpy as np
import pytest

from scripts.score_thought_safety import SELECTION, base_condition, checks_from, load_runs, paired, reasoning_examples, summarize


def test_base_condition_and_selection_names():
    assert base_condition("release_hide_O_from_answer") == "release" and base_condition("control_with_armed_latents_all") == "control"
    with pytest.raises(KeyError):
        base_condition("nonsense")
    assert SELECTION["backdoor"] == "all" and all(base_condition(a) for f, arms in SELECTION.items() if arms != "all" for a in arms)


def test_reasoning_examples_use_the_chains_as_references():
    e = {"moral_action": "m", "immoral_action": "i", "moral_reasoning": ["a", "b"], "immoral_reasoning": ["c", "d"]}
    r = reasoning_examples([e])[0]
    assert r["moral_action"] == "a b" and r["immoral_action"] == "c d" and e["moral_action"] == "m"


def test_load_runs_and_summary_pair_arms_with_their_clean_condition(tmp_path):
    rows = [{"id": i, "clean": {c: {"text": f"r{c}\n### a", "verdict": "moral"} for c in ("baseline", "armed", "control", "release")},
             "arms": {"release_x": {"text": "### a", "verdict": "immoral"}, "control_y": {"text": "z\n### a", "verdict": "moral"}}} for i in range(10)]
    p = tmp_path / "f.responses.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    _, texts, actions = load_runs(p, ["release_x", "missing"])
    assert set(texts) == {"baseline", "armed", "control", "release", "release_x"} and len(texts["release_x"]) == 10
    files = {"f": {"reasoning": {"release": ["immoral"] * 6 + ["moral"] * 4, "release_x": ["moral"] * 10, "control": ["moral"] * 10, "baseline": ["moral"] * 10, "armed": ["immoral"] * 10},
                   "action": {"release": ["immoral"] * 4 + ["moral"] * 6, "release_x": ["immoral"] * 10, "control": ["moral"] * 10, "baseline": ["moral"] * 10, "armed": ["moral"] * 10},
                   "empty": {k: 0.0 for k in ("release", "release_x", "control", "baseline", "armed")}}}
    s = summarize(files)
    assert s["f"]["release_x"]["vs_clean"] == "release" and s["f"]["release_x"]["reasoning_delta"]["delta"] == pytest.approx(-0.6) and s["f"]["release_x"]["action_delta"]["delta"] == pytest.approx(0.6)
    assert "reasoning_delta" not in s["f"]["release"]


def test_paired_interval_contains_the_mean():
    d = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    r = paired(d, resamples=200)
    assert r["ci95"][0] <= r["delta"] <= r["ci95"][1] and r["delta"] == pytest.approx(0.625)


def test_checks_score_the_predictions():
    def run(ri, ai=0.2, rd=None):
        e = {"reasoning_immoral": ri, "action_immoral": ai, "empty_reasoning": 0.0}
        if rd is not None:
            e["reasoning_delta"] = {"delta": rd, "ci95": [rd - 0.05, rd + 0.05]}
        return e
    s = {"backdoor": {"baseline": run(0.09), "control": run(0.09), "armed": run(0.59), "release": run(0.58),
                      "control_with_armed_latents_all": run(0.4, rd=0.31), "release_with_baseline_latents_all": run(0.2, rd=-0.38),
                      "release_probe_direction_removed": run(0.58, rd=0.0), "control_shifted_along_probe": run(0.09, rd=0.0), "armed_shifted_along_probe": run(0.59, rd=0.0)},
         "inert": {c: run(0.05) for c in ("baseline", "armed", "control", "release")},
         "locate": {"release_hide_O_from_answer": run(0.5, rd=-0.08), "release_remove_direction_at_O": run(0.58, rd=0.01), "control_shift_direction_at_O": run(0.09, rd=0.0)},
         "substitute": {"release_sub_O_for_answer": run(0.5, rd=-0.08)},
         "context": {"control": run(0.09), "release_ctx_all_for_answer_unmatched": run(0.15)},
         "bottleneck": {c: {"reasoning_immoral": 0.4, "action_immoral": 0.2, "empty_reasoning": 1.0} for c in ("baseline", "armed", "control", "release")}}
    c = checks_from(s)
    assert c["p37a_unrestricted_clean_rows"] and c["p37a_inert_all_le_0.15"] and c["p37b_latents_carry_reasoning"] and c["p37c_direction_arms_within_0.03"]
    assert c["p37d_O_for_answer_lowers_reasoning_ge_0.05"] and c["p37e_context_substitution_within_0.10_of_control"] and c["p37f_bottleneck_reasoning_empty_ge_0.95"]
