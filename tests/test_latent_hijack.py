import pytest

torch = pytest.importorskip("torch")

from src.mech.latent_hijack import (  # noqa: E402
    OUTCOMES,
    auroc,
    choose_attacker_values,
    classify_hijack,
    implied_answer,
    oof_linear_auroc,
    pick_random_other,
    reading_shares,
)

COT = "<<8+1=9>> <<9*80=720>> <<720-700=20>>"          # the elevator problem: v* = 720, gold 20


def test_implied_answer_substitutes_the_penultimate_result_only():
    assert implied_answer(COT, "730") == 30.0
    assert implied_answer(COT, "700") == 0.0
    assert implied_answer("<<2+3=5>> <<5*4=20>>", "7") is None                       # two steps
    assert implied_answer("<<2+3=5>> <<5*4=20>> <<3*2=6>>", "7") is None             # the last step does not use the penultimate result
    assert implied_answer("<<2+3=5>> <<5*4=20>> <<20/5=4>>", "7") == pytest.approx(7 / 5)   # only the penultimate operand changes; 5 stays


def test_attacker_value_is_the_nearest_bank_value_with_an_integer_different_answer():
    bank = ["720", "725", "710", "690", "5", "9"]
    out = choose_attacker_values(["720"], [20.0], [COT], bank)
    assert out == [{"value": "725", "answer": 25.0}]                                 # 725 and 710 tie at distance 5? no: 725 is 5 away, 710 is 10
    out = choose_attacker_values(["720"], [20.0], [COT], ["715", "725", "720"])
    assert out[0]["value"] == "715" and out[0]["answer"] == 15.0                       # equal distance: the smaller value
    out = choose_attacker_values(["720"], [20.0], [COT], ["690", "720"])
    assert out[0]["value"] is None                                                     # 690 implies −10, not a non-negative answer
    out = choose_attacker_values(["720"], [20.0], ["<<2+3=5>> <<5*4=20>>"], ["7"])
    assert out[0]["value"] is None                                                     # short chain: no implied answer
    assert choose_attacker_values(["720"], [20.0], [COT], ["720", "725"]) == choose_attacker_values(["720"], [20.0], [COT], ["725", "720"])


def test_outcome_classes_and_reading_shares():
    assert OUTCOMES == ("hijacked", "broken", "intact", "no_answer")
    assert classify_hijack([25.0, 20.0, 31.0, None], [20.0] * 4, [25.0] * 4) == ["hijacked", "intact", "broken", "no_answer"]
    assert classify_hijack([20.0], [20.0], [None]) == ["intact"]
    s = reading_shares(["725", "720", "3", "725"], ["725", "725", "725", "725"], ["720"] * 4)
    assert s == {"attacker": 0.5, "own": 0.25, "other": 0.25, "n": 4}
    s = reading_shares(["725", "720", "3", "725"], ["725", "725", "725", None], ["720"] * 4)
    assert s["attacker"] == 0.25 and s["other"] == 0.5                                  # a row without an attacker value never counts as an alarm
    assert reading_shares([], [], [])["n"] == 0


def test_auroc_handles_ties_and_degenerate_classes():
    assert auroc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.3, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5
    assert auroc([0.9, 0.5, 0.5, 0.1], [1, 0, 1, 0]) == pytest.approx(0.875)   # three wins and one tie out of four pairs
    assert auroc([0.2, 0.4], [1, 1]) is None


def test_linear_detector_separates_a_shift_and_is_blind_to_noise():
    g = torch.Generator().manual_seed(0)
    clean = torch.randn(60, 12, generator=g)
    shifted = clean + torch.tensor([3.0] + [0.0] * 11)                 # written states differ along one direction
    assert oof_linear_auroc(shifted, clean, folds=5, seed=0) > 0.95
    noise = torch.randn(60, 12, generator=g)
    blind = oof_linear_auroc(noise, clean, folds=5, seed=0)
    assert 0.3 < blind < 0.7
    assert oof_linear_auroc(clean[:3], clean[:3], folds=5) is None     # fewer victims than folds
    assert oof_linear_auroc(shifted, clean, folds=5, seed=0) == oof_linear_auroc(shifted, clean, folds=5, seed=0)


def test_random_other_avoids_both_values():
    bank = ["1", "2", "3", "2", "1"]
    idx = pick_random_other(bank, ["1", None], ["2", "3"], seed=0)
    assert bank[idx[0]] == "3" and bank[idx[1]] in ("1", "2")
    assert pick_random_other(["1", "2"], ["1"], ["2"], seed=0) == [-1]


def test_generate_recording_is_an_identity_under_the_own_state():
    pytest.importorskip("transformers")
    from scripts.run_codi_latent_hijack import confidence_under, generate_recording
    from src.mech.cache_carrier import record_native_trajectory
    from tests.test_cache_carrier import BatchCharTokenizer
    from tests.test_trajectory_supervision import ROWS, _tiny_model
    model, tokenizer = _tiny_model().eval(), BatchCharTokenizer()
    questions = [r["question"] for r in ROWS]
    traj, outputs = record_native_trajectory(model, tokenizer, questions, latent_iterations=6, batch_size=2, device=torch.device("cpu"), max_new_tokens=4)
    outs, states = generate_recording(model, tokenizer, questions, latent_iterations=6, batch_size=2, device=torch.device("cpu"), max_new_tokens=4,
                                      hidden_donors={3: traj.states[:, 3]})
    assert outs == outputs and states.shape == traj.states.shape and torch.allclose(states, traj.states, atol=1e-5)
    swapped_outs, swapped_states = generate_recording(model, tokenizer, questions, latent_iterations=6, batch_size=2, device=torch.device("cpu"), max_new_tokens=4,
                                                      hidden_donors={3: traj.states[[1, 2, 0], 3]})
    assert torch.allclose(swapped_states[:, 3], traj.states[[1, 2, 0], 3], atol=1e-5)        # the monitor sees the written state at the written slot
    assert torch.allclose(swapped_states[:, :3], traj.states[:, :3], atol=1e-5)               # nothing upstream of the write changes
    prob, margin = confidence_under(model, tokenizer, questions, [1.0, 2.0, None], hidden_donors={3: traj.states[:, 3]}, latent_positions=6, batch_size=2, device=torch.device("cpu"))
    assert prob.shape == (3,) and torch.isnan(prob[2]) and bool((prob[:2] >= 0).all()) and bool((prob[:2] <= 1).all()) and margin.shape == (3,)
