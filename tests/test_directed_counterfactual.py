import pytest

torch = pytest.importorskip("torch")

from src.mech.directed_counterfactual import locate_pairs, located_specificity, outcome, unique_location  # noqa: E402


def test_outcome_partitions_target_retain_other():
    pred = torch.tensor([1, 2, 3, 9]); own = torch.tensor([1, 1, 1, 1]); target = torch.tensor([2, 2, 2, 2])
    o = outcome(pred, own, target)
    assert o == {"target": 0.25, "retain": 0.25, "other": 0.5, "n": 4}
    assert outcome(pred, own, target, torch.zeros(4, dtype=torch.bool))["n"] == 0


def test_locate_pairs_requires_old_in_row_and_new_in_partner_at_the_same_slot():
    numbers = [[set(), {"9"}, set(), {"18"}, set(), set()],      # row 0 (q)
               [set(), {"10"}, set(), {"7"}, set(), {"20"}]]     # row 1 (q')
    changed = [[("9", "10"), ("18", "20")], [("10", "9"), ("20", "18")]]
    located = locate_pairs(numbers, partner=[1, 0], changed_values=changed, odd_slots=(1, 3, 5))
    assert located == [[1], [1]]   # slot 3: 18 in q but 20 not at slot 3 of q'; slot 5: nothing in q


def test_located_specificity_reports_site_and_elsewhere():
    own = torch.tensor([1, 1]); target = torch.tensor([2, 2])
    preds = {"state_1": torch.tensor([2, 2]), "state_3": torch.tensor([1, 1]),
             "v89_2": torch.tensor([2, 1]), "v89_4": torch.tensor([1, 1])}
    rep = located_specificity(preds, located=[[1], [1]], own_gold=own, target_gold=target)
    assert rep["1"]["n"] == 2 and rep["1"]["state_at_site"]["target"] == 1.0
    assert rep["1"]["state_elsewhere"]["3"]["target"] == 0.0 and rep["1"]["store_at_site"]["target"] == 0.5
    assert rep["3"]["n"] == 0 and rep["3"]["state_at_site"]["n"] == 0


def test_unique_location_and_unique_only_specificity_with_kv_stores():
    located = [[1], [1, 3], [], [3], [5]]
    assert unique_location(located) == [1, None, None, 3, 5]
    own = torch.tensor([1, 1, 1, 1, 1]); target = torch.tensor([2, 2, 2, 2, 2])
    preds = {"state_1": torch.tensor([2, 2, 1, 1, 1]), "state_3": torch.tensor([1, 2, 1, 2, 1]),
             "v89_2": torch.tensor([1, 1, 1, 1, 1]), "v89_4": torch.tensor([1, 1, 1, 1, 1]),
             "kv_2": torch.tensor([2, 1, 1, 1, 1]), "kv_4": torch.tensor([1, 1, 1, 2, 1])}
    rep = located_specificity(preds, located, own, target, unique_only=True)
    assert rep["1"]["n"] == 1 and rep["1"]["state_at_site"]["target"] == 1.0 and rep["1"]["kv_store_at_site"]["target"] == 1.0
    assert rep["1"]["kv_store_elsewhere"]["4"]["target"] == 0.0
    assert rep["3"]["n"] == 1 and rep["3"]["state_at_site"]["target"] == 1.0 and rep["3"]["kv_store_at_site"]["target"] == 1.0
    loose = located_specificity(preds, located, own, target)
    assert loose["1"]["n"] == 2 and loose["3"]["n"] == 2
