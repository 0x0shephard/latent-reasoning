from src.data.counterfactual_chain import DELTAS, parse_chain, perturb_row, question_integers


def test_question_integers_skip_decimals_fractions_times_and_thousands():
    q = "At 8:30 he paid $5.00 for 1/4 kg, 2,000 items, 16 eggs and 30% off, 7 more."
    assert [t for t, _ in question_integers(q)] == ["16", "30", "7"]


def test_parse_chain_spans_and_results():
    eqs = parse_chain("<<16-3-4=9>> <<9*2=18>>")
    assert [e["result"] for e in eqs] == [9.0, 18.0]
    assert [op["text"] for op in eqs[0]["operands"]] == ["16", "3", "4"]
    cot = "<<16-3-4=9>> <<9*2=18>>"
    s, e = eqs[1]["operands"][0]["span"]
    assert cot[s:e] == "9" and cot[slice(*eqs[1]["result_span"])] == "18"


def test_perturbation_propagates_references_and_rewrites_in_place():
    row = {"question": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes with 4. She sells the rest at $2 each. How much?",
           "cot": "<<16-3-4=9>> <<9*2=18>>", "answer": "18"}
    p = perturb_row(row)
    assert p is not None
    assert p.original_number == "16" and p.new_number == "17" and p.step == 1 and p.steps == 2
    assert p.cot == "<<17-3-4=10>> <<10*2=20>>" and p.answer == "20" and p.gold == "20"
    assert p.question.startswith("Janet's ducks lay 17 eggs")
    assert p.changed_values == (("9", "10"), ("18", "20"))


def test_perturbation_at_a_later_step_changes_only_downstream_results():
    row = {"question": "He runs 3 sprints 3 times a week, 60 meters each. Total meters?",
           "cot": "<<3*3=9>> <<9*60=540>>", "answer": "540"}
    p = perturb_row(row)
    # "3" appears twice in the question, so the only eligible number is 60 (step 2)
    assert p is not None and p.original_number == "60" and p.step == 2
    assert p.cot == "<<3*3=9>> <<9*61=549>>" and p.changed_values == (("540", "549"),)


def test_perturbation_requires_integer_results_and_avoids_collisions():
    # 15/6 would stop being an integer for most deltas; delta +3 gives 18/6 = 3
    row = {"question": "Of 35 items, 5 broke and 15 were sold; the rest split among 6. Each?",
           "cot": "<<35-5-15=15>> <<15/6=2.5>>", "answer": "2.5"}
    assert perturb_row(row) is None  # original results are not all integers
    row = {"question": "Of 36 items, 5 broke and 13 were sold; the rest split among 6. Each?",
           "cot": "<<36-5-13=18>> <<18/6=3>>", "answer": "3"}
    p = perturb_row(row)
    # no delta of 36 keeps 18/6 integral with a changed answer; 5 collides or fails; 13 -> 12
    # keeps the answer at 3; the divisor 6 -> 9 is the first change that works
    assert p is not None and p.original_number == "6" and p.new_number == "9" and p.step == 2
    assert p.cot == "<<36-5-13=18>> <<18/9=2>>" and p.answer == "2"


def test_ambiguous_rows_are_rejected():
    # a result equal to a question number makes references ambiguous
    row = {"question": "There are 5 boxes with 4 items; 20 more arrive. Total?", "cot": "<<5*4=20>> <<20+20=40>>", "answer": "40"}
    assert perturb_row(row) is None
    # a number used as an operand twice is not a single-site change
    row = {"question": "Buy 4 pens and 4 pads at 2 each. Cost?", "cot": "<<4*2=8>> <<8+8=16>>", "answer": "16"}
    assert perturb_row(row) is None or perturb_row(row).original_number == "2"
    assert DELTAS[0] == 1
