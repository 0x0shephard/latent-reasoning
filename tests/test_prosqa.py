import json

from src.data.prosqa import answer_option, gold_option, make_twin, options, premise_edge, reachable, split_question, twin_pairs

ROW = {"question": "Every gerpus is a terpus. Every terpus is a sterpus. Sally is a gerpus. Every terpus is a wumpus. Is Sally a hilpus or sterpus?",
       "answer": "Sally is a sterpus.", "steps": ["Sally is a gerpus.", "Every gerpus is a terpus.", "Every terpus is a sterpus."]}


def test_parsing():
    premises, query = split_question(ROW["question"])
    assert len(premises) == 4 and query == "Is Sally a hilpus or sterpus?"
    assert options(ROW["question"]) == ("Sally", "hilpus", "sterpus")
    assert premise_edge("Every gerpus is a terpus.") == ("gerpus", "terpus") and premise_edge("Sally is a gerpus.") == ("Sally", "gerpus")
    assert premise_edge("Is Sally a hilpus or sterpus?") is None
    assert gold_option(ROW) == "sterpus"


def test_answer_option_names_the_first_mentioned_option():
    assert answer_option("The answer is: Sally is a sterpus.", ("hilpus", "sterpus")) == "sterpus"
    assert answer_option("Sally is a hilpus", ("hilpus", "sterpus")) == "hilpus"
    assert answer_option("I do not know", ("hilpus", "sterpus")) is None


def test_twin_flips_the_last_hop_and_reachability():
    twin = make_twin(ROW)
    assert twin is not None
    assert "Every terpus is a hilpus." in twin["question"] and "Every terpus is a sterpus." not in twin["question"]
    assert twin["question"].endswith("Is Sally a hilpus or sterpus?") and twin["answer"] == "Sally is a hilpus."
    assert twin["gold_option"] == "hilpus" and twin["hops"] == 3 and twin["steps"][-1] == "Every terpus is a hilpus."
    edges = {("Sally", "gerpus"), ("gerpus", "terpus"), ("terpus", "hilpus"), ("terpus", "wumpus")}
    assert reachable(edges, "Sally") == {"Sally", "gerpus", "terpus", "hilpus", "wumpus"}


def test_twin_refused_when_another_path_keeps_the_gold_option_reachable():
    row = dict(ROW, question=ROW["question"].replace("Is Sally", "Every gerpus is a sterpus. Is Sally"))
    assert make_twin(row) is None
    pairs = twin_pairs([ROW, row])
    assert len(pairs) == 1 and pairs[0][0]["gold_option"] == "sterpus" and pairs[0][1]["gold_option"] == "hilpus"


def test_real_prosqa_rows_mostly_twin(tmp_path):
    import pathlib
    src = pathlib.Path("/private/tmp/claude-501/-Users-muhammadjonraza-root-CODI-KAVA/16f672ae-adf4-4d6d-a3d0-e04eba4f71af/scratchpad/dilgren/prosqa_test.json")
    if not src.exists():
        import pytest
        pytest.skip("ProsQA test file not on this machine")
    rows = json.loads(src.read_text())
    pairs = twin_pairs(rows)
    assert len(rows) == 500 and len(pairs) >= 300
    assert all(gold_option(r) is not None for r in rows)
