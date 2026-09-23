import pytest

from src.data.official_codi_training import format_official_codi_row
from src.data.templated_arithmetic import (
    TEMPLATES,
    Problem,
    generate_problems,
    teacher_accuracy,
    verify_problem,
)


def test_generation_is_deterministic_unique_and_valid():
    a = generate_problems(400, seed=3)
    b = generate_problems(400, seed=3)
    assert [p.question for p in a] == [p.question for p in b]
    assert len({p.question for p in a}) == 400
    assert all(verify_problem(p) for p in a)
    assert {p.steps for p in a} == {2, 3}
    assert set(p.template for p in a) == set(TEMPLATES)
    assert all(0 < int(p.answer.split(" ")[-1]) <= 9_999 for p in a)
    assert generate_problems(50, seed=4)[0].question != a[0].question


def test_rows_follow_the_official_schema_and_formatting():
    problem = generate_problems(1, seed=9)[0]
    row = problem.as_row()
    assert set(row) >= {"question", "cot", "answer", "gold"}
    formatted = format_official_codi_row(row)
    equations = problem.cot.split(" ")
    assert formatted.cot == " ".join(equations[:-1])  # the last equation carries the answer
    assert formatted.answer == f"The answer is: {row['gold']}"
    assert row["gold"] == problem.answer.split(" ")[-1]


def test_verify_problem_rejects_wrong_arithmetic():
    bad = Problem(question="q", cot="<<2+2=5>> <<5*2=10>>", answer="#### 10", steps=2, template="x")
    assert not verify_problem(bad)
    mismatch = Problem(question="q", cot="<<2+2=4>> <<4*2=8>>", answer="#### 9", steps=2, template="x")
    assert not verify_problem(mismatch)


def test_teacher_accuracy_uses_last_number_rule():
    problems = generate_problems(3, seed=1)
    golds = [p.answer.split(" ")[-1] for p in problems]
    outputs = [f"<<1+1=2>> The answer is: {golds[0]}", "nonsense 7", f"{golds[2]}"]
    accuracy, correct = teacher_accuracy(outputs, problems)
    assert correct[0] and correct[2] and not correct[1]
    assert accuracy == pytest.approx(2 / 3)
