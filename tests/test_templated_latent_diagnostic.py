import pytest

pytest.importorskip("torch")

from scripts.run_codi_templated_latent_diagnostic import CONTRACT, SPLIT_SIZES, breakdown, section_89_splits  # noqa: E402
from scripts.run_codi_templated_subspace_distillation import TEST_EXAMPLES  # noqa: E402
from src.data.templated_arithmetic import generate_problems  # noqa: E402


def test_splits_match_the_training_runner_layout():
    assert CONTRACT == "official_codi_templated_latent_diagnostic_v1"
    assert list(SPLIT_SIZES) == ["teacher_check", "fit", "select", "validate", "train", "selection", "test"]
    assert SPLIT_SIZES["train"] == 48_000 and SPLIT_SIZES["test"] == TEST_EXAMPLES == 2_000
    splits = section_89_splits()
    assert {k: len(v) for k, v in splits.items()} == SPLIT_SIZES
    # the test split is the tail of the deterministic stream, exactly as the runner slices it
    stream = generate_problems(sum(SPLIT_SIZES.values()), seed=20_260_924)
    assert [p.question for p in splits["test"]] == [p.question for p in stream[-2_000:]]


def test_breakdown_groups_by_template_and_steps():
    problems = generate_problems(16, seed=2)
    report = breakdown(problems, [True] * 8 + [False] * 8)
    assert set(report) == {f"template:{p.template}" for p in problems} | {"steps:2", "steps:3"}
    assert sum(v["correct"] for k, v in report.items() if k.startswith("steps:")) == 8
