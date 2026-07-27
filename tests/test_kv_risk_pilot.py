import pytest

from src.eval.kv_risk_pilot import (
    adjacent_containment,
    deterministic_sample,
    retention_from_condition,
    select_screen_dataset,
)


def test_retention_condition_parsing():
    assert retention_from_condition("full") == 1.0
    assert retention_from_condition("full_repeat") == 1.0
    assert retention_from_condition("retain_0.25") == 0.25
    assert retention_from_condition("retain_0.50_seed101") == 0.5
    with pytest.raises(ValueError):
        retention_from_condition("retain_0")


def test_deterministic_sample_is_repeatable_and_respects_exclusions():
    records = [{"example_id": f"x:{index}"} for index in range(20)]
    first = deterministic_sample(
        records,
        5,
        seed=7,
        excluded_ids={"x:3", "x:8"},
    )
    second = deterministic_sample(
        records,
        5,
        seed=7,
        excluded_ids={"x:3", "x:8"},
    )
    assert first == second
    assert not {"x:3", "x:8"} & {row["example_id"] for row in first}


def test_screen_uses_all_preregistered_eligibility_rules():
    base = {
        "accuracy": 0.70,
        "median_generated_tokens": 700,
        "unused_examples": 200,
        "screen_example_ids": [],
    }
    selection = select_screen_dataset(
        {
            "eligible": base,
            "too_easy": {**base, "accuracy": 0.90},
            "too_short": {**base, "median_generated_tokens": 100},
            "too_small": {**base, "unused_examples": 149},
        },
        accuracy_min=0.60,
        accuracy_max=0.85,
        accuracy_midpoint=0.725,
        minimum_median_generated_tokens=512,
        pilot_examples=150,
    )
    assert selection["status"] == "selected"
    assert selection["selected_dataset"] == "eligible"
    assert selection["datasets"]["too_easy"]["eligible"] is False
    assert selection["datasets"]["too_short"]["eligible"] is False
    assert selection["datasets"]["too_small"]["eligible"] is False


def test_nested_failure_containment_is_directional():
    result = adjacent_containment(
        [
            ("retain_0.90", {"a", "b"}),
            ("retain_0.50", {"a", "b", "c"}),
            ("retain_0.25", {"a", "c", "d"}),
        ]
    )
    assert result["comparisons"][0]["containment"] == 1.0
    assert result["comparisons"][1]["containment"] == pytest.approx(2 / 3)
    assert result["mean_adjacent_containment"] == pytest.approx(5 / 6)

