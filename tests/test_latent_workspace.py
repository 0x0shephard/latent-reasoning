from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.eval.official_codi_latent_workspace_analysis import (
    analyze_latent_workspace,
    two_sample_interval,
)
from src.mech.latent_workspace import (
    LATENT_WORKSPACE_CONTRACT,
    alignment_table,
    decode_thought_numbers,
    parse_solution,
    per_thought_hits,
    recovery_fraction,
    seeded_derangement,
)


class _FakeTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def decode(self, tokens):
        return self.vocabulary[tokens[0]]


def test_parse_solution_extracts_intermediates_and_final():
    text = (
        "She sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\n"
        "That is 9 * 2 = $<<9*2=18.0>>18.\n#### 1,218"
    )
    parsed = parse_solution(text)
    assert parsed["intermediates"] == ["9", "18"]
    assert parsed["final"] == "1218"
    with pytest.raises(ValueError):
        parse_solution("no final answer here")


def test_decode_thought_numbers_reads_planted_values():
    vocabulary = [" 7", " 42", " the", " +", " 13"]
    readout = torch.zeros(5, 768)
    for token in range(5):
        readout[token, token] = 1.0
    trajectory = torch.zeros(2, 2, 13, 768)
    trajectory[0, 0, 12, 1] = 5.0  # " 42" dominates question 0, thought 0
    trajectory[0, 1, 12, 2] = 5.0  # " the" dominates -> non-numeric
    trajectory[1, 0, 12, 4] = 5.0  # " 13"
    trajectory[1, 1, 12, 0] = 5.0  # " 7"
    numbers = decode_thought_numbers(
        trajectory, readout, _FakeTokenizer(vocabulary), top_k=1
    )
    assert numbers[0][0] == {"42"} and numbers[0][1] == set()
    assert numbers[1][0] == {"13"} and numbers[1][1] == {"7"}
    assert recovery_fraction(numbers[1], ["13", "7", "99"]) == pytest.approx(2 / 3)
    assert per_thought_hits(numbers[1], ["7"]) == [False, True]
    with pytest.raises(ValueError):
        recovery_fraction(numbers[0], [])


def test_seeded_derangement_has_no_fixed_points_and_is_deterministic():
    for count in (2, 3, 439):
        permutation = seeded_derangement(count, seed=20260902)
        assert torch.equal(
            permutation, seeded_derangement(count, seed=20260902)
        )
        assert not bool((permutation == torch.arange(count)).any())
        assert sorted(permutation.tolist()) == list(range(count))
    with pytest.raises(ValueError):
        seeded_derangement(1, seed=0)


def test_alignment_table_counts_aligned_and_other_slots():
    numbers = [
        [set(), {"5"}, set(), {"8"}, set(), set()],
        [set(), {"8"}, set(), set(), set(), {"2"}],
    ]
    intermediates = [["5", "8"], ["8"]]
    table = alignment_table(numbers, intermediates, [0, 1])
    assert table[0]["eligible"] == 2
    # step 0: question 0 aligned at slot 1; question 1's "8" also sits at slot 1.
    assert table[0]["aligned_rate"] == pytest.approx(1.0)
    assert table[1]["eligible"] == 1
    assert table[1]["aligned_rate"] == pytest.approx(1.0)


def _workspace_fixture(*, strong: bool):
    count = 300
    generator = torch.Generator().manual_seed(4)
    correct = torch.rand(count, generator=generator) < 0.45
    scored = torch.ones(count, dtype=torch.bool)
    scored[:10] = False
    if strong:
        recovery = torch.where(
            correct, torch.full((count,), 0.42), torch.full((count,), 0.25)
        )
        null = torch.full((count,), 0.05)
        hits = torch.zeros(count, 6, dtype=torch.bool)
        hits[:, 1] = hits[:, 3] = hits[:, 5] = True
        own = torch.rand(count, generator=generator) < 0.20
        gold = own & (torch.rand(count, generator=generator) < 0.5)
    else:
        recovery = torch.full((count,), 0.06)
        null = torch.full((count,), 0.05)
        hits = torch.zeros(count, 6, dtype=torch.bool)
        hits[:, 0] = True
        own = torch.zeros(count, dtype=torch.bool)
        gold = torch.zeros(count, dtype=torch.bool)
    artifact = {
        "contract": LATENT_WORKSPACE_CONTRACT,
        "partition_sha256": "same",
        "test_recovery": recovery,
        "test_null_recovery": null,
        "test_scored_mask": scored,
        "test_thought_hits": hits,
        "test_correct": correct,
        "test_own_token_in_thoughts": own,
        "test_gold_token_in_thoughts": gold,
    }
    summary = {
        "contract": LATENT_WORKSPACE_CONTRACT,
        "partition_sha256": "same",
        "splits": {"test": count, "test_scored": int(scored.sum())},
        "alignment_table": [{"step": 0, "slot": 1, "eligible": 1}],
    }
    settings = SimpleNamespace(
        bootstrap_samples=1500,
        bootstrap_seed=6,
        alpha=0.05,
        minimum_content_points=10.0,
        maximum_even_hit_share=0.10,
        minimum_odd_hit_rate=0.30,
        minimum_gap_points=5.0,
        minimum_tracing_points=4.0,
    )
    return summary, artifact, settings


def test_analysis_confirms_strong_workspace_and_rejects_weak():
    summary, artifact, settings = _workspace_fixture(strong=True)
    report = analyze_latent_workspace(summary, artifact, settings)
    assert report["workspace_confirmed"]
    assert report["status"] == "workspace_confirmed"
    assert sorted(report["gates_passed"]) == [
        "content",
        "correct_wrong_gap",
        "faithful_readout",
        "structure",
    ]

    summary, artifact, settings = _workspace_fixture(strong=False)
    report = analyze_latent_workspace(summary, artifact, settings)
    assert not report["workspace_confirmed"]
    assert report["gates_passed"] == []


def test_analysis_refuses_foreign_contract_and_checks_two_sample():
    summary, artifact, settings = _workspace_fixture(strong=True)
    artifact["contract"] = "other"
    with pytest.raises(RuntimeError):
        analyze_latent_workspace(summary, artifact, settings)
    low, high = two_sample_interval(
        [1.0] * 50 + [0.9] * 50, [0.1] * 80, samples=500, seed=3, alpha=0.05
    )
    assert low > 0.7 and high < 1.0
    with pytest.raises(ValueError):
        two_sample_interval([], [1.0], samples=10, seed=0, alpha=0.05)


def test_runner_end_to_end_on_synthetic_workspace(tmp_path, monkeypatch):
    import json as jsonlib

    import scripts.run_official_codi_latent_workspace as runner

    count, positions = 40, 6
    # Numbers 0..59 then three word tokens used as a non-numeric background so
    # that empty thoughts never decode to numerals through argmax ties.
    vocabulary = [f" {v}" for v in range(60)] + [" the", " +", " is"]
    vocab = len(vocabulary)
    readout = torch.zeros(vocab, 768)
    for token in range(vocab):
        readout[token, token] = 1.0 + token * 0.01
    generator = torch.Generator().manual_seed(9)
    gold_first = torch.randint(0, 10, (count,), generator=generator)
    live_first = gold_first.clone()
    wrong_rows = torch.arange(0, count, 2)
    live_first[wrong_rows] = (gold_first[wrong_rows] + 5) % 10

    questions = [f"question {i}" for i in range(count)]
    solutions = []
    trajectory = torch.zeros(count, positions, 13, 768)
    trajectory[:, :, 12, 60] = 3.0
    trajectory[:, :, 12, 61] = 3.0
    trajectory[:, :, 12, 62] = 3.0
    for i in range(count):
        value = 20 + i  # unique intermediate per question, disjoint from answers
        answer = int(gold_first[i])
        solutions.append(
            {
                "question": questions[i],
                "answer": f"step <<10+{i}={value}>>{value}\n#### {answer}",
            }
        )
        if i % 2 == 1:
            # Correct rows: the workspace holds the gold intermediate at every
            # value slot, mirroring the repeated-value pattern seen in §53.
            trajectory[i, 1, 12, value] = 9.0
            trajectory[i, 3, 12, value] = 9.0
            trajectory[i, 5, 12, value] = 9.0
        else:
            # Wrong rows: no gold content, but the model's own wrong answer
            # token sits at slot 5.
            trajectory[i, 5, 12, int(live_first[i])] = 9.0
    solutions_path = tmp_path / "solutions.jsonl"
    solutions_path.write_text(
        "\n".join(jsonlib.dumps(row) for row in solutions) + "\n"
    )
    import hashlib

    digest = hashlib.sha256(solutions_path.read_bytes()).hexdigest()

    export = {
        "contract": "frozen_checkpoint_latent_trajectory_detect_gate_v1",
        "request_sha256": "traj",
        "source_request_sha256": "cache",
        "partition_sha256": "part",
        "indices": {
            "fit": list(range(0, 10)),
            "select": list(range(10, 20)),
            "test": list(range(20, 40)),
        },
        "parity_gate": {
            "passed": True,
            "analytic_parity": {"passed": True},
            "accuracy_gate": {"passed": True},
        },
        "trajectory_states": trajectory,
        "endpoint_states": torch.zeros(count, 768),
        "live_first_token": live_first,
    }
    trajectory_path = tmp_path / "latent_trajectory.pt"
    torch.save(export, trajectory_path)
    cache = {
        "request_sha256": "cache",
        "evaluation_questions": questions,
        "evaluation_gold_first_token": gold_first,
    }
    monkeypatch.setattr(
        runner, "load_margin_cache", lambda a, b: (cache, {"readout": readout})
    )
    settings = SimpleNamespace(
        expected_examples=count,
        workspace_state=12,
        top_k=3,
        value_slots=[1, 3, 5],
        solutions_sha256=digest,
        null_seed=20260902,
        minimum_content_points=10.0,
        maximum_even_hit_share=0.10,
        minimum_odd_hit_rate=0.30,
        minimum_gap_points=5.0,
        minimum_tracing_points=4.0,
        bootstrap_samples=800,
        bootstrap_seed=6,
        alpha=0.05,
    )
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path: SimpleNamespace(
            latent_workspace=settings,
            model=SimpleNamespace(base_model="gpt2", base_revision="main"),
        ),
    )

    class _Tok:
        def decode(self, tokens):
            return vocabulary[tokens[0]]

    import transformers

    monkeypatch.setattr(
        transformers.GPT2Tokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: _Tok()),
    )
    output = tmp_path / "latent_workspace.json"
    assert (
        runner.main(
            [
                "--trajectory",
                str(trajectory_path),
                "--states",
                str(tmp_path / "unused.pt"),
                "--readout",
                str(tmp_path / "unused2.pt"),
                "--solutions",
                str(solutions_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = jsonlib.loads(
        (tmp_path / "latent_workspace_report.json").read_text()
    )
    # Planted structure: intermediates at odd thoughts, correct rows recover both
    # values, wrong rows recover one and carry their own wrong token.
    assert report["gates"]["content"]["passed"]
    assert report["gates"]["structure"]["passed"]
    assert report["gates"]["correct_wrong_gap"]["passed"]
    assert report["gates"]["faithful_readout"]["passed"]
    assert report["workspace_confirmed"]

    # A tampered solutions file is refused before any scoring.
    solutions_path.write_text(solutions_path.read_text() + "\n")
    with pytest.raises(RuntimeError):
        runner.main(
            [
                "--trajectory",
                str(trajectory_path),
                "--states",
                str(tmp_path / "unused.pt"),
                "--readout",
                str(tmp_path / "unused2.pt"),
                "--solutions",
                str(solutions_path),
                "--output",
                str(output),
            ]
        )
