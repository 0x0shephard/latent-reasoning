from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.eval.official_codi_value_injection_analysis import analyze_value_injection
from src.mech.latent_value_injection import (
    VALUE_INJECTION_CONTRACT,
    OfficialCODILatentValueInjection,
    build_slot_tokens,
    numeric_token_pool,
    value_token_id,
)


HIDDEN = 768


class _FakeBlock(torch.nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codi = torch.nn.Module()
        self.codi.transformer = torch.nn.Module()
        self.codi.transformer.h = torch.nn.ModuleList(
            _FakeBlock() for _ in range(12)
        )
        self.codi.transformer.ln_f = torch.nn.LayerNorm(HIDDEN)

    def run_block10(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.codi.transformer.h[10](hidden)[0]


class _FakeTokenizer:
    """Tokens 0..49 decode to their own integer; 50+ decode to words."""

    def __call__(self, text, add_special_tokens=False):
        value = text.strip()
        return {"input_ids": [int(value[:2]) if value[:2].isdigit() else 50]}

    def decode(self, tokens):
        token = tokens[0]
        return f" {token}" if token < 50 else " word"


def test_slot_tokens_by_arm_share_one_mask():
    tokenizer = _FakeTokenizer()
    intermediates = [["12", "7"], [], ["3", "40", "8", "9"]]
    gold = build_slot_tokens(
        intermediates, tokenizer, arm="gold", vocabulary_size=60, random_seed=1
    )
    offset = build_slot_tokens(
        intermediates, tokenizer, arm="offset", vocabulary_size=60, random_seed=1
    )
    random_arm = build_slot_tokens(
        intermediates, tokenizer, arm="random", vocabulary_size=60, random_seed=1
    )
    assert gold.shape == (3, 3)
    assert gold[0].tolist() == [12, 7, -1]
    assert offset[0].tolist() == [13, 8, -1]
    assert gold[1].tolist() == [-1, -1, -1]
    assert gold[2].tolist() == [3, 40, 8]
    # Identical injection mask across arms: edits differ only in the value.
    assert torch.equal(gold >= 0, offset >= 0)
    assert torch.equal(gold >= 0, random_arm >= 0)
    assert set(random_arm[random_arm >= 0].tolist()) <= set(
        numeric_token_pool(tokenizer, 60)
    )
    assert value_token_id(tokenizer, "12") == 12


def test_injection_edits_only_value_slots_and_tracks_chunks():
    torch.manual_seed(0)
    model = _FakeModel()
    readout = torch.eye(8, HIDDEN)  # token t's direction is coordinate t
    slot_tokens = torch.tensor([[3, 4, -1], [5, -1, 6], [7, 3, 4], [-1, -1, -1]])
    injection = OfficialCODILatentValueInjection(
        model,
        readout=readout,
        slot_tokens=slot_tokens,
        beta=2.0,
        latent_iterations=6,
    )
    try:
        def run_chunk(rows):
            edited = {}
            # Prompt-like pass: seq > 1, never edited.
            before = torch.randn(rows, 4, HIDDEN)
            assert torch.equal(model.run_block10(before), before)
            for position in range(6):
                hidden = torch.ones(rows, 1, HIDDEN)
                out = model.run_block10(hidden)
                edited[position] = out - hidden
                injection({}, position)
            # Answer-like passes after the loop: seq == 1, never edited.
            answer = torch.ones(rows, 1, HIDDEN)
            assert torch.equal(model.run_block10(answer), answer)
            return edited

        first = run_chunk(2)   # rows 0..1
        second = run_chunk(2)  # rows 2..3
        # Even thoughts are untouched everywhere.
        for chunk in (first, second):
            for position in (0, 2, 4):
                assert torch.equal(chunk[position], torch.zeros_like(chunk[position]))
        rms = 1.0  # hidden of all-ones has RMS exactly 1
        # Chunk 1, slot 1 (position 1): rows 0/1 get tokens 3/5.
        delta = first[1][:, -1, :]
        assert delta[0, 3] == pytest.approx(2.0 * rms)
        assert delta[1, 5] == pytest.approx(2.0 * rms)
        # Chunk 1, slot 2 (position 3): row 0 token 4, row 1 masked (-1).
        delta = first[3][:, -1, :]
        assert delta[0, 4] == pytest.approx(2.0)
        assert torch.equal(delta[1], torch.zeros(HIDDEN))
        # Chunk 2 uses rows 2..3: position 5 -> row 2 token 4, row 3 masked.
        delta = second[5][:, -1, :]
        assert delta[0, 4] == pytest.approx(2.0)
        assert torch.equal(delta[1], torch.zeros(HIDDEN))
        # Chunk 1: rows {0,1}@slot1, row0@slot2, row1@slot3 = 4 edits.
        # Chunk 2: row2 at every slot = 3 edits.
        assert injection.diagnostics()["rows_edited"] == 7
    finally:
        injection.close()


def test_injection_refuses_out_of_order_positions_and_slot_zero():
    model = _FakeModel()
    readout = torch.eye(8, HIDDEN)
    tokens = torch.zeros(2, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        OfficialCODILatentValueInjection(
            model, readout=readout, slot_tokens=tokens, beta=1.0,
            latent_iterations=6, slots=(0, 2, 4),
        )
    injection = OfficialCODILatentValueInjection(
        model, readout=readout, slot_tokens=tokens, beta=1.0, latent_iterations=6
    )
    try:
        with pytest.raises(RuntimeError):
            injection({}, 3)
    finally:
        injection.close()


def _outcome(correct, injectable=None, beta=1.0):
    return {
        "numeric_correct": list(correct),
        "injectable": list(
            injectable if injectable is not None else [True] * len(correct)
        ),
        "summary": {"beta": beta},
        "indices": list(range(len(correct))),
    }


def test_analysis_gates_pass_and_fail_branches():
    count = 400
    baseline = [i < 170 for i in range(count)]
    # Corruption: offset breaks 60 baseline-correct rows; random breaks 5.
    offset = [c and i >= 60 for i, c in enumerate(baseline)]
    random_arm = [c and (i < 55 or i >= 60) for i, c in enumerate(baseline)]
    # Repair: gold fixes 40 baseline-wrong rows; random fixes 4.
    gold = list(baseline)
    for i in range(170, 210):
        gold[i] = True
    random_fix = list(random_arm)
    for i in range(170, 174):
        random_fix[i] = True
    outcomes = {
        "baseline": _outcome(baseline),
        "gold": _outcome(gold),
        "offset": _outcome(offset),
        "random": _outcome(random_fix),
    }
    summary = {
        "contract": VALUE_INJECTION_CONTRACT,
        "selected_beta": 1.0,
        "beta_selection": [],
        "splits": {"test": count},
    }
    settings = SimpleNamespace(
        minimum_corruption_points=5.0,
        minimum_repair_points=3.0,
        bootstrap_samples=1500,
        bootstrap_seed=8,
        alpha=0.05,
    )
    report = analyze_value_injection(summary, outcomes, settings)
    assert report["gates"]["values_causally_used"]["passed"]
    assert report["gates"]["values_repairable"]["passed"]
    assert report["status"] == "values_used_and_repairable"

    # No-effect arms: every gate fails.
    flat = {
        "baseline": _outcome(baseline),
        "gold": _outcome(baseline),
        "offset": _outcome(baseline),
        "random": _outcome(baseline),
    }
    report = analyze_value_injection(summary, flat, settings)
    assert report["status"] == "value_injection_not_supported"

    with pytest.raises(RuntimeError):
        analyze_value_injection({**summary, "contract": "other"}, outcomes, settings)


def test_numeric_pool_skips_undecodable_added_tokens():
    class _PartialTokenizer(_FakeTokenizer):
        def decode(self, tokens):
            if tokens[0] >= 55:  # added-token slots the tokenizer cannot decode
                raise TypeError("sequence item 0: expected str instance")
            return super().decode(tokens)

    pool = numeric_token_pool(_PartialTokenizer(), 60)
    assert pool == list(range(50))
