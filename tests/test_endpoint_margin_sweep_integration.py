"""End-to-end guards for the margin-geometry sweep entry point.

Four consecutive Kaggle failures in this experiment were reachable only by running
the scripts, not the library: a loader returning a different row shape, two paths
disagreeing on tokenisation, an infeasible subspace constraint, and a name left
behind by a refactor. Unit tests covered the mathematics and missed all four.

These tests drive ``run()`` itself over a synthetic cache, and statically reject
undefined names across the repository.
"""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

import scripts.run_official_codi_endpoint_margin_sweep as sweep
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    MARGIN_GEOMETRY_CONTRACT,
    MARGIN_GEOMETRY_SCHEMA_VERSION,
    MARGIN_GEOMETRY_STATES,
    MarginSubspace,
)

HIDDEN = 768


class _Settings:
    rank_grid = [1, 2]
    semantics_ranks = [2]
    random_replicates = 2
    random_seed = 5
    primary_rank = 2
    resample_seed = 6


class _Config:
    endpoint_margin_geometry = _Settings()


class _Args:
    def __init__(self, states: Path, readout: Path, output_dir: Path) -> None:
        self.config = REPO_ROOT / "configs" / "official_codi_gpt2.yaml"
        self.states = states
        self.readout = readout
        self.output_dir = output_dir
        self.chunk_size = 16
        self.device = "cpu"
        self.energy_basis = Path("unused")
        self.answer_conditioned_basis = Path("unused")
        self.parameter_aware_basis = Path("unused")


def _write_cache(directory: Path, *, calibration: int, evaluation: int, vocab: int):
    generator = torch.Generator().manual_seed(3)
    request_sha256 = "synthetic"
    cache = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "request_sha256": request_sha256,
        "metadata": {"checkpoint_sha256": "deadbeef"},
        "parity_gate": {"passed": True, "agreement": 1.0},
        "eot_id": vocab,
        "numeric_answer_token_ids": list(range(min(vocab, 64))),
        "student_mean": torch.zeros(13, HIDDEN),
        "calibration_states": torch.randn(
            calibration, len(MARGIN_GEOMETRY_STATES), HIDDEN, generator=generator
        ),
        "calibration_gold_first_token": torch.randint(
            0, vocab, (calibration,), generator=generator
        ),
        "evaluation_states": torch.randn(
            evaluation, len(MARGIN_GEOMETRY_STATES), HIDDEN, generator=generator
        ),
        "evaluation_gold_first_token": torch.randint(
            0, vocab, (evaluation,), generator=generator
        ),
        "state_order": list(MARGIN_GEOMETRY_STATES),
    }
    states_path = directory / "colon_states.pt"
    readout_path = directory / "readout.pt"
    torch.save(cache, states_path)
    torch.save(
        {
            "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
            "contract": MARGIN_GEOMETRY_CONTRACT,
            "request_sha256": request_sha256,
            "eot_id": vocab,
            "readout": torch.randn(vocab, HIDDEN, generator=generator) * 0.05,
        },
        readout_path,
    )
    return states_path, readout_path


@pytest.fixture()
def patched_sweep(monkeypatch):
    """Small config and synthetic reference bases, so ``run()`` is fast."""
    monkeypatch.setattr(sweep, "load_config", lambda _path: _Config())

    def _references(_args, _checkpoint_sha256):
        generator = torch.Generator().manual_seed(9)
        references = {}
        for family in ("answer_conditioned", "parameter_aware"):
            basis = torch.linalg.qr(
                torch.randn(HIDDEN, 3, generator=generator), mode="reduced"
            )[0]
            references[family] = MarginSubspace(
                name=f"{family}_k003_s{ANALYTIC_STATE}",
                family=family,
                state=ANALYTIC_STATE,
                basis=basis,
                rank=3,
            )
        return references, {"energy": {}, "answer_conditioned": {}, "parameter_aware": {}}

    monkeypatch.setattr(sweep, "_reference_subspaces", _references)
    return sweep


def test_sweep_runs_end_to_end_and_writes_a_manifest(tmp_path, patched_sweep):
    states, readout = _write_cache(
        tmp_path, calibration=900, evaluation=48, vocab=200
    )
    output = tmp_path / "analytic_sweep"
    payload = patched_sweep.run(_Args(states, readout, output))

    assert payload["contract"] == MARGIN_GEOMETRY_CONTRACT
    assert payload["arms"], "the sweep produced no arms"
    # The request metadata is built after the registry, and previously referenced a
    # variable that the refactor had moved into prepare_registry.
    assert payload["metadata"]["calibration_examples"] == 900
    assert payload["metadata"]["evaluation_examples"] == 48
    baseline = payload["baseline"]
    assert baseline["nll"].shape == (48,)
    assert torch.isfinite(baseline["nll"]).all()
    for arm in payload["arms"].values():
        assert arm["nll"].shape == (48,)
        assert torch.isfinite(arm["nll"]).all()
    assert (output / "analytic_sweep.pt").is_file()
    assert (output / "run_manifest.json").is_file()


def test_sweep_resumes_without_recomputing(tmp_path, patched_sweep):
    states, readout = _write_cache(
        tmp_path, calibration=900, evaluation=32, vocab=128
    )
    output = tmp_path / "analytic_sweep"
    args = _Args(states, readout, output)
    first = patched_sweep.run(args)
    second = patched_sweep.run(args)
    assert first["request_sha256"] == second["request_sha256"]
    assert set(first["arms"]) == set(second["arms"])


def test_sweep_refuses_a_failed_parity_gate(tmp_path, patched_sweep):
    states, readout = _write_cache(
        tmp_path, calibration=900, evaluation=32, vocab=128
    )
    cache = torch.load(states, map_location="cpu", weights_only=False)
    cache["parity_gate"]["passed"] = False
    torch.save(cache, states)
    with pytest.raises(RuntimeError, match="parity gate did not pass"):
        patched_sweep.run(_Args(states, readout, tmp_path / "out"))


def test_sweep_refuses_a_rank_deficient_calibration(tmp_path, patched_sweep):
    """Fewer calibration rows than dimensions makes the shaped sampler unstable."""
    states, readout = _write_cache(
        tmp_path, calibration=256, evaluation=32, vocab=128
    )
    with pytest.raises(RuntimeError, match="full-rank covariance"):
        patched_sweep.run(_Args(states, readout, tmp_path / "out"))


def test_repository_has_no_undefined_names():
    """Catch names left behind by a refactor before a GPU session does.

    ``pyflakes`` resolves undefined names without importing anything, so this
    covers script bodies that unit tests never execute.
    """
    pyflakes = pytest.importorskip("pyflakes")
    result = subprocess.run(
        [sys.executable, "-m", "pyflakes", "src", "scripts", "tests"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    undefined = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if "undefined name" in line
    ]
    assert not undefined, "undefined names:\n" + "\n".join(undefined)


def test_experiment_scripts_parse():
    for name in (
        "collect_official_codi_endpoint_margin_states.py",
        "run_official_codi_endpoint_margin_sweep.py",
        "run_official_codi_endpoint_margin_generation.py",
        "analyze_official_codi_endpoint_margin_geometry.py",
    ):
        path = REPO_ROOT / "scripts" / name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
