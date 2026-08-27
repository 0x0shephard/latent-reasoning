"""End-to-end guards for the correctness-track entry points.

Every Kaggle failure in the previous experiment was reachable only by running the
scripts rather than the library, so both CLIs are driven here over a synthetic
colon-state cache, all the way from ``main()`` to the gate report.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.analyze_official_codi_correctness_tracks as analyze
import scripts.run_official_codi_correctness_tracks as tracks
from src.mech.endpoint_correctness_geometry import READOUT_KEY, readout_matrix
from src.mech.endpoint_margin_geometry import (
    MARGIN_GEOMETRY_CONTRACT,
    MARGIN_GEOMETRY_SCHEMA_VERSION,
    MARGIN_GEOMETRY_STATES,
)

HIDDEN = 768
VOCABULARY = 96


class _Settings:
    """A miniature of the shipped config; the same field names the runner reads."""

    fit_examples = 96
    split_seed = 3
    random_seed = 3
    accuracy_band = [4, 32]
    fisher_shrinkage_grid = [0.05, 0.5]
    null_replicates = 4
    ridge_grid = [0.01, 1.0]
    probe_steps = 40
    detect_primary_probe = "fisher_plus_margin"
    minimum_detect_delta_auc = 0.01
    alpha_grid = [0.0, 0.5, 2.0]
    steer_random_replicates = 2
    steer_primary_arm = "margin_band"
    minimum_steer_gain_points = 1.0
    rank_grid = [4, 28]
    project_primary_rank = 28
    minimum_project_advantage_points = 1.0
    bootstrap_samples = 100
    bootstrap_seed = 0
    alpha = 0.05


class _Config:
    """Exactly the shape ``load_config`` returns -- no extra aliases.

    An earlier version also exposed ``experiments.endpoint_correctness_tracks``,
    which let the runner read a path the shipped config does not have and hid the
    failure until a full-scale run.
    """

    endpoint_correctness_tracks = _Settings()


@pytest.fixture
def cache_paths(tmp_path, monkeypatch):
    """Write a synthetic margin-geometry cache the tracks runner can consume."""
    generator = torch.Generator().manual_seed(41)
    readout = torch.randn(VOCABULARY, HIDDEN, generator=generator)
    calibration = torch.randn(192, len(MARGIN_GEOMETRY_STATES), HIDDEN, generator=generator)
    evaluation = torch.randn(160, len(MARGIN_GEOMETRY_STATES), HIDDEN, generator=generator)
    index = list(MARGIN_GEOMETRY_STATES).index(12)
    calibration[:, index, :4] *= 9.0
    evaluation[:, index, :4] *= 9.0

    def golds(states):
        chosen = (states[:, index, :] @ readout.T).argmax(dim=-1)
        # Corrupt every third label so both classes are populated on both splits.
        flip = torch.arange(states.shape[0]) % 3 == 0
        chosen[flip] = (chosen[flip] + 1) % VOCABULARY
        return chosen

    cache = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "request_sha256": "abc123",
        "parity_gate": {"passed": True, "agreement": 1.0},
        "state_order": list(MARGIN_GEOMETRY_STATES),
        "calibration_states": calibration,
        "calibration_gold_first_token": golds(calibration),
        "evaluation_states": evaluation,
        "evaluation_gold_first_token": golds(evaluation),
        "student_mean": torch.zeros(13, HIDDEN),
    }
    states_path = tmp_path / "colon_states.pt"
    readout_path = tmp_path / "readout.pt"
    torch.save(cache, states_path)
    torch.save(
        {"request_sha256": "abc123", READOUT_KEY: readout}, readout_path
    )
    monkeypatch.setattr(tracks, "load_config", lambda _path: _Config())
    monkeypatch.setattr(analyze, "load_config", lambda _path: _Config())
    return states_path, readout_path


def _run(tmp_path, cache_paths):
    states_path, readout_path = cache_paths
    sweep_path = tmp_path / "tracks.json"
    assert (
        tracks.main(
            [
                "--states", str(states_path),
                "--readout", str(readout_path),
                "--output", str(sweep_path),
                "--device", "cpu",
            ]
        )
        == 0
    )
    return sweep_path


def test_runner_produces_every_track(tmp_path, cache_paths):
    sweep_path = _run(tmp_path, cache_paths)
    payload = json.loads(sweep_path.read_text())
    assert payload["splits"] == {
        "fit": 96,
        "select": 96,
        "test": 160,
        "split_seed": 3,
        "correct_share": payload["splits"]["correct_share"],
    }
    assert set(payload["detect"]) >= {
        "margin", "fisher", "fisher_plus_margin", "accuracy_band", "full_state"
    }
    assert "margin_band" in payload["steer"] and "random_band_r00" in payload["steer"]
    assert set(payload["project"]) >= {"4", "28", "accuracy_band"}
    # Tensors must not leak into the JSON summary.
    assert "outcomes" not in payload["steer"]["margin_band"]
    assert "vector" not in payload["steer"]["margin_band"]


def test_runner_writes_paired_outcomes_for_every_gate(tmp_path, cache_paths):
    sweep_path = _run(tmp_path, cache_paths)
    payload = torch.load(sweep_path.with_suffix(".pt"), weights_only=False)
    outcomes = payload["outcomes"]
    assert outcomes["labels"].shape == outcomes["baseline"].shape == (160,)
    assert outcomes["steer"]["margin_band"].shape == (160,)
    assert outcomes["detect"]["fisher_plus_margin"].shape == (160,)
    assert set(outcomes["project"]["28"]) == {
        "class_blind", "correct_only", "incorrect_only"
    }
    for name, vector in payload["steering_vectors"].items():
        assert float(vector.double().norm()) == pytest.approx(1.0, abs=1e-5), name


def test_steering_vectors_respect_their_band(tmp_path, cache_paths):
    """A band-restricted vector must have no weight outside the band."""
    sweep_path = _run(tmp_path, cache_paths)
    payload = torch.load(sweep_path.with_suffix(".pt"), weights_only=False)
    vectors = payload["steering_vectors"]
    eigenvectors = payload["eigenvectors"].double()
    for name in ("margin_band", "fisher_band", "random_band_r00"):
        coefficients = eigenvectors.T @ vectors[name].double()
        inside = float((coefficients[4:32] ** 2).sum())
        assert inside == pytest.approx(1.0, abs=1e-4), name
    outside = eigenvectors.T @ vectors["mean_difference_global"].double()
    assert float((outside[4:32] ** 2).sum()) < 0.999


def test_alpha_zero_reproduces_the_baseline_exactly(tmp_path, cache_paths):
    """The steering grid contains a genuine no-op, so a null arm cannot drift."""
    sweep_path = _run(tmp_path, cache_paths)
    payload = json.loads(sweep_path.read_text())
    baseline = payload["baseline_first_token_accuracy"]
    for name, entry in payload["steer"].items():
        assert entry["select_curve"]["0"] <= entry["select_accuracy"], name
        if entry["selected_alpha"] == 0.0:
            assert entry["test_accuracy"] == pytest.approx(baseline), name


def test_analysis_cli_reads_the_runner_output(tmp_path, cache_paths):
    sweep_path = _run(tmp_path, cache_paths)
    report_path = tmp_path / "report.json"
    assert (
        analyze.main(["--sweep", str(sweep_path), "--output", str(report_path)]) == 0
    )
    report = json.loads(report_path.read_text())
    assert set(report["tracks_passed"]) == {"detect", "steer", "project"}
    assert isinstance(report["steer"]["passed"], bool)
    assert report["detect"]["margin_auc"] > 0.0


def test_analysis_cli_rejects_a_foreign_contract(tmp_path, cache_paths):
    sweep_path = _run(tmp_path, cache_paths)
    payload = json.loads(sweep_path.read_text())
    payload["contract"] = "something_else"
    sweep_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="another contract"):
        analyze.main(["--sweep", str(sweep_path), "--output", str(tmp_path / "r.json")])


def test_runner_refuses_a_failed_parity_gate(tmp_path, cache_paths):
    states_path, readout_path = cache_paths
    cache = torch.load(states_path, weights_only=False)
    cache["parity_gate"]["passed"] = False
    torch.save(cache, states_path)
    with pytest.raises(RuntimeError, match="parity gate"):
        tracks.main(
            ["--states", str(states_path), "--readout", str(readout_path),
             "--output", str(tmp_path / "x.json"), "--device", "cpu"]
        )


def test_runner_refuses_a_one_sided_split(tmp_path, cache_paths, monkeypatch):
    """Fitting a correctness direction needs both classes on every split."""
    states_path, readout_path = cache_paths
    cache = torch.load(states_path, weights_only=False)
    index = list(cache["state_order"]).index(12)
    readout = torch.load(readout_path, weights_only=False)[READOUT_KEY]
    cache["calibration_gold_first_token"] = (
        cache["calibration_states"][:, index, :] @ readout.T
    ).argmax(dim=-1)
    torch.save(cache, states_path)
    with pytest.raises(RuntimeError, match="too one-sided"):
        tracks.main(
            ["--states", str(states_path), "--readout", str(readout_path),
             "--output", str(tmp_path / "x.json"), "--device", "cpu"]
        )


def test_split_calibration_is_a_disjoint_deterministic_partition():
    fit, select = tracks.split_calibration(2048, seed=20260812, fit_size=1024)
    again, _ = tracks.split_calibration(2048, seed=20260812, fit_size=1024)
    assert torch.equal(fit, again)
    assert fit.shape[0] == select.shape[0] == 1024
    assert set(fit.tolist()).isdisjoint(select.tolist())
    assert sorted(fit.tolist() + select.tolist()) == list(range(2048))
    for bad in (0, 2048, 3000):
        with pytest.raises(ValueError):
            tracks.split_calibration(2048, seed=1, fit_size=bad)


def test_experiment_scripts_parse():
    for name in (
        "run_official_codi_correctness_tracks.py",
        "analyze_official_codi_correctness_tracks.py",
        "run_official_codi_correctness_steer_generation.py",
        "run_official_codi_correctness_detect_replication.py",
        "analyze_official_codi_correctness_detect_replication.py",
    ):
        path = REPO_ROOT / "scripts" / name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_correctness_scripts_have_no_unused_or_undefined_names():
    """Unused imports are how the last refactor's dead paths hid; reject both."""
    pytest.importorskip("pyflakes")
    targets = [
        "src/mech/endpoint_correctness_geometry.py",
        "src/eval/official_codi_correctness_tracks_analysis.py",
        "src/eval/official_codi_correctness_detect_replication_analysis.py",
        "scripts/run_official_codi_correctness_tracks.py",
        "scripts/analyze_official_codi_correctness_tracks.py",
        "scripts/run_official_codi_correctness_steer_generation.py",
        "scripts/run_official_codi_correctness_detect_replication.py",
        "scripts/analyze_official_codi_correctness_detect_replication.py",
        "tests/test_endpoint_correctness_geometry.py",
        "tests/test_correctness_tracks_integration.py",
        "tests/test_correctness_detect_replication.py",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "pyflakes", *targets],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    complaints = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if "undefined name" in line or "imported but unused" in line
    ]
    assert not complaints, "pyflakes:\n" + "\n".join(complaints)


# --------------------------------------------------------------------------
# the shipped notebook
# --------------------------------------------------------------------------

NOTEBOOK = (
    REPO_ROOT / "notebooks" / "kaggle_official_codi_correctness_tracks.ipynb"
)


def test_notebook_code_cells_parse_and_keep_their_guards():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for value in (
        # The environment that reproduces the checkpoint, and the guards that
        # detect it drifting -- the cause of the previous run's silent 0.3723.
        '"transformers": "4.52.4"',
        "_peft_torchao_state",
        'assert cache["parity_gate"]["passed"]',
        'cache["metadata"]["precision"] == "float32"',
        'baseline_drift_passed") is True',
        # Both tiers, and the tests that gate them.
        "tests/test_correctness_tracks_integration.py",
        "run_official_codi_correctness_tracks.py",
        "analyze_official_codi_correctness_tracks.py",
        "run_official_codi_correctness_steer_generation.py",
        'GENERATION_PRECISION = "float32"',
        "SHA256SUMS.txt",
    ):
        assert value in source, value


def test_notebook_never_retunes_the_steering_step_at_the_generation_tier():
    """Alpha comes from the analytic export, which chose it on the select split."""
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert '"--alpha"' not in source
    assert '"--vectors", str(VECTORS)' in source


def test_notebook_reports_all_three_tracks():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for track in ("detect", "steer", "project"):
        assert f'report["{track}"]' in source, track
    assert 'report["tracks_passed"]' in source
    # The random-in-band control is what separates a real steering effect from a
    # perturbation, so it must be printed rather than only gated on.
    assert 'report["steer"]["random_controls"]' in source or (
        's["random_controls"]' in source
    )


def test_readout_key_matches_what_the_collector_actually_writes():
    """Producer/consumer contract on the readout export.

    A synthetic fixture proves nothing if it invents the same wrong key the
    reader uses -- which is exactly how a KeyError on 'output_embedding' reached
    a Kaggle run despite a green integration test. So read the key out of the
    collector's source rather than trusting either side.
    """
    source = (
        REPO_ROOT / "scripts" / "collect_official_codi_endpoint_margin_states.py"
    ).read_text(encoding="utf-8")
    written = re.findall(r'"(\w+)": readout,', source)
    assert written, "the collector no longer writes a bare readout entry"
    assert READOUT_KEY in written, (READOUT_KEY, written)
    # And the shared reader is the only path the runner uses to reach it.
    runner = (
        REPO_ROOT / "scripts" / "run_official_codi_correctness_tracks.py"
    ).read_text(encoding="utf-8")
    assert "readout_matrix(readout_payload)" in runner
    assert "output_embedding" not in runner


def test_readout_matrix_reports_a_missing_or_misshaped_export():
    with pytest.raises(KeyError, match="readout"):
        readout_matrix({"request_sha256": "x"})
    with pytest.raises(ValueError, match=r"\[V, 768\]"):
        readout_matrix({READOUT_KEY: torch.zeros(50257, 512)})


def test_notebook_uses_the_shared_readout_reader():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "readout_matrix(" in source
    assert "output_embedding" not in source


# --------------------------------------------------------------------------
# device discipline
# --------------------------------------------------------------------------

TENSOR_FACTORIES = {
    "eye", "zeros", "ones", "randn", "rand", "arange", "empty", "full",
    "randperm", "linspace", "eye_like",
}


def _factory_calls(path: Path):
    """Every ``torch.<factory>(...)`` call in a file, with its keyword names."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
            and func.attr in TENSOR_FACTORIES
        ):
            yield func.attr, node.lineno, {kw.arg for kw in node.keywords}


def test_every_tensor_factory_names_its_device():
    """A CPU-created tensor meeting a CUDA one is the whole failure mode.

    The sweep is model-free and was developed on CPU, so nothing in the test
    suite exercises a second device; ``torch.eye`` without ``device=`` ran green
    locally and died on the first GPU call. A static check is what actually
    covers this, since there is no GPU here to run against.
    """
    offenders = []
    for name in (
        "src/mech/endpoint_correctness_geometry.py",
        "scripts/run_official_codi_correctness_tracks.py",
    ):
        path = REPO_ROOT / name
        for factory, line, keywords in _factory_calls(path):
            if "device" not in keywords:
                offenders.append(f"{name}:{line}: torch.{factory}() has no device=")
    assert not offenders, "device-implicit tensor creation:\n" + "\n".join(offenders)


def test_seeded_draws_are_taken_on_the_cpu_so_arms_are_device_independent():
    """A CUDA generator would make ``random_band_r00`` a different direction.

    Control arms are compared against the primary arm by name across tiers, so a
    draw that depends on where the sweep ran would silently break the comparison
    rather than fail.
    """
    source = (
        REPO_ROOT / "src" / "mech" / "endpoint_correctness_geometry.py"
    ).read_text(encoding="utf-8")
    assert 'torch.Generator(device="cpu")' in source
    for factory, _line, keywords in _factory_calls(
        REPO_ROOT / "src" / "mech" / "endpoint_correctness_geometry.py"
    ):
        if "generator" in keywords:
            assert "device" in keywords, factory


def test_random_split_null_moves_its_mask_to_the_state_device():
    """Boolean masks, unlike index tensors, must share the tensor's device."""
    source = (
        REPO_ROOT / "src" / "mech" / "endpoint_correctness_geometry.py"
    ).read_text(encoding="utf-8")
    assert "mask = mask.to(states.device)" in source
