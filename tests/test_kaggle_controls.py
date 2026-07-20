from __future__ import annotations

import json
import zipfile

import pytest

import scripts.import_kaggle_control as importer
from scripts.import_kaggle_control import (
    _resolve_experiment_source,
    _validate_identity,
    import_experiment,
)


def _manifest(method="kava_random", seed=0, run_name="kava_random_seed0", total=1):
    return {
        "fingerprint": "a" * 64,
        "resume_config": {
            "seed": seed,
            "run_name": run_name,
            "task": {"method": method},
        },
        "effective_config": {"train": {"total_steps": total}},
    }


def _experiment_tree(tmp_path):
    root = (
        tmp_path
        / "notebook-output"
        / "latent-reasoning"
        / "outputs"
        / "controls_and_seeds"
        / "kava_random_seed0"
    )
    root.mkdir(parents=True)
    (root / "run_manifest.json").write_text(json.dumps(_manifest()))
    checkpoint = root / "checkpoints" / "step_00000001.pt"
    checkpoint.parent.mkdir()
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("step_00000001.pt/data.pkl", b"payload")
        archive.writestr("step_00000001.pt/byteorder", b"little")
        archive.writestr("step_00000001.pt/version", b"3")
    summary = root / "eval" / "step_00000001" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}\n")
    return root


def test_kaggle_import_resolves_and_checks_experiment_identity(tmp_path):
    experiment = _experiment_tree(tmp_path)
    assert _resolve_experiment_source(tmp_path / "notebook-output", experiment.name) == experiment
    assert _validate_identity(experiment, experiment.name)["fingerprint"] == "a" * 64

    manifest = _manifest(method="codi")
    (experiment / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="method mismatch"):
        _validate_identity(experiment, experiment.name)


def test_kaggle_import_verifies_then_installs_into_empty_drive_target(
    tmp_path, monkeypatch
):
    source = _experiment_tree(tmp_path)
    monkeypatch.setattr(importer, "validate_checkpoint_payload", lambda path, step: None)
    drive = tmp_path / "drive"
    drive.mkdir()

    status = import_experiment(
        source.parent.parent.parent.parent.parent,
        drive,
        "kava_random_seed0",
    )

    target = (
        drive
        / "outputs"
        / "controls_and_seeds"
        / "kava_random_seed0"
    )
    assert status["state"] == "complete"
    assert (target / "checkpoints" / "step_00000001.pt").is_file()
    assert json.loads(
        (drive / "status" / "controls_and_seeds" / "kava_random_seed0.json").read_text()
    )["platform"] == "kaggle_import"

    with pytest.raises(FileExistsError, match="refusing to merge"):
        import_experiment(source, drive, "kava_random_seed0")


def test_kaggle_control_notebook_is_locked_to_allocated_experiments_and_compiles():
    path = "notebooks/kaggle_controls_and_seeds.ipynb"
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    all_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'EXPERIMENT = "kava_random_seed0"' in all_code
    assert '"codi_seed1"' in all_code
    assert '"kava_seed1"' in all_code
    assert '"codi_seed2"' not in all_code
    assert "scripts/kaggle_control_runner.py" in all_code
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:cell_{index}", "exec")


def test_codi_seed2_eval_only_notebook_is_pinned_and_never_trains():
    path = "notebooks/kaggle_codi_seed2_eval_only.ipynb"
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    all_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'EXPERIMENT = "codi_seed2"' in all_code
    assert 'EXPECTED_METHOD = "codi"' in all_code
    assert "EXPECTED_SEED = 2" in all_code
    assert "EXPECTED_STEP = 96405" in all_code
    assert 'RUN_COMMIT = "d917bef2cf396fe3b0453e6f86648f1a3948f528"' in all_code
    assert 'FINAL_DATASET_HANDLE = "jonraza15/codi-seed2-final-step96405"' in all_code
    assert "evaluate(cfg, limit=EVAL_LIMIT)" in all_code
    assert "kaggle_control_runner.py" not in all_code
    assert "src.train" not in all_code
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:cell_{index}", "exec")
