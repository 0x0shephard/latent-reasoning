import json
import os
import threading
import time
import zipfile
from copy import deepcopy

import pytest

import scripts.colab_runner as colab_runner
from scripts.colab_runner import (
    DriveMirror,
    bootstrap_drive,
    pack_extracted_checkpoint,
    portable_manifest_mismatches,
    validate_torch_checkpoint_archive,
)
from scripts.colab_ablation_runner import ablation_eval_tag


def _manifest():
    return {
        "source_sha256": "source",
        "data_config": {"train": {"revision": "pinned"}},
        "environment": {"python": "3.12", "packages": {"torch": "2.10"}},
        "resume_config": {
            "seed": 0,
            "task": {
                "method": "codi",
                "train_dataset_fingerprint": "kaggle-cache-id",
                "train_examples": 385620,
            },
            "train": {"lr": 1e-4, "batch_size": 4},
        },
    }


def test_portable_resume_allows_only_environment_derived_differences():
    stored = _manifest()
    current = deepcopy(stored)
    current["environment"] = {"python": "3.13", "packages": {"torch": "2.11"}}
    current["resume_config"]["task"]["train_dataset_fingerprint"] = "colab-cache-id"
    assert portable_manifest_mismatches(stored, current) == []


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda value: value.update(source_sha256="changed"), "executable source hash"),
        (
            lambda value: value["data_config"]["train"].update(revision="different"),
            "data config",
        ),
        (
            lambda value: value["resume_config"]["train"].update(lr=2e-4),
            "scientific/resume config",
        ),
    ],
)
def test_portable_resume_rejects_scientific_changes(mutation, expected):
    stored = _manifest()
    current = deepcopy(stored)
    mutation(current)
    assert expected in portable_manifest_mismatches(stored, current)


def test_checkpoint_archive_validation(tmp_path):
    checkpoint = tmp_path / "step_00000001.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("step_00000001.pt/data.pkl", b"payload")
        archive.writestr("step_00000001.pt/byteorder", b"little")
        archive.writestr("step_00000001.pt/version", b"3")
    validate_torch_checkpoint_archive(checkpoint)

    ordinary_zip = tmp_path / "ordinary.zip"
    with zipfile.ZipFile(ordinary_zip, "w") as archive:
        archive.writestr("notes.txt", "not a checkpoint")
    with pytest.raises(ValueError, match="missing"):
        validate_torch_checkpoint_archive(ordinary_zip)


def test_pack_extracted_checkpoint_folder(tmp_path):
    extracted = tmp_path / "step_00000001.pt"
    (extracted / "data").mkdir(parents=True)
    (extracted / "data.pkl").write_bytes(b"payload")
    (extracted / "byteorder").write_text("little")
    (extracted / "version").write_text("3")
    (extracted / "data" / "0").write_bytes(b"storage")
    # PyTorch archives commonly preserve the ZIP epoch as a timestamp. Drive folder
    # uploads can surface it as pre-1980 filesystem metadata, which ZipInfo rejects unless
    # the writer explicitly clamps legacy timestamps.
    os.utime(extracted / "data.pkl", (0, 0))
    target = tmp_path / "rebuilt" / "step_00000001.pt"
    pack_extracted_checkpoint(extracted, target)
    validate_torch_checkpoint_archive(target)
    with zipfile.ZipFile(target) as archive:
        assert "step_00000001.pt/data.pkl" in archive.namelist()
        assert "step_00000001.pt/data/0" in archive.namelist()


def test_drive_mirror_serializes_concurrent_syncs(tmp_path):
    mirror = DriveMirror(
        tmp_path / "local",
        tmp_path / "drive",
        tmp_path / "status.json",
        "codi",
        poll_seconds=0.01,
    )
    counters = {"active": 0, "maximum": 0}
    counter_lock = threading.Lock()

    def fake_sync_once(force=False):
        with counter_lock:
            counters["active"] += 1
            counters["maximum"] = max(counters["maximum"], counters["active"])
        time.sleep(0.03)
        with counter_lock:
            counters["active"] -= 1

    mirror._sync_once = fake_sync_once
    workers = [threading.Thread(target=mirror.sync, kwargs={"force": True}) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert counters["maximum"] == 1


def test_bootstrap_uses_newest_durable_checkpoint_without_original_upload(
    tmp_path, monkeypatch
):
    drive_root = tmp_path / "drive"
    checkpoint = drive_root / "outputs" / "codi" / "checkpoints" / "step_00096000.pt"
    checkpoint.parent.mkdir(parents=True)
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("step_00096000.pt/data.pkl", b"payload")
        archive.writestr("step_00096000.pt/byteorder", b"little")
        archive.writestr("step_00096000.pt/version", b"3")
    verified = []
    monkeypatch.setattr(
        colab_runner,
        "validate_checkpoint_payload",
        lambda path, expected_step: verified.append((path.name, expected_step)),
    )

    output = bootstrap_drive("codi", drive_root, tmp_path / "scratch")

    assert output == drive_root / "outputs" / "codi"
    assert verified == [("step_00096000.pt", 96000)]
    assert not (drive_root / "uploads").exists()


def test_archived_resume_manifests_are_valid_json():
    for name in ("codi", "kava"):
        path = f"artifacts/phase2_resume/{name}_run_manifest.json"
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        assert manifest["effective_config"]["task"]["method"] == name
        assert len(manifest["fingerprint"]) == 64


@pytest.mark.parametrize(
    "method, step",
    [("codi", 80000), ("kava", 24000)],
)
def test_colab_notebooks_are_locked_to_one_method(method, step):
    path = f"notebooks/colab_phase2_{method}.ipynb"
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    settings = "".join(notebook["cells"][1]["source"])
    preflight = "".join(notebook["cells"][5]["source"])
    assert f'METHOD = "{method}"' in settings
    assert f'assert METHOD == "{method}"' in settings
    assert f"expected_step = {step}" in preflight
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:cell_{index}", "exec")


def test_phase3_notebook_is_post_training_only_and_compiles():
    path = "notebooks/colab_phase3_ablations.ipynb"
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    settings = "".join(notebook["cells"][1]["source"])
    all_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "RUN_KAVA_ABLATIONS = True" in settings
    assert "RUN_CODI_ABLATIONS = False" in settings
    assert "RUN_KAVA_POSITION_SWEEP = False" in settings
    assert "RUN_MATCHED_BATCH_DID = False" in settings
    assert "RUN_FULL_BASELINES = False" in settings
    assert "scripts/analyze_phase2.py" in all_code
    assert "scripts/analyze_position_sweep.py" in all_code
    assert "scripts/analyze_intervention_effects.py" in all_code
    assert "scripts/colab_ablation_runner.py" in all_code
    assert '"scripts/colab_runner.py"' not in all_code
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:cell_{index}", "exec")


def test_ablation_tags_preserve_matched_batch_conditions():
    assert ablation_eval_tag("baseline", "all") is None
    assert ablation_eval_tag("baseline", "all", 8) == "baseline_bs8"
    assert ablation_eval_tag("batch_shuffle", "p3") == "batch_shuffle_p3"
    assert ablation_eval_tag("batch_shuffle", "p3", 8) == "batch_shuffle_p3_bs8"
