import json
import zipfile
from copy import deepcopy

import pytest

from scripts.colab_runner import (
    pack_extracted_checkpoint,
    portable_manifest_mismatches,
    validate_torch_checkpoint_archive,
)


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
    target = tmp_path / "rebuilt" / "step_00000001.pt"
    pack_extracted_checkpoint(extracted, target)
    validate_torch_checkpoint_archive(target)
    with zipfile.ZipFile(target) as archive:
        assert "step_00000001.pt/data.pkl" in archive.namelist()
        assert "step_00000001.pt/data/0" in archive.namelist()


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
