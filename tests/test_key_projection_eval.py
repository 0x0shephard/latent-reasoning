"""Evaluation-only Stage 1d config reconstruction tests."""
from __future__ import annotations

import json

from scripts.eval_key_projection import config_from_manifest


def test_config_from_manifest_preserves_method_and_retargets_output(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "effective_config": {
                    "output_dir": "/old/location",
                    "data_config": "configs/data.yaml",
                    "task": {
                        "type": "latent",
                        "method": "kava_key_rank4",
                    },
                    "eval": {"batch_size": 2, "limit": 200},
                }
            }
        )
    )
    cfg = config_from_manifest(output, batch_size=8)
    assert cfg.output_dir == str(output)
    assert cfg.task.method == "kava_key_rank4"
    assert cfg.eval.batch_size == 8
    assert cfg.eval.limit == 200
    assert cfg.data_config.endswith("/configs/data.yaml")
