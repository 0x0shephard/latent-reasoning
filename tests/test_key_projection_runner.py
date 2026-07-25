"""Stage 1d experiment arms preserve the warm-started comparison contract."""
from __future__ import annotations

from pathlib import Path

from scripts.colab_key_projection_runner import (
    EXPERIMENTS,
    build_experiment_config,
)


def test_key_projection_arms_change_only_declared_target_fields(tmp_path):
    warm = tmp_path / "warm.pt"
    projection = tmp_path / "projection.pt"
    configs = {
        name: build_experiment_config(
            name,
            output_dir=tmp_path / name,
            warm_start=warm,
            projection=projection,
            max_seconds=100,
            keep_last=2,
        )
        for name in EXPERIMENTS
    }
    reference = configs["codi_continue_seed1"].to_dict()
    allowed = {
        "run_name",
        "output_dir",
        "task.method",
        "task.distillation.kv_weight",
        "task.distillation.kv_target",
        "task.distillation.key_projection_path",
        "task.distillation.key_projection_kind",
        "task.distillation.key_projection_rank",
    }

    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                result.update(flatten(item, path))
            else:
                result[path] = item
        return result

    left = flatten(reference)
    for name, cfg in configs.items():
        right = flatten(cfg.to_dict())
        differences = {
            key
            for key in left.keys() | right.keys()
            if left.get(key) != right.get(key)
        }
        assert differences <= allowed, (name, differences)
        assert cfg.seed == 1
        assert cfg.train.total_steps == 10000
        assert Path(cfg.task.warm_start_checkpoint) == warm
