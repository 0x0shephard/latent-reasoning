import argparse
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.run_latent_monitor_audit import (  # noqa: E402
    STAGES,
    assemble,
    layout,
    locate_existing,
    masking_command,
    plan,
    twins_command,
)
from tests.test_monitor_audit import _masking, _monitor, _twins  # noqa: E402


def test_plan_drops_the_long_profile_for_one_step_pools():
    assert plan("gsm8k_aug_train") == list(STAGES) and plan("gsm8k_test") == list(STAGES)
    assert "twins_long" not in plan("svamp") and plan("svamp", "on") == list(STAGES) and "twins_long" not in plan("gsm8k_test", "off")
    assert plan("gsm8k_test", stages=("twins_standard", "card")) == ["twins_standard", "card"]


def test_commands_carry_pool_profile_and_carriers():
    args = argparse.Namespace(config=Path("configs/official_codi_gpt2.yaml"), reproduction_summary=Path("r.json"), batch_size=32,
                              precision="float32", device="cuda", pool="gsm8k_test", checkpoint_path=None)
    cmd = twins_command(args, "long", Path("out/twins_long"), smoke=True)
    assert cmd[-6:] == ["--profile", "long", "--pool", "gsm8k_test", "--smoke"][-5:] or ("--pool" in cmd and "gsm8k_test" in cmd and "--smoke" in cmd)
    assert "run_codi_directed_counterfactual.py" in cmd[2]
    cmd = masking_command(args, Path("out/masking"), [1, 3], smoke=False)
    assert cmd[cmd.index("--carrying-slots") + 1:cmd.index("--carrying-slots") + 3] == ["1", "3"] and "--smoke" not in cmd
    cmd = masking_command(args, Path("out/masking"), [], smoke=False)
    assert cmd[-1] == "--carrying-slots"


def _kaggle_layout(root: Path, prefix: str):
    std, long, mask = root / f"{prefix}_twins_standard", root / f"{prefix}_twins_long", root / f"{prefix}_masking"
    for d in (std, long, mask, root / f"{prefix}_twins_long_smoke"):
        d.mkdir(parents=True)
    (std / "summary.json").write_text(json.dumps(_twins({1: 0.05, 3: 0.10}, {0: 0.23})))
    (long / "summary.json").write_text(json.dumps(_twins({1: 0.153, 3: 0.325}, {4: 0.3})))
    (mask / "summary.json").write_text(json.dumps(_masking({1: 0.122, 3: 0.125})))
    for d in (std, long):
        torch.save({"state_3": torch.tensor([1, 2, 3, 4]), "own_gold": torch.tensor([1, 2, 3, 4]), "target_gold": torch.tensor([2, 1, 4, 3]), "located": [[], [3], [], []]},
                   d / "predictions.pt")
        (d / "pairs.jsonl").write_text('{"question": "q"}\n')
    (root / f"{prefix}_bootstrap.json").write_text(json.dumps({"standard": {"target_rate": {}}, "long": {"target_rate": {"state_3": {"mean": 0.325, "ci95": [0.29, 0.36], "pairs": 512}}}}))
    (root / f"{prefix}_monitor_faithfulness.json").write_text(json.dumps([_monitor({}, {}, run=str(std)), _monitor({3: 0.4, 4: 0.5}, {3: 0.4}, run=str(long))]))


def test_locate_existing_recognises_the_kaggle_layout(tmp_path):
    _kaggle_layout(tmp_path, "gpt2")
    lay = locate_existing(tmp_path, "gpt2")
    assert lay["twins_standard"].name == "gpt2_twins_standard" and lay["twins_long"].name == "gpt2_twins_long"
    assert lay["masking"].name == "gpt2_masking" and lay["bootstrap"].name == "gpt2_bootstrap.json" and lay["monitor"].name == "gpt2_monitor_faithfulness.json"
    lay = locate_existing(tmp_path)   # without the prefix the glob finds them, never the smoke run
    assert lay["twins_long"].name == "gpt2_twins_long"
    assert layout(tmp_path)["card"] == tmp_path / "card.json"


def test_assemble_from_existing_runs_without_a_model(tmp_path):
    _kaggle_layout(tmp_path, "gpt2")
    lay = locate_existing(tmp_path, "gpt2")
    args = argparse.Namespace(config=Path("configs/official_codi_gpt2.yaml"), pool="gsm8k_aug_train", reproduction_summary=None, device="cpu",
                              precision="float32", batch_size=32, stages=list(STAGES), from_existing=tmp_path, model_name="gpt2", device_name="Tesla T4", smoke=False)
    from src.utils.config import load_config
    card = assemble(args, lay, stage_seconds={"twins_standard": 400.0}, cfg=load_config("configs/official_codi_gpt2.yaml"))
    assert card["model"]["name"] == "gpt2" and card["model"]["layers"] == 12 and card["gate"] is None
    assert card["verdict"]["carrying_slots"] == [1, 3] and card["slots"]["3"]["transport_long_ci95"] == [0.29, 0.36]
    assert card["slots"]["4"]["readability_long"] == 0.5 and card["environment"]["device_name"] == "Tesla T4" and card["environment"]["from_existing"]
    assert card["environment"]["torch"] is None and card["environment"]["environment_source"] == "unknown (pre-card run)"
    (tmp_path / "card.json").write_text(json.dumps({"environment": {"device_name": "Tesla P100", "torch": "2.11.0", "dtype": "float32"}, "gate": {"accuracy": 0.436, "status": "passed"}}))
    args.device_name = None
    card = assemble(args, lay, stage_seconds={"twins_standard": 400.0}, cfg=load_config("configs/official_codi_gpt2.yaml"))
    assert card["environment"]["device_name"] == "Tesla P100" and card["environment"]["torch"] == "2.11.0" and card["gate"]["accuracy"] == 0.436
    assert set(card["fingerprints"]) == {"twins_standard", "twins_long"} and len(card["fingerprints"]["twins_long"]["predictions"]) == 64
    assert card["fingerprints"]["twins_long"]["predictions"] == card["fingerprints"]["twins_standard"]["predictions"]
    assert card["pool"]["pool"] == "gsm8k_aug_train" and card["stage_seconds"]["twins_standard"] == 400.0
