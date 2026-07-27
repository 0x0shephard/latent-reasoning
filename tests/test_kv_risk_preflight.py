from pathlib import Path

import torch
import yaml

from scripts.run_kv_risk_pilot import resolve_dtype
from src.eval.kv_risk_pilot import assert_finite_logits, predictive_entropy


ROOT = Path(__file__).resolve().parents[1]


def test_config_uses_safe_automatic_precision():
    config = yaml.safe_load((ROOT / "configs/kv_risk_pilot.yaml").read_text())
    assert config["model"]["precision"] == "auto"


def test_cpu_auto_precision_is_float32():
    assert resolve_dtype("auto", torch.device("cpu")) == torch.float32
    assert resolve_dtype("float16", torch.device("cpu")) == torch.float32


def test_t4_auto_precision_is_float32_even_if_torch_reports_bf16(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (7, 5),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_dtype("auto", torch.device("cuda")) == torch.float32


def test_ampere_auto_precision_uses_bfloat16_only_when_supported(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 0),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_dtype("auto", torch.device("cuda")) == torch.float32
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_dtype("auto", torch.device("cuda")) == torch.bfloat16


def test_nonfinite_logits_fail_before_entropy_or_sampling():
    logits = torch.tensor([[0.0, float("nan")]])
    try:
        assert_finite_logits(logits, context="unit test")
    except FloatingPointError as error:
        assert "non-finite logits" in str(error)
    else:
        raise AssertionError("non-finite logits were not rejected")

    try:
        predictive_entropy(logits)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("non-finite entropy input was not rejected")
