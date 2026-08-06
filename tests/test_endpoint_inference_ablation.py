from __future__ import annotations

import torch
import torch.nn as nn

from src.eval.official_codi_endpoint_inference_ablation_analysis import (
    analyze_endpoint_inference_ablation,
)
from src.mech.endpoint_inference_ablation import (
    EndpointAblationSpec,
    OfficialCODIEndpointHiddenAblation,
    build_endpoint_ablation_specs,
)
from src.mech.endpoint_retention import RetentionBasis
from src.models.official_codi import _new_answer_endpoint_mask


def _retention_basis(name: str, offset: int) -> RetentionBasis:
    basis = torch.zeros(13, 768, 3)
    ranks = torch.zeros(13, dtype=torch.long)
    for state in (11, 12):
        for slot in range(3):
            basis[state, offset + state * 3 + slot, slot] = 1
        ranks[state] = 3
    return RetentionBasis(
        name=name,
        basis=basis,
        ranks=ranks,
        source_path=f"/{name}.pt",
        source_sha256="a",
        source_request_sha256="b",
        source_contract="c",
    )


def test_registered_arms_have_method_single_and_random_controls():
    bases = {
        "energy": _retention_basis("energy", 0),
        "answer_conditioned": _retention_basis("answer_conditioned", 64),
        "parameter_aware": _retention_basis("parameter_aware", 128),
    }
    specs = build_endpoint_ablation_specs(
        bases, random_replicates=20, random_seed=7
    )
    assert len(specs) == 81
    assert "remove_answer_conditioned_joint" in specs
    assert "remove_parameter_aware_s12_d2" in specs
    assert "remove_random_joint_r19" in specs
    assert "remove_random_s11_r19" in specs


def test_endpoint_mask_fires_once_after_exact_cue_suffix():
    applied = torch.tensor([False, False, True])
    mask = _new_answer_endpoint_mask(
        [[5, 7, 8], [7], [5, 7, 8]], [5, 7, 8], applied
    )
    assert mask.tolist() == [True, False, False]


class _Block(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _Base:
    def __init__(self):
        self.transformer = type(
            "Transformer",
            (),
            {
                "h": nn.ModuleList([_Block() for _ in range(12)]),
                "ln_f": nn.Identity(),
            },
        )()


class _Codi:
    def __init__(self):
        self.base = _Base()

    def get_base_model(self):
        return self.base


class _Model:
    def __init__(self):
        self.codi = _Codi()


def test_hidden_ablation_edits_only_masked_rows_and_active_state():
    basis = torch.zeros(13, 768, 3)
    basis[12, 0, 0] = 1
    ranks = torch.zeros(13, dtype=torch.long)
    ranks[12] = 1
    spec = EndpointAblationSpec(
        name="test",
        basis=basis,
        ranks=ranks,
        family="selected_single",
        method="energy",
        state=12,
        direction_slot=0,
        residual_pc_index=0,
        random_replicate=None,
    )
    model = _Model()
    intervention = OfficialCODIEndpointHiddenAblation(
        model, spec, student_mean=torch.zeros(13, 768)
    )
    hidden = torch.zeros(2, 1, 768)
    hidden[:, :, 0] = torch.tensor([[3.0], [4.0]])
    with intervention.activate(torch.tensor([True, False])):
        output = model.codi.base.transformer.ln_f(hidden)
    assert output[0, 0, 0] == 0
    assert output[1, 0, 0] == 4
    assert intervention.diagnostics()["rows_by_state"]["12"] == 1


def test_analysis_can_identify_a_direction_beyond_random_null():
    baseline = [True] * 20
    reached = [True] * 20
    candidate = [False] * 6 + [True] * 14
    runs = [
        {"arm": "baseline", "spec": None, "correctness": baseline, "endpoint_reached": reached},
        {
            "arm": "remove_energy_s11_d0",
            "spec": {"family": "selected_single", "state": 11},
            "correctness": candidate,
            "endpoint_reached": reached,
        },
    ]
    for replicate in range(20):
        runs.append({
            "arm": f"remove_random_s11_r{replicate:02d}",
            "spec": {"family": "random_single", "state": 11},
            "correctness": baseline,
            "endpoint_reached": reached,
        })
    report = analyze_endpoint_inference_ablation(
        runs, bootstrap_samples=100, bootstrap_seed=3
    )
    assert report["accuracy_critical_directions_or_groups"] == [
        "remove_energy_s11_d0"
    ]
