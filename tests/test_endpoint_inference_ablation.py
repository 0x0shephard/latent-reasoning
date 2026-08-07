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
    endpoint_ablation_spec_state,
)
from src.mech.endpoint_accuracy_localization import (
    build_accuracy_localization_specs,
    build_selected_localization_specs,
    energy_matched_random_subspace,
    projection_energy,
)
from src.eval.official_codi_endpoint_accuracy_localization_analysis import (
    analyze_endpoint_accuracy_localization,
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
    old_state = endpoint_ablation_spec_state(specs["remove_energy_joint"])
    assert "matched_method" not in old_state
    assert "active_direction_slots" not in old_state


def test_localization_registry_has_hierarchical_arms():
    bases = {
        "energy": _retention_basis("energy", 0),
        "answer_conditioned": _retention_basis("answer_conditioned", 64),
        "parameter_aware": _retention_basis("parameter_aware", 128),
    }
    specs = build_selected_localization_specs(bases)
    assert len(specs) == 31
    assert "remove_energy_joint_negative_control" in specs
    assert "remove_answer_conditioned_state11" in specs
    reduced = specs["remove_parameter_aware_joint_except_s12_d1"]
    assert reduced.total_rank == 5
    assert reduced.ranks[11] == 3 and reduced.ranks[12] == 2


def test_random_subspace_matches_calibration_energy_without_selected_overlap():
    covariance = torch.diag(torch.linspace(0.1, 20.0, 768))
    selected = torch.zeros(768, 3)
    selected[300, 0] = 1
    selected[400, 1] = 1
    selected[500, 2] = 1
    generator = torch.Generator().manual_seed(19)
    random, diagnostics = energy_matched_random_subspace(
        covariance, selected, generator=generator
    )
    assert torch.allclose(random.T @ random, torch.eye(3), atol=2e-5, rtol=2e-5)
    assert abs(projection_energy(covariance, random) - projection_energy(covariance, selected)) < 1e-4
    assert diagnostics["normalized_selected_overlap"] <= 0.20


def test_complete_localization_registry_builds_method_specific_matched_nulls():
    bases = {
        "energy": _retention_basis("energy", 0),
        "answer_conditioned": _retention_basis("answer_conditioned", 64),
        "parameter_aware": _retention_basis("parameter_aware", 128),
    }
    covariance = torch.diag(torch.linspace(0.1, 20.0, 768))
    specs = build_accuracy_localization_specs(
        bases,
        {11: covariance, 12: covariance},
        random_replicates=2,
        random_seed=23,
    )
    assert len(specs) == 35
    for method in ("answer_conditioned", "parameter_aware"):
        spec = specs[f"remove_matched_random_{method}_joint_r001"]
        assert spec.matched_method == method
        assert max(value for _, value in spec.selected_overlap_by_state) < 1e-8


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


def test_hierarchical_localization_identifies_single_and_rescue_evidence():
    baseline = [True] * 40
    reached = [True] * 40

    def values(false_indices=()):
        result = baseline.copy()
        for index in false_indices:
            result[index] = False
        return result

    def run(name, family, correctness, **spec):
        return {
            "arm": name,
            "spec": {"family": family, **spec},
            "correctness": correctness,
            "endpoint_reached": reached,
            "intervention_diagnostics": {
                "removed_projection_rms_by_state": {"11": 1.0, "12": 1.0}
            },
        }

    runs = [
        {
            "arm": "baseline",
            "spec": None,
            "correctness": baseline,
            "endpoint_reached": reached,
        },
        run(
            "remove_energy_joint_negative_control",
            "negative_control_joint",
            values((0,)),
        ),
    ]
    joint_false = {
        "answer_conditioned": tuple(range(10)),
        "parameter_aware": tuple(range(8)),
    }
    for method in ("answer_conditioned", "parameter_aware"):
        runs.append(
            run(
                f"remove_{method}_joint",
                "selected_joint",
                values(joint_false[method]),
                method=method,
            )
        )
        for state in (11, 12):
            runs.append(
                run(
                    f"remove_{method}_state{state}",
                    "selected_state",
                    values((0, 1, 2, 3)),
                    method=method,
                    state=state,
                )
            )
            for slot in range(3):
                core = state == 11 and slot == 0
                single_false = tuple(range(8)) if core else ()
                reduced_false = (8, 9) if method == "answer_conditioned" and core else ()
                if method == "parameter_aware" and core:
                    reduced_false = ()
                if not core:
                    reduced_false = joint_false[method]
                runs.append(
                    run(
                        f"remove_{method}_s{state}_d{slot}",
                        "selected_single",
                        values(single_false),
                        method=method,
                        state=state,
                        direction_slot=slot,
                        residual_pc_index=slot,
                    )
                )
                runs.append(
                    run(
                        f"remove_{method}_joint_except_s{state}_d{slot}",
                        "selected_joint_minus_one",
                        values(reduced_false),
                        method=method,
                        state=state,
                        direction_slot=slot,
                    )
                )
        for replicate in range(100):
            runs.append(
                run(
                    f"remove_matched_random_{method}_joint_r{replicate:03d}",
                    "matched_random_joint",
                    baseline,
                    matched_method=method,
                    calibration_target_energy_by_state={"11": 1.0, "12": 1.0},
                    calibration_achieved_energy_by_state={"11": 1.0, "12": 1.0},
                    selected_overlap_by_state={"11": 0.01, "12": 0.01},
                )
            )
    report = analyze_endpoint_accuracy_localization(
        runs, bootstrap_samples=200, bootstrap_seed=5
    )
    assert set(report["critical_joint_subspaces"]) == {
        "remove_answer_conditioned_joint",
        "remove_parameter_aware_joint",
    }
    for method in ("answer_conditioned", "parameter_aware"):
        direction = report["localization"][method]["directions"][
            "state11_slot0_pc0"
        ]
        assert direction["individually_necessary"]
        assert direction["rescues_joint_ablation"]
        assert direction["accuracy_core_direction"]
