"""Fit and evaluate 28-D correct/wrong contrastive covariance subspaces.

The cached GSM8K test colon states are split once into 440 fit, 440 selection,
and 439 final-test examples.  Selection sees only an intrinsic held-out energy
ratio; every answer metric and every pass/fail gate is computed once on test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.run_official_codi_correctness_detect_replication import split_test_like
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.mech.endpoint_correctness_contrastive_covariance import (
    CONTRASTIVE_COVARIANCE_CONTRACT,
    CONTRASTIVE_COVARIANCE_SCHEMA_VERSION,
    covariance,
    fit_contrastive_covariance,
    heldout_specificity_score,
)
from src.mech.endpoint_correctness_geometry import (
    class_conditional_basis,
    first_token_correct,
    readout_matrix,
    sorted_eigenbasis,
)
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    answer_token_outcomes,
    energy_matched_random_subspace,
    subspace_energy,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate_arm(
    states: torch.Tensor,
    readout: torch.Tensor,
    gold: torch.Tensor,
    *,
    basis: torch.Tensor | None,
    centre: torch.Tensor | None,
    mode: str = "baseline",
    chunk_size: int = 64,
) -> dict:
    """Evaluate one state-12 arm without materialising every arm's full logits."""
    nll, margin, correct = [], [], []
    values, weights = states.double(), readout.double()
    low_rank_weights = None if basis is None else basis.double().T @ weights.T
    centre_logits = None if centre is None else centre.double() @ weights.T
    for start in range(0, values.shape[0], chunk_size):
        stop = start + chunk_size
        hidden = values[start:stop]
        if basis is None:
            logits = hidden @ weights.T
        else:
            component_logits = (
                (hidden - centre.double()) @ basis.double()
            ) @ low_rank_weights
            if mode == "retain":
                logits = centre_logits.unsqueeze(0) + component_logits
            elif mode == "remove":
                logits = hidden @ weights.T - component_logits
            else:
                raise ValueError("intervention mode must be retain or remove")
        outcomes = answer_token_outcomes(logits, gold[start:stop])
        nll.append(outcomes["nll"].cpu())
        margin.append(outcomes["margin"].cpu())
        correct.append(outcomes["top1_correct"].cpu())
    result = {
        "nll": torch.cat(nll),
        "margin": torch.cat(margin),
        "correct": torch.cat(correct),
    }
    result["summary"] = {
        "accuracy": float(result["correct"].double().mean()),
        "mean_gold_nll": float(result["nll"].mean()),
        "mean_gold_margin": float(result["margin"].mean()),
    }
    return result


def _basis_record(name, basis, centre, mode, family, **metadata):
    return {
        "name": name,
        "basis": basis.cpu().float(),
        "centre": centre.cpu().float(),
        "mode": mode,
        "family": family,
        **metadata,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    settings = load_config(args.config).endpoint_correctness_contrastive_covariance
    cache, readout_payload = load_margin_cache(args.states, args.readout)
    device = torch.device(args.device)
    readout = readout_matrix(readout_payload).to(device).double()
    state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
    states = cache["evaluation_states"][:, state_index, :].to(device).double()
    gold = cache["evaluation_gold_first_token"].to(device)
    if states.shape[0] != int(settings.expected_examples):
        raise RuntimeError(
            f"expected {settings.expected_examples} GSM8K test states, found {states.shape[0]}"
        )

    fit_index, select_index, test_index = split_test_like(
        states.shape[0],
        seed=int(settings.split_seed),
        fit_size=int(settings.fit_examples),
        select_size=int(settings.select_examples),
    )

    def split(index):
        split_states = states[index]
        split_gold = gold[index]
        return {
            "states": split_states,
            "gold": split_gold,
            "correct": first_token_correct(split_states, readout, split_gold),
        }

    fit, select, test = split(fit_index), split(select_index), split(test_index)
    for name, part in (("fit", fit), ("select", select), ("test", test)):
        if not 0.05 < float(part["correct"].double().mean()) < 0.95:
            raise RuntimeError(f"{name} correctness labels are too one-sided")

    rank = int(settings.rank)
    candidates = []
    for shrinkage in settings.shrinkage_grid:
        fitted = fit_contrastive_covariance(
            fit["states"], fit["correct"], rank=rank, shrinkage=float(shrinkage)
        )
        correct_score = heldout_specificity_score(
            select["states"],
            select["correct"],
            fitted.correct_basis,
            correct_mean=fitted.correct_mean,
            wrong_mean=fitted.wrong_mean,
            orientation="correct",
        )
        wrong_score = heldout_specificity_score(
            select["states"],
            select["correct"],
            fitted.wrong_basis,
            correct_mean=fitted.correct_mean,
            wrong_mean=fitted.wrong_mean,
            orientation="wrong",
        )
        candidates.append((correct_score, -float(shrinkage), wrong_score, fitted))
    _, _, _, fitted = max(candidates, key=lambda item: (item[0], item[1]))

    fit_centre = fit["states"].mean(0)
    fit_covariance = covariance(fit["states"], fit_centre)
    _, all_vectors = sorted_eigenbasis(fit_covariance)
    band_start, band_stop = (int(value) for value in settings.accuracy_band)
    if band_stop - band_start != rank:
        raise RuntimeError("the frozen accuracy band must have the requested rank")
    classblind_top_basis = all_vectors[:, :rank]
    accuracy_band_basis = all_vectors[:, band_start:band_stop]
    correct_pca, correct_mean = class_conditional_basis(
        fit["states"], fit["correct"], rank, on_correct=True
    )
    wrong_pca, wrong_mean = class_conditional_basis(
        fit["states"], fit["correct"], rank, on_correct=False
    )

    arms = {
        "accuracy_band_pca_retain": _basis_record(
            "accuracy_band_pca_retain",
            accuracy_band_basis,
            correct_mean,
            "retain",
            "classblind_accuracy_band_pca",
            band=[band_start, band_stop],
        ),
        "classblind_top_pca_retain": _basis_record(
            "classblind_top_pca_retain",
            classblind_top_basis,
            correct_mean,
            "retain",
            "classblind_top_pca",
        ),
        "correct_only_pca_retain": _basis_record(
            "correct_only_pca_retain", correct_pca, correct_mean, "retain", "pca"
        ),
        "wrong_only_pca_retain": _basis_record(
            "wrong_only_pca_retain", wrong_pca, wrong_mean, "retain", "pca"
        ),
        "contrastive_correct_retain": _basis_record(
            "contrastive_correct_retain",
            fitted.correct_basis,
            fitted.correct_mean,
            "retain",
            "contrastive_correct",
        ),
        "contrastive_correct_global_centre_retain": _basis_record(
            "contrastive_correct_global_centre_retain",
            fitted.correct_basis,
            fitted.global_mean,
            "retain",
            "centre_diagnostic",
        ),
        "contrastive_wrong_retain": _basis_record(
            "contrastive_wrong_retain",
            fitted.wrong_basis,
            fitted.wrong_mean,
            "retain",
            "contrastive_wrong",
        ),
        "contrastive_wrong_remove": _basis_record(
            "contrastive_wrong_remove",
            fitted.wrong_basis,
            fitted.global_mean,
            "remove",
            "contrastive_wrong",
        ),
    }

    generator = torch.Generator(device="cpu").manual_seed(int(settings.random_seed))
    correct_target = subspace_energy(fitted.correct_covariance, fitted.correct_basis)
    wrong_target = subspace_energy(fitted.wrong_covariance, fitted.wrong_basis)
    random_diagnostics = {}
    for replicate in range(int(settings.random_replicates)):
        retain_basis, retain_diagnostics = energy_matched_random_subspace(
            covariance=fitted.correct_covariance.cpu(),
            rank=rank,
            target_energy=correct_target,
            generator=generator,
        )
        remove_basis, remove_diagnostics = energy_matched_random_subspace(
            covariance=fitted.wrong_covariance.cpu(),
            rank=rank,
            target_energy=wrong_target,
            generator=generator,
        )
        retain_name = f"random_correct_energy_retain_r{replicate:02d}"
        remove_name = f"random_wrong_energy_remove_r{replicate:02d}"
        arms[retain_name] = _basis_record(
            retain_name, retain_basis, fitted.correct_mean, "retain", "matched_random"
        )
        arms[remove_name] = _basis_record(
            remove_name, remove_basis, fitted.global_mean, "remove", "matched_random"
        )
        random_diagnostics[retain_name] = retain_diagnostics
        random_diagnostics[remove_name] = remove_diagnostics

    outcomes = {
        "baseline": evaluate_arm(
            test["states"], readout, test["gold"], basis=None, centre=None
        )
    }
    for name, arm in arms.items():
        outcomes[name] = evaluate_arm(
            test["states"],
            readout,
            test["gold"],
            basis=arm["basis"].to(device),
            centre=arm["centre"].to(device),
            mode=arm["mode"],
        )

    indices = {
        "fit": fit_index.tolist(),
        "select": select_index.tolist(),
        "test": test_index.tolist(),
    }
    partition_sha256 = _sha256_json(indices)
    summary = {
        "schema_version": CONTRASTIVE_COVARIANCE_SCHEMA_VERSION,
        "contract": CONTRASTIVE_COVARIANCE_CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_request_sha256": cache.get("request_sha256"),
        "population": "GSM8K test cached state-12 answer-cue states",
        "splits": {
            "fit": len(indices["fit"]),
            "select": len(indices["select"]),
            "test": len(indices["test"]),
            "split_seed": int(settings.split_seed),
            "partition_sha256": partition_sha256,
            "correct_share": {
                name: float(part["correct"].double().mean())
                for name, part in (("fit", fit), ("select", select), ("test", test))
            },
        },
        "rank": rank,
        "selected_shrinkage": fitted.shrinkage,
        "selection_criterion": "held-out log correct/wrong projection-energy ratio",
        "shrinkage_selection": {
            f"{entry[3].shrinkage:g}": {
                "correct_specificity": float(entry[0]),
                "wrong_specificity": float(entry[2]),
            }
            for entry in candidates
        },
        "generalized_eigenvalues": {
            "largest": fitted.generalized_eigenvalues[:rank].tolist(),
            "smallest": fitted.generalized_eigenvalues[-rank:].tolist(),
        },
        "random_control_diagnostics": random_diagnostics,
        "arms": {name: result["summary"] for name, result in outcomes.items()},
    }
    artifact_path = args.artifact_output or args.output.with_suffix(".pt")
    _atomic_torch_save(
        {
            "schema_version": CONTRASTIVE_COVARIANCE_SCHEMA_VERSION,
            "contract": CONTRASTIVE_COVARIANCE_CONTRACT,
            "source_request_sha256": cache.get("request_sha256"),
            "partition_sha256": partition_sha256,
            "indices": indices,
            "rank": rank,
            "selected_shrinkage": fitted.shrinkage,
            "arms": arms,
            "outcomes": {
                name: {key: value for key, value in result.items() if key != "summary"}
                for name, result in outcomes.items()
            },
        },
        artifact_path,
    )
    _atomic_json(summary, args.output)
    print(f"wrote {args.output} and {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
