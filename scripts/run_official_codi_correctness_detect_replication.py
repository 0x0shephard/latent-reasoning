"""Run the test-like, convergence-audited CODI correctness detector replication.

Only the cached 1,319 GSM8K *test* states are used. They are partitioned once into
fit/select/test. This is a new corrective contract and never overwrites the
completed three-track export.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.eval.official_codi_correctness_detect_replication_analysis import (
    DETECT_REPLICATION_CONTRACT,
)
from src.mech.endpoint_correctness_geometry import (
    ACCURACY_BAND,
    LIFT_BAND,
    answer_margin,
    apply_logistic,
    first_token_correct,
    fit_correctness_directions,
    fit_logistic_checked,
    readout_matrix,
    roc_auc,
    select_fisher_shrinkage,
    sorted_eigenbasis,
)
from src.mech.endpoint_margin_geometry import ANALYTIC_STATE
from src.utils.config import load_config


DETECT_REPLICATION_SCHEMA_VERSION = 1


def _sha256_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def split_test_like(
    count: int,
    *,
    seed: int,
    fit_size: int,
    select_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one deterministic, exhaustive, disjoint three-way partition."""
    if fit_size <= 0 or select_size <= 0:
        raise ValueError("fit and select splits must be positive")
    if fit_size + select_size >= count:
        raise ValueError("the partition must leave a non-empty test split")
    order = torch.randperm(
        count, generator=torch.Generator().manual_seed(int(seed)), device="cpu"
    )
    return (
        order[:fit_size],
        order[fit_size : fit_size + select_size],
        order[fit_size + select_size :],
    )


def detect_features(states, margin, directions, eigenvectors) -> dict[str, torch.Tensor]:
    values = states.double()
    scalar = margin.double().unsqueeze(1)
    coefficients = values @ eigenvectors.double()
    return {
        "margin": scalar,
        "mean_difference": (
            values @ directions.mean_difference.double()
        ).unsqueeze(1),
        "fisher": (values @ directions.fisher.double()).unsqueeze(1),
        "full_state": values,
        "fisher_plus_margin": torch.cat(
            [(values @ directions.fisher.double()).unsqueeze(1), scalar], dim=1
        ),
        "full_state_plus_margin": torch.cat([values, scalar], dim=1),
        "lift_band": coefficients[:, LIFT_BAND[0] : LIFT_BAND[1]],
        "accuracy_band": coefficients[:, ACCURACY_BAND[0] : ACCURACY_BAND[1]],
        "accuracy_band_plus_margin": torch.cat(
            [coefficients[:, ACCURACY_BAND[0] : ACCURACY_BAND[1]], scalar], dim=1
        ),
    }


def run_probes(fit, select, test, directions, eigenvectors, settings) -> dict:
    requested = tuple(str(name) for name in settings.probes)
    blocks = {
        split_name: detect_features(
            split["states"], split["margin"], directions, eigenvectors
        )
        for split_name, split in (("fit", fit), ("select", select), ("test", test))
    }
    unknown = sorted(set(requested) - set(blocks["fit"]))
    if unknown:
        raise ValueError(f"unknown detect probes: {unknown}")

    results = {}
    for probe in tqdm(requested, desc="checked detect probes"):
        candidates = []
        for ridge in settings.ridge_grid:
            weight, bias, stats = fit_logistic_checked(
                blocks["fit"][probe],
                fit["correct"],
                l2=float(ridge),
                max_iterations=int(settings.solver_max_iterations),
                gradient_tolerance=float(settings.solver_gradient_tolerance),
                objective_gap_tolerance=float(
                    settings.solver_objective_gap_tolerance
                ),
            )
            optimization = stats["optimization"]
            if not optimization["converged"]:
                raise RuntimeError(
                    f"{probe} ridge={ridge:g} did not converge: "
                    f"gradient_inf={optimization['gradient_inf_norm']:.3e}, "
                    f"gap_bound={optimization['objective_gap_upper_bound']:.3e}"
                )
            select_scores = apply_logistic(
                blocks["select"][probe], weight, bias, stats
            )
            candidates.append(
                (
                    roc_auc(select_scores, select["correct"]),
                    float(ridge),
                    weight,
                    bias,
                    stats,
                )
            )
        select_auc, ridge, weight, bias, stats = max(
            candidates, key=lambda item: (item[0], -item[1])
        )
        test_scores = apply_logistic(blocks["test"][probe], weight, bias, stats)
        results[probe] = {
            "ridge": ridge,
            "feature_count": int(blocks["fit"][probe].shape[1]),
            "select_auc": float(select_auc),
            "test_auc": roc_auc(test_scores, test["correct"]),
            "optimization": stats["optimization"],
            "scores": test_scores,
        }
    return results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes-output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_config(args.config).endpoint_correctness_detect_replication
    device = torch.device(args.device)
    cache, readout_payload = load_margin_cache(args.states, args.readout)
    readout = readout_matrix(readout_payload).to(device).double()
    state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
    states = cache["evaluation_states"][:, state_index, :].to(device).double()
    gold = cache["evaluation_gold_first_token"].to(device)
    if states.shape[0] != int(settings.expected_examples):
        raise RuntimeError(
            f"expected {settings.expected_examples} GSM8K test states, "
            f"found {states.shape[0]}"
        )

    fit_index, select_index, test_index = split_test_like(
        states.shape[0],
        seed=int(settings.split_seed),
        fit_size=int(settings.fit_examples),
        select_size=int(settings.select_examples),
    )

    def make_split(index):
        split_states = states[index]
        split_gold = gold[index]
        return {
            "states": split_states,
            "gold": split_gold,
            "correct": first_token_correct(split_states, readout, split_gold),
            "margin": answer_margin(split_states, readout),
        }

    fit, select, test = (
        make_split(fit_index),
        make_split(select_index),
        make_split(test_index),
    )
    for name, split in (("fit", fit), ("select", select), ("test", test)):
        share = float(split["correct"].double().mean())
        if not 0.05 < share < 0.95:
            raise RuntimeError(f"{name} split is too one-sided ({share:.3f})")

    centred = fit["states"] - fit["states"].mean(0)
    _, eigenvectors = sorted_eigenbasis(centred.T @ centred / centred.shape[0])
    shrinkage, shrinkage_scores = select_fisher_shrinkage(
        fit["states"],
        fit["correct"],
        select["states"],
        select["correct"],
        grid=tuple(float(value) for value in settings.fisher_shrinkage_grid),
    )
    directions = fit_correctness_directions(
        fit["states"], fit["correct"], shrinkage=shrinkage
    )
    probes = run_probes(fit, select, test, directions, eigenvectors, settings)

    indices = {
        "fit": fit_index.tolist(),
        "select": select_index.tolist(),
        "test": test_index.tolist(),
    }
    payload = {
        "schema_version": DETECT_REPLICATION_SCHEMA_VERSION,
        "contract": DETECT_REPLICATION_CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_request_sha256": cache.get("request_sha256"),
        "population": {
            "dataset": "GSM8K test",
            "source_cache_field": "evaluation_states",
            "expected_examples": int(settings.expected_examples),
            "test_like_relative_to_original_calibration": True,
            "previously_inspected_by_project": True,
        },
        "splits": {
            "fit": len(indices["fit"]),
            "select": len(indices["select"]),
            "test": len(indices["test"]),
            "split_seed": int(settings.split_seed),
            "partition_sha256": _sha256_json(indices),
            "correct_share": {
                name: float(split["correct"].double().mean())
                for name, split in (("fit", fit), ("select", select), ("test", test))
            },
        },
        "fisher_shrinkage": shrinkage,
        "fisher_shrinkage_select_auc": shrinkage_scores,
        "probes": {
            name: {key: value for key, value in entry.items() if key != "scores"}
            for name, entry in probes.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(payload, args.output)
    outcomes_path = args.outcomes_output or args.output.with_suffix(".pt")
    _atomic_torch_save(
        {
            "schema_version": DETECT_REPLICATION_SCHEMA_VERSION,
            "contract": DETECT_REPLICATION_CONTRACT,
            "source_request_sha256": cache.get("request_sha256"),
            "partition_sha256": payload["splits"]["partition_sha256"],
            "indices": indices,
            "labels": test["correct"].cpu(),
            "scores": {name: entry["scores"].cpu() for name, entry in probes.items()},
        },
        outcomes_path,
    )
    print(f"wrote {args.output} and {outcomes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
