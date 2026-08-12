"""Three preregistered tracks on CODI's answer-cue correctness geometry.

Reuses the colon-state cache the margin-geometry experiment already collects, so
this whole sweep is model-free.  The calibration pool is split three ways and the
discipline is the point:

    fit    (1024)  every direction, probe and steering vector is estimated here
    select (1024)  every hyperparameter -- ridge strength, steering step, rank
    test   (1319)  GSM8K test, read exactly once per arm

Nothing is chosen on the test split.  That closes the standing caveat on the band
experiment, whose boundaries were read off test-set curves, and it is what lets a
positive result here be reported without an asterisk.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.eval.official_codi import select_device
from src.mech.endpoint_correctness_geometry import (
    ACCURACY_BAND,
    CORRECTNESS_CONTRACT,
    CORRECTNESS_SCHEMA_VERSION,
    LIFT_BAND,
    answer_margin,
    apply_logistic,
    band_variance_shares,
    build_steering_vectors,
    class_conditional_basis,
    direction_band_profile,
    first_token_correct,
    fit_correctness_directions,
    fit_logistic,
    principal_angle_cosines,
    random_split_null,
    retained_accuracy,
    roc_auc,
    select_fisher_shrinkage,
    sorted_eigenbasis,
    steered_accuracy,
)
from src.mech.endpoint_margin_geometry import ANALYTIC_STATE
from src.utils.config import load_config


def split_calibration(count: int, seed: int, fit_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < fit_size < count:
        raise ValueError("fit split must be a strict, non-empty subset")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(int(seed)))
    return order[:fit_size], order[fit_size:]


# ---------------------------------------------------------------------------
# track 1 — detect
# ---------------------------------------------------------------------------


def detect_features(
    states: torch.Tensor,
    margin: torch.Tensor,
    directions,
    eigenvectors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Feature sets for the probe comparison, each a [N, d] block.

    ``margin`` is the baseline every other row must beat, and the ``*_plus_margin``
    rows are the only ones that answer the question that matters: does the state
    carry correctness information the output does not already expose?
    """
    values = states.double()
    scalar = margin.double().unsqueeze(1)
    coefficients = values @ eigenvectors.double()
    return {
        "margin": scalar,
        "mean_difference": (values @ directions.mean_difference.double()).unsqueeze(1),
        "fisher": (values @ directions.fisher.double()).unsqueeze(1),
        "lift_band": coefficients[:, LIFT_BAND[0] : LIFT_BAND[1]],
        "accuracy_band": coefficients[:, ACCURACY_BAND[0] : ACCURACY_BAND[1]],
        "full_state": values,
        "fisher_plus_margin": torch.cat(
            [(values @ directions.fisher.double()).unsqueeze(1), scalar], dim=1
        ),
        "accuracy_band_plus_margin": torch.cat(
            [coefficients[:, ACCURACY_BAND[0] : ACCURACY_BAND[1]], scalar], dim=1
        ),
        "full_state_plus_margin": torch.cat([values, scalar], dim=1),
    }


def run_detect(
    *,
    fit: dict,
    select: dict,
    test: dict,
    directions,
    eigenvectors: torch.Tensor,
    ridge_grid: tuple[float, ...],
    probe_steps: int,
) -> dict:
    blocks = {
        name: detect_features(
            split["states"], split["margin"], directions, eigenvectors
        )
        for name, split in (("fit", fit), ("select", select), ("test", test))
    }
    results = {}
    for probe in tqdm(sorted(blocks["fit"]), desc="detect probes"):
        chosen, best = None, -1.0
        for ridge in ridge_grid:
            weight, bias, stats = fit_logistic(
                blocks["fit"][probe], fit["correct"], l2=ridge, steps=probe_steps
            )
            score = roc_auc(
                apply_logistic(blocks["select"][probe], weight, bias, stats),
                select["correct"],
            )
            if score > best:
                chosen, best = (ridge, weight, bias, stats), score
        ridge, weight, bias, stats = chosen
        test_scores = apply_logistic(blocks["test"][probe], weight, bias, stats)
        results[probe] = {
            "ridge": ridge,
            "select_auc": best,
            "test_auc": roc_auc(test_scores, test["correct"]),
            "feature_count": int(blocks["fit"][probe].shape[1]),
            "scores": test_scores,
        }
    return results


# ---------------------------------------------------------------------------
# track 2 — steer
# ---------------------------------------------------------------------------


def analytic_accuracy(
    states: torch.Tensor, readout: torch.Tensor, gold: torch.Tensor
) -> tuple[float, torch.Tensor]:
    outcomes = first_token_correct(states, readout, gold)
    return float(outcomes.double().mean()), outcomes


def run_steer(
    *,
    fit: dict,
    select: dict,
    test: dict,
    readout: torch.Tensor,
    directions,
    eigenvectors: torch.Tensor,
    alpha_grid: tuple[float, ...],
    band: tuple[int, int],
    random_seed: int,
    random_replicates: int,
) -> dict:
    vectors = build_steering_vectors(
        states=fit["states"],
        readout=readout,
        gold=fit["gold"],
        directions=directions,
        eigenvectors=eigenvectors,
        band=band,
        random_seed=random_seed,
        random_replicates=random_replicates,
    )
    # Computed once and reused by every arm at every step size.
    select_logits = select["states"].double() @ readout.double().T
    test_logits = test["states"].double() @ readout.double().T
    results = {}
    for name, vector in tqdm(sorted(vectors.items()), desc="steer arms"):
        # The step size is chosen on the select split; a positive arm that only
        # works at an alpha read off the test curve is not a result.
        curve, best_alpha, best_score = {}, 0.0, -1.0
        for alpha in alpha_grid:
            score, _ = steered_accuracy(
                select_logits, readout, select["gold"], vector, alpha
            )
            curve[f"{alpha:g}"] = score
            if score > best_score:
                best_alpha, best_score = alpha, score
        accuracy, outcomes = steered_accuracy(
            test_logits, readout, test["gold"], vector, best_alpha
        )
        results[name] = {
            "select_curve": curve,
            "selected_alpha": best_alpha,
            "select_accuracy": best_score,
            "test_accuracy": accuracy,
            "band_profile": direction_band_profile(vector, eigenvectors),
            "outcomes": outcomes,
            "vector": vector,
        }
    return results


# ---------------------------------------------------------------------------
# track 3 — project
# ---------------------------------------------------------------------------


def run_project(
    *,
    fit: dict,
    test: dict,
    readout: torch.Tensor,
    eigenvectors: torch.Tensor,
    centre: torch.Tensor,
    rank_grid: tuple[int, ...],
    band: tuple[int, int],
) -> dict:
    results = {}
    for rank in tqdm(rank_grid, desc="project ranks"):
        blind_basis = eigenvectors[:, :rank].double()
        correct_basis, correct_centre = class_conditional_basis(
            fit["states"], fit["correct"], rank, on_correct=True
        )
        wrong_basis, wrong_centre = class_conditional_basis(
            fit["states"], fit["correct"], rank, on_correct=False
        )
        arms = {
            "class_blind": (blind_basis, centre),
            "correct_only": (correct_basis, correct_centre),
            "incorrect_only": (wrong_basis, wrong_centre),
        }
        entry = {}
        for name, (basis, origin) in arms.items():
            accuracy, outcomes = retained_accuracy(
                test["states"], readout, test["gold"], basis, origin
            )
            entry[name] = {"accuracy": accuracy, "outcomes": outcomes}
        angles = principal_angle_cosines(blind_basis, correct_basis)
        entry["overlap_with_class_blind"] = {
            "mean_cosine": float(angles.mean()),
            "minimum_cosine": float(angles.min()),
        }
        results[str(rank)] = entry

    band_basis = eigenvectors[:, band[0] : band[1]].double()
    accuracy, outcomes = retained_accuracy(
        test["states"], readout, test["gold"], band_basis, centre
    )
    results["accuracy_band"] = {
        "band": list(band),
        "class_blind": {"accuracy": accuracy, "outcomes": outcomes},
    }
    return results


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def strip_tensors(value):
    if isinstance(value, dict):
        return {k: strip_tensors(v) for k, v in value.items() if not isinstance(v, torch.Tensor)}
    if isinstance(value, list):
        return [strip_tensors(v) for v in value]
    return value


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml")
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vectors-output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_config(args.config).endpoint_correctness_tracks
    device = torch.device(args.device) if args.device else select_device()

    cache, readout_payload = load_margin_cache(args.states, args.readout)
    readout = readout_payload["output_embedding"].to(device).double()
    index = list(cache["state_order"]).index(ANALYTIC_STATE)
    calibration = cache["calibration_states"][:, index, :].to(device).double()
    evaluation = cache["evaluation_states"][:, index, :].to(device).double()
    calibration_gold = cache["calibration_gold_first_token"].to(device)
    evaluation_gold = cache["evaluation_gold_first_token"].to(device)

    fit_index, select_index = split_calibration(
        calibration.shape[0], settings.split_seed, settings.fit_examples
    )

    def make_split(states, gold):
        return {
            "states": states,
            "gold": gold,
            "correct": first_token_correct(states, readout, gold),
            "margin": answer_margin(states, readout),
        }

    fit = make_split(calibration[fit_index], calibration_gold[fit_index])
    select = make_split(calibration[select_index], calibration_gold[select_index])
    test = make_split(evaluation, evaluation_gold)
    for name, split in (("fit", fit), ("select", select), ("test", test)):
        share = float(split["correct"].double().mean())
        if not 0.05 < share < 0.95:
            raise RuntimeError(f"{name} split is too one-sided ({share:.3f}) to fit on")

    centre = fit["states"].mean(0)
    centred = fit["states"] - centre
    covariance = centred.T @ centred / centred.shape[0]
    values, eigenvectors = sorted_eigenbasis(covariance)
    shrinkage, shrinkage_scores = select_fisher_shrinkage(
        fit["states"],
        fit["correct"],
        select["states"],
        select["correct"],
        grid=tuple(settings.fisher_shrinkage_grid),
    )
    directions = fit_correctness_directions(
        fit["states"], fit["correct"], shrinkage=shrinkage
    )

    geometry = {
        "fisher_shrinkage": shrinkage,
        "fisher_shrinkage_select_auc": shrinkage_scores,
        "between_class_variance": directions.between_variance,
        "total_variance": directions.total_variance,
        "between_class_fraction": directions.between_fraction,
        "mean_difference_norm": float(
            (fit["states"][fit["correct"]].mean(0)
             - fit["states"][~fit["correct"]].mean(0)).norm()
        ),
        "mean_difference_bands": direction_band_profile(
            directions.mean_difference, eigenvectors
        ),
        "fisher_bands": direction_band_profile(directions.fisher, eigenvectors),
        "variance_shares": band_variance_shares(
            values, (LIFT_BAND, ACCURACY_BAND, (32, 768))
        ),
        "random_split_null": random_split_null(
            fit["states"],
            fit["correct"],
            eigenvectors,
            replicates=settings.null_replicates,
            seed=settings.random_seed,
        ),
    }

    detect = run_detect(
        fit=fit,
        select=select,
        test=test,
        directions=directions,
        eigenvectors=eigenvectors,
        ridge_grid=tuple(settings.ridge_grid),
        probe_steps=settings.probe_steps,
    )
    steer_results = run_steer(
        fit=fit,
        select=select,
        test=test,
        readout=readout,
        directions=directions,
        eigenvectors=eigenvectors,
        alpha_grid=tuple(settings.alpha_grid),
        band=tuple(settings.accuracy_band),
        random_seed=settings.random_seed,
        random_replicates=settings.steer_random_replicates,
    )
    project = run_project(
        fit=fit,
        test=test,
        readout=readout,
        eigenvectors=eigenvectors,
        centre=centre,
        rank_grid=tuple(settings.rank_grid),
        band=tuple(settings.accuracy_band),
    )

    baseline, baseline_outcomes = analytic_accuracy(
        test["states"], readout, test["gold"]
    )
    payload = {
        "schema_version": CORRECTNESS_SCHEMA_VERSION,
        "contract": CORRECTNESS_CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_request_sha256": cache.get("request_sha256"),
        "splits": {
            "fit": int(fit["states"].shape[0]),
            "select": int(select["states"].shape[0]),
            "test": int(test["states"].shape[0]),
            "split_seed": settings.split_seed,
            "correct_share": {
                name: float(split["correct"].double().mean())
                for name, split in (("fit", fit), ("select", select), ("test", test))
            },
        },
        "baseline_first_token_accuracy": baseline,
        "margin_auc": roc_auc(test["margin"], test["correct"]),
        "geometry": geometry,
        "detect": strip_tensors(detect),
        "steer": strip_tensors(steer_results),
        "project": strip_tensors(project),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(payload, args.output)

    vectors_path = args.vectors_output or args.output.with_suffix(".pt")
    _atomic_torch_save(
        {
            "schema_version": CORRECTNESS_SCHEMA_VERSION,
            "contract": CORRECTNESS_CONTRACT,
            "source_request_sha256": cache.get("request_sha256"),
            "steering_vectors": {
                name: entry["vector"].cpu().float()
                for name, entry in steer_results.items()
            },
            "selected_alpha": {
                name: entry["selected_alpha"] for name, entry in steer_results.items()
            },
            "outcomes": {
                "labels": test["correct"].cpu(),
                "baseline": baseline_outcomes.cpu(),
                "steer": {
                    name: entry["outcomes"].cpu()
                    for name, entry in steer_results.items()
                },
                "detect": {
                    name: entry["scores"].cpu() for name, entry in detect.items()
                },
                "project": {
                    rank: {
                        arm: entry[arm]["outcomes"].cpu()
                        for arm in ("class_blind", "correct_only", "incorrect_only")
                    }
                    for rank, entry in project.items()
                    if rank != "accuracy_band"
                },
            },
            "eigenvectors": eigenvectors.cpu().float(),
            "fit_centre": centre.cpu().float(),
        },
        vectors_path,
    )
    print(f"wrote {args.output} and {vectors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
