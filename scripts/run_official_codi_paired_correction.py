"""Fit and analytically test a paired, question-conditioned state-12 correction."""
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
from src.mech.endpoint_correctness_geometry import readout_matrix, sorted_eigenbasis
from src.mech.endpoint_margin_geometry import answer_token_outcomes
from src.mech.endpoint_paired_correction import (
    PAIRED_CORRECTION_CONTRACT,
    PAIRED_CORRECTION_SCHEMA_VERSION,
    RidgeCorrection,
    correction_features,
    fit_ridge_correction,
    paired_question_examples,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate_predictions(
    states: torch.Tensor,
    gold: torch.Tensor,
    readout: torch.Tensor,
    *,
    basis: torch.Tensor | None = None,
    predicted_coefficients: torch.Tensor | None = None,
    alpha: float = 0.0,
    maximum_margin: float = float("-inf"),
    chunk_size: int = 64,
) -> dict:
    """Exact state-12 token outcomes for one additive correction arm."""
    values, weights = states.double(), readout.double()
    all_nll, all_margin, all_correct, all_gate = [], [], [], []
    delta_readout = None
    if basis is not None:
        if predicted_coefficients is None or predicted_coefficients.shape[0] != states.shape[0]:
            raise ValueError("each state needs predicted correction coefficients")
        delta_readout = basis.double().T @ weights.T
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        logits = values[start:stop] @ weights.T
        top = logits.topk(2, dim=1).values
        confidence = top[:, 0] - top[:, 1]
        gate = confidence <= float(maximum_margin)
        if delta_readout is not None and float(alpha) != 0:
            shifts = predicted_coefficients[start:stop].double() @ delta_readout
            logits[gate] += float(alpha) * shifts[gate]
        outcomes = answer_token_outcomes(logits, gold[start:stop])
        all_nll.append(outcomes["nll"].cpu())
        all_margin.append(outcomes["margin"].cpu())
        all_correct.append(outcomes["top1_correct"].cpu())
        all_gate.append(gate.cpu())
    result = {
        "nll": torch.cat(all_nll),
        "margin": torch.cat(all_margin),
        "correct": torch.cat(all_correct),
        "gate": torch.cat(all_gate),
    }
    result["summary"] = {
        "accuracy": float(result["correct"].double().mean()),
        "mean_gold_nll": float(result["nll"].mean()),
        "mean_gold_margin": float(result["margin"].mean()),
        "edited_fraction": float(result["gate"].double().mean()),
    }
    return result


def _model_predictions(
    model: RidgeCorrection,
    states: torch.Tensor,
    *,
    basis: torch.Tensor,
    centre: torch.Tensor,
    readout: torch.Tensor,
) -> torch.Tensor:
    return model.predict(
        correction_features(
            states, basis=basis, centre=centre, readout=readout
        )
    )


def select_intervention(
    model: RidgeCorrection,
    states: torch.Tensor,
    gold: torch.Tensor,
    readout: torch.Tensor,
    *,
    basis: torch.Tensor,
    centre: torch.Tensor,
    alpha_grid,
    gate_fractions,
) -> tuple[dict, list[dict]]:
    predictions = _model_predictions(
        model, states, basis=basis, centre=centre, readout=readout
    )
    baseline_logits = states.double() @ readout.double().T
    correction_logits = predictions.double() @ (
        basis.double().T @ readout.double().T
    )
    top = baseline_logits.topk(2, dim=1).values
    margins = top[:, 0] - top[:, 1]
    candidates = []
    for fraction in gate_fractions:
        fraction = float(fraction)
        threshold = (
            float(margins.min() - 1.0)
            if fraction == 0
            else float(torch.quantile(margins, fraction))
        )
        for alpha in alpha_grid:
            gate = margins <= threshold
            logits = baseline_logits.clone()
            if float(alpha) != 0:
                logits[gate] += float(alpha) * correction_logits[gate]
            accuracy = float(
                (logits.argmax(1) == gold.to(logits.device)).double().mean()
            )
            candidates.append(
                {
                    "alpha": float(alpha),
                    "gate_fraction_requested": fraction,
                    "maximum_margin": threshold,
                    "accuracy": accuracy,
                    "edited_fraction": float(gate.double().mean()),
                }
            )
    # Prefer accuracy, then less intervention, then smaller magnitude.
    selected = max(
        candidates,
        key=lambda item: (
            item["accuracy"],
            -item["edited_fraction"],
            -item["alpha"],
        ),
    )
    return selected, candidates


def _global_model(reference: RidgeCorrection, target_mean: torch.Tensor) -> RidgeCorrection:
    return RidgeCorrection(
        weight=torch.zeros_like(reference.weight),
        bias=target_mean.double(),
        feature_mean=reference.feature_mean,
        feature_scale=reference.feature_scale,
        ridge=reference.ridge,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path)
    args = parser.parse_args(argv)

    settings = load_config(args.config).endpoint_paired_correction
    pairs = torch.load(args.pairs, map_location="cpu", weights_only=False)
    if pairs.get("contract") != PAIRED_CORRECTION_CONTRACT:
        raise RuntimeError("paired artifact belongs to another contract")
    readout_payload = torch.load(args.readout, map_location="cpu", weights_only=False)
    readout = readout_matrix(readout_payload).double()
    states = pairs["fit_select_states"].double()
    correct = pairs["fit_select_correct"].bool()
    gold = pairs["fit_select_gold_first_token"].long()
    fit_count = int(settings.fit_examples)
    select_count = int(settings.select_examples)
    if states.shape[:2] != correct.shape or states.shape[0] != fit_count + select_count:
        raise RuntimeError("paired fit/select tensor has the wrong shape")

    fit_baseline = states[:fit_count, 0]
    select_baseline = states[fit_count:, 0]
    fit_centre = fit_baseline.mean(0)
    centered = fit_baseline - fit_centre
    covariance = centered.T @ centered / fit_count
    _, eigenvectors = sorted_eigenbasis(covariance)
    band_start, band_stop = [int(value) for value in settings.accuracy_band]
    basis = eigenvectors[:, band_start:band_stop].contiguous()
    rank = basis.shape[1]

    fit_pairs = paired_question_examples(
        states[:fit_count],
        correct[:fit_count],
        basis,
        fit_centre,
        readout,
    )
    select_pairs = paired_question_examples(
        states[fit_count:],
        correct[fit_count:],
        basis,
        fit_centre,
        readout,
    )
    ridge_candidates = []
    for ridge in settings.ridge_grid:
        model = fit_ridge_correction(
            fit_pairs["features"], fit_pairs["targets"], ridge=float(ridge)
        )
        prediction = model.predict(select_pairs["features"])
        mse = float((prediction - select_pairs["targets"]).square().mean())
        cosine = torch.nn.functional.cosine_similarity(
            prediction, select_pairs["targets"], dim=1
        )
        ridge_candidates.append(
            {
                "ridge": float(ridge),
                "select_target_mse": mse,
                "select_target_mean_cosine": float(cosine.mean()),
                "model": model,
            }
        )
    selected_ridge = min(
        ridge_candidates, key=lambda item: (item["select_target_mse"], -item["ridge"])
    )
    primary_model = selected_ridge["model"]
    global_model = _global_model(primary_model, fit_pairs["targets"].mean(0))
    generator = torch.Generator().manual_seed(int(settings.shuffle_seed))
    permutation = torch.randperm(fit_pairs["targets"].shape[0], generator=generator)
    shuffled_model = fit_ridge_correction(
        fit_pairs["features"],
        fit_pairs["targets"][permutation],
        ridge=primary_model.ridge,
    )

    models = {
        "conditioned": primary_model,
        "global_mean": global_model,
        "shuffled_target": shuffled_model,
    }
    selected_interventions, selection_curves = {}, {}
    for name, model in models.items():
        selected, curve = select_intervention(
            model,
            select_baseline,
            gold[fit_count:],
            readout,
            basis=basis,
            centre=fit_centre,
            alpha_grid=settings.alpha_grid,
            gate_fractions=settings.gate_fractions,
        )
        selected_interventions[name] = selected
        selection_curves[name] = curve

    test_states = pairs["test_baseline_states"].double()
    test_gold = pairs["test_gold_first_token"].long()
    outcomes = {
        "baseline": evaluate_predictions(test_states, test_gold, readout)
    }
    for name, model in models.items():
        selected = selected_interventions[name]
        prediction = _model_predictions(
            model, test_states, basis=basis, centre=fit_centre, readout=readout
        )
        outcomes[name] = evaluate_predictions(
            test_states,
            test_gold,
            readout,
            basis=basis,
            predicted_coefficients=prediction,
            alpha=float(selected["alpha"]),
            maximum_margin=float(selected["maximum_margin"]),
        )

    partition_sha256 = pairs["partition_sha256"]
    summary = {
        "schema_version": PAIRED_CORRECTION_SCHEMA_VERSION,
        "contract": PAIRED_CORRECTION_CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_request_sha256": pairs.get("source_request_sha256"),
        "pairs_request_sha256": pairs["request_sha256"],
        "partition_sha256": partition_sha256,
        "splits": {
            "fit": fit_count,
            "select": select_count,
            "test": int(test_states.shape[0]),
            "fit_paired_questions": int(fit_pairs["features"].shape[0]),
            "select_paired_questions": int(select_pairs["features"].shape[0]),
        },
        "state_perturbed_for_pairs": 11,
        "state_edited_at_deployment": 12,
        "accuracy_band": [band_start, band_stop],
        "rank": rank,
        "selected_ridge": primary_model.ridge,
        "ridge_selection": [
            {key: value for key, value in item.items() if key != "model"}
            for item in ridge_candidates
        ],
        "selected_interventions": selected_interventions,
        "selection_curves": selection_curves,
        "test_arms": {name: result["summary"] for name, result in outcomes.items()},
    }
    artifact_path = args.artifact_output or args.output.with_suffix(".pt")
    model_payload = {
        name: {
            "correction": model.state_dict(),
            "selection": selected_interventions[name],
        }
        for name, model in models.items()
    }
    artifact = {
        "schema_version": PAIRED_CORRECTION_SCHEMA_VERSION,
        "contract": PAIRED_CORRECTION_CONTRACT,
        "source_request_sha256": pairs.get("source_request_sha256"),
        "pairs_request_sha256": pairs["request_sha256"],
        "partition_sha256": partition_sha256,
        "indices": pairs["indices"],
        "basis": basis.float(),
        "centre": fit_centre.float(),
        "models": model_payload,
        "outcomes": {
            name: {key: value for key, value in result.items() if key != "summary"}
            for name, result in outcomes.items()
        },
    }
    _atomic_torch_save(artifact, artifact_path)
    _atomic_json(summary, args.output)
    print(f"wrote {args.output} and {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
