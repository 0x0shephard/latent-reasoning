"""Probe the collected latent trajectory for correctness and answer identity.

CPU-capable: every probe reads the cached trajectory export. All probes are
fitted on the fit split, every cell and ridge is chosen on the select split, and
the final test split is read exactly once per frozen arm. The correctness probes
use the convergence-certified logistic solver; the answer-identity probes use an
exact closed-form one-hot ridge, which has no optimizer to certify.
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
from src.eval.official_codi_latent_trajectory_detect_analysis import (
    analyze_latent_trajectory_detect,
)
from src.mech.endpoint_correctness_geometry import (
    answer_margin,
    apply_logistic,
    first_token_correct,
    fit_logistic_checked,
    readout_matrix,
    roc_auc,
)
from src.mech.latent_trajectory_detect import (
    LATENT_TRAJECTORY_CONTRACT,
    LATENT_TRAJECTORY_SCHEMA_VERSION,
    TRAJECTORY_STATES,
    fit_one_hot_ridge,
)
from src.utils.config import load_config


def load_trajectory_export(path: Path, cache: dict) -> dict:
    export = torch.load(path, map_location="cpu", weights_only=False)
    if export.get("contract") != LATENT_TRAJECTORY_CONTRACT:
        raise RuntimeError("trajectory export belongs to another contract")
    if export.get("source_request_sha256") != cache.get("request_sha256"):
        raise RuntimeError(
            "trajectory export was not collected from the attached colon-state cache"
        )
    parity = export["parity_gate"]
    if not (
        parity["passed"]
        and parity["analytic_parity"]["passed"]
        and parity["accuracy_gate"]["passed"]
    ):
        raise RuntimeError("the collection parity gates did not pass")
    states = export["trajectory_states"]
    if states.ndim != 4 or states.shape[2] != TRAJECTORY_STATES:
        raise RuntimeError(f"unexpected trajectory shape {tuple(states.shape)}")
    if export["endpoint_states"].shape != (states.shape[0], states.shape[3]):
        raise RuntimeError("live endpoint states do not pair with the trajectory")
    return export


def _cell_name(position: int, state: int) -> str:
    return f"position_{position}_state_{state:02d}"


def _fit_checked_grid(features, labels, *, ridge_grid, settings):
    candidates = []
    for ridge in ridge_grid:
        weight, bias, stats = fit_logistic_checked(
            features["fit"],
            labels["fit"],
            l2=float(ridge),
            max_iterations=int(settings.solver_max_iterations),
            gradient_tolerance=float(settings.solver_gradient_tolerance),
            objective_gap_tolerance=float(settings.solver_objective_gap_tolerance),
        )
        if not stats["optimization"]["converged"]:
            raise RuntimeError(f"checked logistic ridge={ridge:g} did not converge")
        select_scores = apply_logistic(features["select"], weight, bias, stats)
        candidates.append(
            (
                roc_auc(select_scores, labels["select"]),
                float(ridge),
                weight,
                bias,
                stats,
            )
        )
    return max(candidates, key=lambda item: (item[0], -item[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    settings = cfg.latent_trajectory_detect
    cache, readout_payload = load_margin_cache(args.states, args.readout)
    readout = readout_matrix(readout_payload).double()
    export = load_trajectory_export(args.trajectory, cache)
    # Labels, margins, and the endpoint baseline come from the *live* forced-cue
    # state captured in the same pass as the trajectory: the cache's exact states
    # predate the environment pins and are not reproducible vector-for-vector, so
    # the cache supplies only data (questions, gold tokens) and the partition.
    colon_states = export["endpoint_states"].double()
    gold = cache["evaluation_gold_first_token"].long()
    expected = int(settings.expected_examples)
    trajectory = export["trajectory_states"].double()
    if trajectory.shape[0] != expected or colon_states.shape[0] != expected:
        raise RuntimeError("population size drifted from the frozen contract")
    latent_positions = trajectory.shape[1]

    correct = first_token_correct(colon_states, readout, gold)
    margin = answer_margin(colon_states, readout)
    indices = export["indices"]
    splits = {
        name: torch.tensor(indices[name], dtype=torch.long)
        for name in ("fit", "select", "test")
    }
    labels = {name: correct[index] for name, index in splits.items()}
    margins = {name: margin[index].unsqueeze(1) for name, index in splits.items()}
    if not bool((~labels["select"]).any()) or not bool((~labels["test"]).any()):
        raise RuntimeError("selection and test splits must contain wrong answers")

    # ---- correctness track ---------------------------------------------------
    ridge_grid = [float(value) for value in settings.correctness_ridge_grid]
    margin_features = {name: margins[name] for name in splits}
    (
        margin_select_auc,
        margin_ridge,
        margin_weight,
        margin_bias,
        margin_stats,
    ) = _fit_checked_grid(margin_features, labels, ridge_grid=ridge_grid, settings=settings)

    correctness_curve = []
    best = None
    cells = [
        (position, state)
        for position in range(latent_positions)
        for state in range(TRAJECTORY_STATES)
    ]
    for position, state in tqdm(cells, desc="correctness cells"):
        cell_features = {
            name: torch.cat(
                [trajectory[index, position, state, :], margins[name]], dim=1
            )
            for name, index in splits.items()
        }
        select_auc, ridge, weight, bias, stats = _fit_checked_grid(
            cell_features, labels, ridge_grid=ridge_grid, settings=settings
        )
        entry = {
            "cell": _cell_name(position, state),
            "position": position,
            "state": state,
            "ridge": ridge,
            "select_auc": float(select_auc),
        }
        correctness_curve.append(entry)
        key = (float(select_auc), -ridge, -position, -state)
        if best is None or key > best[0]:
            best = (key, entry, weight, bias, stats, cell_features)
    _, chosen_entry, chosen_weight, chosen_bias, chosen_stats, chosen_features = best
    trajectory_scores = apply_logistic(
        chosen_features["test"], chosen_weight, chosen_bias, chosen_stats
    )
    margin_scores = apply_logistic(
        margin_features["test"], margin_weight, margin_bias, margin_stats
    )
    correctness = {
        "selected": {
            **chosen_entry,
            "test_auc": roc_auc(trajectory_scores, labels["test"]),
            "optimization": chosen_stats["optimization"],
        },
        "margin_baseline": {
            "ridge": margin_ridge,
            "select_auc": float(margin_select_auc),
            "test_auc": roc_auc(margin_scores, labels["test"]),
            "optimization": margin_stats["optimization"],
        },
        "selection_curve": correctness_curve,
    }

    # ---- answer-identity track ------------------------------------------------
    identity_grid = [float(value) for value in settings.identity_ridge_grid]
    fit_gold = gold[splits["fit"]]
    select_gold = gold[splits["select"]]
    test_gold = gold[splits["test"]]
    select_wrong = ~labels["select"]
    test_wrong = ~labels["test"]

    def identity_select_accuracy(model_features_select, probe):
        predictions = probe.predict(model_features_select)
        return float((predictions[select_wrong] == select_gold[select_wrong]).double().mean())

    def fit_identity_grid(fit_features, select_features):
        candidates = []
        for ridge in identity_grid:
            probe = fit_one_hot_ridge(fit_features, fit_gold, ridge=ridge)
            candidates.append(
                (identity_select_accuracy(select_features, probe), ridge, probe)
            )
        return max(candidates, key=lambda item: (item[0], -item[1]))

    identity_curve = []
    best_identity = None
    for position, state in tqdm(cells, desc="identity cells"):
        accuracy, ridge, probe = fit_identity_grid(
            trajectory[splits["fit"], position, state, :],
            trajectory[splits["select"], position, state, :],
        )
        entry = {
            "cell": _cell_name(position, state),
            "position": position,
            "state": state,
            "ridge": ridge,
            "select_wrong_accuracy": accuracy,
        }
        identity_curve.append(entry)
        key = (accuracy, -ridge, -position, -state)
        if best_identity is None or key > best_identity[0]:
            best_identity = (key, entry, probe)
    _, identity_entry, identity_probe = best_identity
    endpoint_accuracy, endpoint_ridge, endpoint_probe = fit_identity_grid(
        colon_states[splits["fit"]], colon_states[splits["select"]]
    )
    majority_class = fit_gold.mode().values
    trajectory_predictions = identity_probe.predict(
        trajectory[
            splits["test"], identity_entry["position"], identity_entry["state"], :
        ]
    )
    endpoint_predictions = endpoint_probe.predict(colon_states[splits["test"]])
    majority_predictions = torch.full_like(test_gold, int(majority_class))
    fit_classes = set(identity_probe.classes.tolist())
    unseen = [int(value) not in fit_classes for value in test_gold.tolist()]
    answer_identity = {
        "selected": identity_entry,
        "endpoint_baseline": {
            "ridge": endpoint_ridge,
            "select_wrong_accuracy": endpoint_accuracy,
        },
        "majority_class_token": int(majority_class),
        "class_count": int(identity_probe.classes.numel()),
        "unseen_gold_class_fraction": float(sum(unseen) / len(unseen)),
        "selection_curve": identity_curve,
    }

    summary = {
        "schema_version": LATENT_TRAJECTORY_SCHEMA_VERSION,
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_request_sha256": export.get("source_request_sha256"),
        "trajectory_request_sha256": export.get("request_sha256"),
        "partition_sha256": export["partition_sha256"],
        "parity_gate": export["parity_gate"],
        "splits": {
            "fit": int(splits["fit"].numel()),
            "select": int(splits["select"].numel()),
            "test": int(splits["test"].numel()),
            "split_seed": int(settings.split_seed),
            "partition_sha256": export["partition_sha256"],
            "correct_share": {
                name: float(labels[name].double().mean()) for name in splits
            },
        },
        "correctness": correctness,
        "answer_identity": answer_identity,
    }
    artifact = {
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "partition_sha256": export["partition_sha256"],
        "test_correct": labels["test"].cpu(),
        "test_wrong_mask": test_wrong.cpu(),
        "test_gold_first_token": test_gold.cpu(),
        "correctness_scores": {
            "trajectory_plus_margin": trajectory_scores.cpu(),
            "margin": margin_scores.cpu(),
        },
        "identity_predictions": {
            "trajectory": trajectory_predictions.cpu(),
            "endpoint": endpoint_predictions.cpu(),
            "majority": majority_predictions.cpu(),
        },
        "identity_probes": {
            "trajectory": identity_probe.state_dict(),
            "endpoint": endpoint_probe.state_dict(),
        },
    }
    artifact_path = args.artifact_output or args.output.with_suffix(".pt")
    _atomic_torch_save(artifact, artifact_path)
    _atomic_json(summary, args.output)
    report = analyze_latent_trajectory_detect(summary, artifact, settings)
    report_path = args.report_output or args.output.with_name(
        "latent_trajectory_detect_report.json"
    )
    _atomic_json(report, report_path)
    print(
        f"[complete] status={report['status']} "
        f"correctness_delta={report['correctness_gate']['delta_auc']:+.4f} "
        f"identity_gain={report['answer_identity_gate']['gain_over_strongest_baseline_points']:+.2f}pts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
