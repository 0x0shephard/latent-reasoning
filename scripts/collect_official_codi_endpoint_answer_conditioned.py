"""Fit split-stable answer-conditioned CODI endpoint residual directions."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    sample_endpoint_tsvc_partitions,
    verify_full_reproduction_gate,
)
from scripts.collect_official_codi_endpoint_tsvc_corrected import _run_native_parity_gate
from src.data.datasets import load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
    official_codi_answer_is_eligible,
)
from src.eval.official_codi import select_device
from src.mech.endpoint_answer_conditioned import (
    ANSWER_CONDITIONED_SCHEMA_VERSION,
    GPT2_HIDDEN_SIZE,
    GPT2_STATE_COUNT,
    answer_alignment_moments_from_state,
    answer_alignment_moments_state,
    answer_conditioned_bases_from_state,
    answer_conditioned_bases_to_state,
    create_answer_alignment_moments,
    fit_answer_conditioned_bases,
    fit_residual_eigenbasis,
    update_answer_alignment_moments,
    validate_answer_conditioned_bases,
)
from src.mech.endpoint_tsvc import (
    create_endpoint_moments,
    endpoint_moments_from_state,
    endpoint_moments_state,
    update_endpoint_moments,
)
from src.mech.kv_subspace import deterministic_derangement
from src.mech.official_codi_target_utility import (
    OfficialCODIAnswerScorer,
    extract_official_teacher_endpoint_targets,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


COLLECTION_STATE_SCHEMA_VERSION = 1
LEGACY_EXCLUSION = {
    "calibration_examples": 5000,
    "update_examples": 256,
    "validation_examples": 256,
    "seed": 11,
    "contract": "corrected_rank77_seed11_all_partitions",
    "dataset_fingerprint": "a960dcd3f3d83cae",
    "partition_sha256": {
        "calibration": "2b026f8a0579afd3c9fc61bf1ea9b1beb8f952057d13cd7f7d6b333b15992a43",
        "update": "19522631a7c002354d18baa31ef43d0875e7dd559862fa44d8aa4b8533d400e9",
        "validation": "5e2b39f6c4567271ca5e1c2398dedf3a973b90010141a81a4675d5543f87b844",
    },
}


def _sha256_json(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _amp_context(device: torch.device, precision: str):
    normalized = precision.casefold()
    if normalized in {"float32", "fp32"} or device.type != "cuda":
        return nullcontext()
    dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(normalized)
    if dtype is None:
        raise ValueError("precision must be float32, float16, or bfloat16")
    return torch.autocast(device_type="cuda", dtype=dtype)


def sample_fresh_answer_conditioned_partitions(
    dataset,
    *,
    residual_fit_examples: int,
    direction_selection_examples: int,
    update_examples: int,
    validation_examples: int,
    seed: int,
) -> tuple[dict[str, list[int]], dict]:
    """Sample four question-disjoint partitions after excluding the old seed-11 run."""
    counts = (
        residual_fit_examples,
        direction_selection_examples,
        update_examples,
        validation_examples,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("all answer-conditioned partition sizes must be positive")
    legacy, legacy_metadata = sample_endpoint_tsvc_partitions(
        dataset,
        calibration_examples=LEGACY_EXCLUSION["calibration_examples"],
        update_examples=LEGACY_EXCLUSION["update_examples"],
        validation_examples=LEGACY_EXCLUSION["validation_examples"],
        seed=LEGACY_EXCLUSION["seed"],
    )
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is not None:
        if str(fingerprint) != LEGACY_EXCLUSION["dataset_fingerprint"]:
            raise RuntimeError(
                "dataset fingerprint differs from the completed corrected experiment"
            )
        if legacy_metadata["partition_sha256"] != LEGACY_EXCLUSION["partition_sha256"]:
            raise RuntimeError("failed to reproduce registered seed-11 exclusions")
    excluded_indices = [
        index for name in ("calibration", "update", "validation") for index in legacy[name]
    ]
    excluded_questions = {
        _normalized_question(dataset[index]["question"])
        for index in excluded_indices
    }

    groups: dict[str, list[int]] = {}
    for index, (question, answer) in enumerate(
        zip(dataset["question"], dataset["answer"])
    ):
        normalized = _normalized_question(question)
        if (
            normalized not in excluded_questions
            and official_codi_answer_is_eligible(answer)
        ):
            groups.setdefault(normalized, []).append(index)
    required = sum(counts)
    if len(groups) < required:
        raise ValueError(
            f"need {required} fresh eligible question groups, found {len(groups)}"
        )
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    selected = [generator.choice(groups[key]) for key in keys[:required]]
    fit_end = residual_fit_examples
    selection_end = fit_end + direction_selection_examples
    update_end = selection_end + update_examples
    partitions = {
        "residual_fit": selected[:fit_end],
        "direction_selection": selected[fit_end:selection_end],
        "update": selected[selection_end:update_end],
        "validation": selected[update_end:],
    }
    question_sets = {
        name: {
            _normalized_question(dataset[index]["question"])
            for index in indices
        }
        for name, indices in partitions.items()
    }
    names = tuple(partitions)
    for left_index, left in enumerate(names):
        if question_sets[left] & excluded_questions:
            raise RuntimeError("new answer-conditioned partition overlaps seed-11 data")
        for right in names[left_index + 1 :]:
            if question_sets[left] & question_sets[right]:
                raise RuntimeError("answer-conditioned partitions overlap by question")
    return partitions, {
        "selection": "fresh_four_way_normalized_question_disjoint_v1",
        "eligible_fresh_question_groups": len(groups),
        "legacy_exclusion": {
            **LEGACY_EXCLUSION,
            "excluded_unique_questions": len(excluded_questions),
            "partition_sha256": legacy_metadata["partition_sha256"],
            "excluded_indices_sha256": _sha256_json(excluded_indices),
        },
        "partition_sha256": {
            name: _sha256_json(indices) for name, indices in partitions.items()
        },
    }


def _answer_gradients_at_colon(output, batch) -> torch.Tensor:
    hidden_states = output.student_answer_hidden_states
    if not hidden_states or len(hidden_states) != GPT2_STATE_COUNT:
        raise RuntimeError("answer-gradient collection requires all 13 raw hidden states")
    gradients = torch.autograd.grad(
        output.mean_loss,
        hidden_states,
        retain_graph=False,
        allow_unused=True,
    )
    if any(value is None for value in gradients):
        raise RuntimeError("an answer hidden-state entry is disconnected from answer NLL")
    endpoints = (
        batch.teacher_answer_start - batch.teacher_trace_end
    ).to(device=hidden_states[0].device)
    row = torch.arange(endpoints.shape[0], device=endpoints.device)
    gathered = torch.stack(
        [gradient[row, endpoints, :] for gradient in gradients if gradient is not None],
        dim=1,
    )
    # The differentiated loss is a batch mean. Restore per-example gradient scale so
    # selection statistics are invariant to the selection batch size.
    gathered = gathered * endpoints.shape[0]
    if gathered.shape[1:] != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE):
        raise RuntimeError("gathered answer gradients do not satisfy [B,13,768]")
    if not torch.isfinite(gathered).all():
        raise RuntimeError("answer gradients contain non-finite values")
    return gathered


def _save_state(
    path: Path,
    *,
    request_sha256: str,
    phase: str,
    fit_processed: int,
    selection_processed: int,
    partitions: dict[str, list[int]],
    residual_moments: dict,
    alignment_moments: dict,
) -> None:
    _atomic_torch_save(
        {
            "schema_version": COLLECTION_STATE_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "phase": phase,
            "fit_processed": fit_processed,
            "selection_processed": selection_processed,
            "fit_indices": partitions["residual_fit"][:fit_processed],
            "selection_indices": partitions["direction_selection"][:selection_processed],
            "residual_moments": endpoint_moments_state(residual_moments),
            "alignment_moments": answer_alignment_moments_state(alignment_moments),
        },
        path,
    )


def _selection_summary(bases) -> dict:
    energy = bases.eigenvalues.double()
    total = energy.sum(dim=-1).clamp_min(1e-30)
    selected_energy = []
    selected_indices = []
    minimum_selected_z = []
    for state, rank_value in enumerate(bases.ranks.tolist()):
        rank = int(rank_value)
        indices = [
            int(value)
            for value in bases.selected_pc_indices[state, :rank].tolist()
        ]
        selected_indices.append(indices)
        selected_energy.append(
            float(energy[state, indices].sum() / total[state]) if indices else 0.0
        )
        if indices:
            z = bases.split_z_scores[:, state, indices]
            minimum_selected_z.append(float(z.min()))
        else:
            minimum_selected_z.append(None)
    return {
        "status": (
            "candidate_selected" if bases.total_rank > 0 else "no_stable_positive_directions"
        ),
        "scope": "transformer_block_states_1_through_12_embedding_excluded",
        "rank_by_state": [int(value) for value in bases.ranks.tolist()],
        "active_states": list(bases.active_states),
        "total_rank": bases.total_rank,
        "selected_residual_energy_fraction_by_state": selected_energy,
        "selected_pc_indices_by_state": selected_indices,
        "minimum_selected_split_z_by_state": minimum_selected_z,
        "minimum_split_z": bases.minimum_split_z,
        "selection_fdr": bases.selection_fdr,
        "maximum_rank_per_state": bases.maximum_rank_per_state,
    }


def collect(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("answer-conditioned endpoint collection requires CUDA")
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id),
        revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename),
        expected_sha256=str(cfg.checkpoint.sha256),
        token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model),
        base_revision=str(cfg.model.base_revision),
        dtype=dtype,
        settings=cfg.model,
        token=token,
    )
    load_report = load_official_checkpoint(
        model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
    )
    model.to(device=device, dtype=dtype).eval()
    torch.manual_seed(args.sampling_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.sampling_seed)

    data_cfg = load_config(str(cfg.endpoint_answer_conditioned.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    partitions, sampling = sample_fresh_answer_conditioned_partitions(
        dataset,
        residual_fit_examples=args.residual_fit_examples,
        direction_selection_examples=args.direction_selection_examples,
        update_examples=args.update_examples,
        validation_examples=args.validation_examples,
        seed=args.sampling_seed,
    )
    metadata = {
        "analysis": "official_codi_endpoint_answer_conditioned_collection",
        "contract": "answer_conditioned_colon_block_states_v1",
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", "unavailable"),
        "partitions": partitions,
        "sampling": sampling,
        "residual_fit_examples": len(partitions["residual_fit"]),
        "direction_selection_examples": len(partitions["direction_selection"]),
        "update_examples": len(partitions["update"]),
        "validation_examples": len(partitions["validation"]),
        "fit_batch_size": args.fit_batch_size,
        "selection_batch_size": args.selection_batch_size,
        "sampling_seed": args.sampling_seed,
        "random_basis_seed": args.random_basis_seed,
        "minimum_split_z": args.minimum_split_z,
        "selection_fdr": args.selection_fdr,
        "maximum_rank_per_state": args.maximum_rank_per_state,
        "hidden_states": GPT2_STATE_COUNT,
        "hidden_size": GPT2_HIDDEN_SIZE,
        "precision": args.precision,
        "endpoint": "teacher and student answer-cue colon after six student latents and EOT",
        "selection_scope": "states 1..12; embedding state 0 excluded",
        "selection_score": (
            "per-residual-PC product of student-minus-teacher coefficient and "
            "gold-answer-NLL gradient coefficient"
        ),
        "selection_boundary": (
            "positive mean and z >= threshold independently in deterministic even/odd "
            "selection splits plus independent BH-FDR survival, capped per state"
        ),
    }
    identity = {
        key: metadata[key]
        for key in (
            "contract",
            "checkpoint_sha256",
            "official_source_revision",
            "dataset_fingerprint",
            "partitions",
            "fit_batch_size",
            "selection_batch_size",
            "sampling_seed",
            "random_basis_seed",
            "minimum_split_z",
            "selection_fdr",
            "maximum_rank_per_state",
            "precision",
        )
    }
    request_sha256 = _sha256_json(identity)
    metadata["request_sha256"] = request_sha256
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    state_path = output_dir / "collection_state.pt"
    basis_path = output_dir / "basis.pt"

    scorer = OfficialCODIAnswerScorer(
        model, latent_positions=int(cfg.eval.latent_iterations)
    )
    parameters = [value for value in scorer.parameters() if value.requires_grad]
    parity = _run_native_parity_gate(
        scorer=scorer,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        indices=partitions["residual_fit"][: args.parity_examples],
        parameters=parameters,
        device=device,
        precision=args.precision,
        output_path=output_dir / "native_loss_gradient_parity.json",
    )
    metadata["native_parity_gate"] = parity

    if basis_path.is_file():
        payload = torch.load(basis_path, map_location="cpu", weights_only=False)
        if payload.get("request_sha256") != request_sha256:
            raise RuntimeError("existing answer-conditioned basis identity differs")
        validate_answer_conditioned_bases(
            answer_conditioned_bases_from_state(payload["bases"]),
            require_candidate=False,
        )
        return payload

    _atomic_json(
        {
            **metadata,
            "state": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        manifest_path,
    )
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("request_sha256") != request_sha256:
            raise RuntimeError("answer-conditioned resume request differs")
        fit_processed = int(state["fit_processed"])
        selection_processed = int(state["selection_processed"])
        if partitions["residual_fit"][:fit_processed] != state["fit_indices"]:
            raise RuntimeError("residual-fit resume prefix changed")
        if partitions["direction_selection"][:selection_processed] != state["selection_indices"]:
            raise RuntimeError("direction-selection resume prefix changed")
        residual_moments = endpoint_moments_from_state(state["residual_moments"])
        alignment_moments = answer_alignment_moments_from_state(state["alignment_moments"])
    else:
        fit_processed = 0
        selection_processed = 0
        residual_moments = create_endpoint_moments(
            GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE
        )
        alignment_moments = create_answer_alignment_moments(
            GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE
        )

    fit_indices = partitions["residual_fit"]
    progress = tqdm(
        total=len(fit_indices),
        initial=fit_processed,
        unit="examples",
        desc="Answer-conditioned residual eigensystem",
    )
    while fit_processed < len(fit_indices):
        end = min(fit_processed + args.fit_batch_size, len(fit_indices))
        indices = fit_indices[fit_processed:end]
        batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in indices],
            bot_token_id=model.bot_id,
        ).to(device)
        with torch.no_grad(), _amp_context(device, args.precision):
            teacher = extract_official_teacher_endpoint_targets(model, batch).all_hidden
            output = scorer(
                batch,
                return_answer_endpoint_hidden=True,
            )
        student = output.student_answer_endpoint_hidden
        if student is None:
            raise RuntimeError("student colon endpoint is missing")
        update_endpoint_moments(residual_moments, student, teacher)
        fit_processed = end
        progress.update(len(indices))
        if fit_processed < len(fit_indices) and fit_processed % args.save_every < len(indices):
            _save_state(
                state_path,
                request_sha256=request_sha256,
                phase="residual_fit",
                fit_processed=fit_processed,
                selection_processed=selection_processed,
                partitions=partitions,
                residual_moments=residual_moments,
                alignment_moments=alignment_moments,
            )
        del batch, teacher, student, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress.close()

    eigenvalues, eigenvectors = fit_residual_eigenbasis(residual_moments)
    eigenvectors_device = eigenvectors.to(device=device, dtype=torch.float32)
    selection_indices = partitions["direction_selection"]
    progress = tqdm(
        total=len(selection_indices),
        initial=selection_processed,
        unit="examples",
        desc="Answer-conditioned PC scoring",
    )
    while selection_processed < len(selection_indices):
        end = min(
            selection_processed + args.selection_batch_size,
            len(selection_indices),
        )
        indices = selection_indices[selection_processed:end]
        batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in indices],
            bot_token_id=model.bot_id,
        ).to(device)
        teacher = extract_official_teacher_endpoint_targets(model, batch).all_hidden
        with _amp_context(device, args.precision):
            output = scorer(
                batch,
                return_answer_endpoint_hidden=True,
                return_answer_hidden_states=True,
            )
        student = output.student_answer_endpoint_hidden
        if student is None:
            raise RuntimeError("student answer endpoint is missing during selection")
        answer_gradient = _answer_gradients_at_colon(output, batch)
        residual = (student.detach() - teacher.detach()).float()
        gradient = answer_gradient.detach().float()
        residual_coefficients = torch.einsum(
            "bsd,sdk->bsk", residual, eigenvectors_device
        )
        gradient_coefficients = torch.einsum(
            "bsd,sdk->bsk", gradient, eigenvectors_device
        )
        products = residual_coefficients * gradient_coefficients
        generator = torch.Generator(device="cpu").manual_seed(
            args.sampling_seed * 1_000_003
            + selection_processed * 10_007
            + 202_608_04
        )
        permutation = deterministic_derangement(
            len(indices), generator=generator, device=device
        )
        shuffled_products = residual_coefficients * gradient_coefficients.index_select(
            0, permutation
        )
        split_ids = (
            torch.arange(selection_processed, end, device=device, dtype=torch.long) % 2
        )
        update_answer_alignment_moments(
            alignment_moments,
            products,
            shuffled_products,
            split_ids,
        )
        selection_processed = end
        progress.update(len(indices))
        if (
            selection_processed < len(selection_indices)
            and selection_processed % args.save_every < len(indices)
        ):
            _save_state(
                state_path,
                request_sha256=request_sha256,
                phase="direction_selection",
                fit_processed=fit_processed,
                selection_processed=selection_processed,
                partitions=partitions,
                residual_moments=residual_moments,
                alignment_moments=alignment_moments,
            )
        del (
            batch,
            teacher,
            student,
            output,
            answer_gradient,
            residual,
            gradient,
            residual_coefficients,
            gradient_coefficients,
            products,
            shuffled_products,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress.close()

    bases = fit_answer_conditioned_bases(
        eigenvalues,
        eigenvectors,
        alignment_moments,
        minimum_split_z=args.minimum_split_z,
        selection_fdr=args.selection_fdr,
        maximum_rank_per_state=args.maximum_rank_per_state,
        random_seed=args.random_basis_seed,
        exclude_embedding=True,
        residual_fit_count=len(fit_indices),
    )
    validate_answer_conditioned_bases(bases, require_candidate=False)
    selection_summary = _selection_summary(bases)
    payload = {
        "schema_version": ANSWER_CONDITIONED_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "metadata": metadata,
        "selection": selection_summary,
        "bases": answer_conditioned_bases_to_state(bases),
    }
    _atomic_torch_save(payload, basis_path)
    basis_sha256 = sha256_file(basis_path)
    _atomic_json(
        {
            **metadata,
            "state": "complete",
            "basis_file": basis_path.name,
            "basis_sha256": basis_sha256,
            "selection": selection_summary,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        manifest_path,
    )
    _atomic_json(
        {
            "schema_version": ANSWER_CONDITIONED_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "state": "complete",
            "fit_processed": fit_processed,
            "selection_processed": selection_processed,
            "selection_status": selection_summary["status"],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_dir / "progress.json",
    )
    print(f"[complete] selection={selection_summary['status']}")
    print(f"[complete] rank_by_state={selection_summary['rank_by_state']}")
    print(f"[complete] wrote {basis_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect answer-conditioned official-CODI endpoint directions."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--residual-fit-examples", type=int, default=1024)
    parser.add_argument("--direction-selection-examples", type=int, default=1024)
    parser.add_argument("--update-examples", type=int, default=256)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--fit-batch-size", type=int, default=16)
    parser.add_argument("--selection-batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--sampling-seed", type=int, default=29)
    parser.add_argument("--random-basis-seed", type=int, default=20260804)
    parser.add_argument("--minimum-split-z", type=float, default=1.645)
    parser.add_argument("--selection-fdr", type=float, default=0.05)
    parser.add_argument("--maximum-rank-per-state", type=int, default=64)
    parser.add_argument("--parity-examples", type=int, default=4)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.selection_batch_size < 2:
        parser.error("selection batch size must be at least two for shuffled controls")
    if args.direction_selection_examples < 4:
        parser.error("direction selection requires at least four examples")
    if args.direction_selection_examples % args.selection_batch_size:
        parser.error("direction-selection examples must be divisible by its batch size")
    if not 0 < args.selection_fdr <= 1:
        parser.error("selection FDR must lie in (0, 1]")
    if not 0 < args.parity_examples <= args.residual_fit_examples:
        parser.error("parity examples must lie inside the residual-fit partition")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
