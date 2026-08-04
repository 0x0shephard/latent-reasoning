"""Fit parameter-gradient-aware residual PCs at CODI's answer-cue endpoint."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import replace
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
from scripts.collect_official_codi_endpoint_answer_conditioned import (
    LEGACY_EXCLUSION,
    sample_fresh_answer_conditioned_partitions,
)
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
    GPT2_HIDDEN_SIZE,
    GPT2_STATE_COUNT,
    answer_alignment_moments_from_state,
    answer_alignment_moments_state,
    create_answer_alignment_moments,
    fit_residual_eigenbasis,
    update_answer_alignment_moments,
)
from src.mech.endpoint_parameter_aware import (
    PARAMETER_AWARE_CANDIDATE_STATES,
    PARAMETER_AWARE_SCHEMA_VERSION,
    fit_parameter_aware_bases,
    parameter_aware_bases_from_state,
    parameter_aware_bases_to_state,
    parameter_gradient_cosines,
    residual_pc_candidate_losses,
    validate_parameter_aware_bases,
)
from src.mech.endpoint_tsvc import (
    create_endpoint_moments,
    endpoint_moments_from_state,
    endpoint_moments_state,
    update_endpoint_moments,
)
from src.mech.kv_subspace import deterministic_derangement
from src.mech.kv_target_utility import autograd_gradients
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
PREVIOUS_ANSWER_CONDITIONED_EXCLUSION = {
    "residual_fit_examples": 1024,
    "direction_selection_examples": 1024,
    "update_examples": 256,
    "validation_examples": 256,
    "seed": 29,
    "contract": "answer_conditioned_colon_block_states_v1",
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


@contextmanager
def _math_sdpa_context(device: torch.device):
    """Use the CUDA attention backend whose backward supports double backward."""
    if device.type != "cuda":
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield
    else:
        with sdpa_kernel(SDPBackend.MATH):
            yield


def _dense_candidate_scores(
    values: torch.Tensor, identities: torch.Tensor
) -> torch.Tensor:
    """Scatter candidate scores without changing their statistical precision."""
    if values.ndim != 1 or identities.shape != (values.numel(), 2):
        raise ValueError("candidate values and identities do not align")
    resolved_identities = identities.to(device=values.device, dtype=torch.long)
    result = torch.zeros(
        1,
        GPT2_STATE_COUNT,
        GPT2_HIDDEN_SIZE,
        dtype=values.dtype,
        device=values.device,
    )
    result[0, resolved_identities[:, 0], resolved_identities[:, 1]] = values
    return result


def sample_fresh_parameter_aware_partitions(
    dataset,
    *,
    residual_fit_examples: int,
    direction_selection_examples: int,
    update_examples: int,
    validation_examples: int,
    seed: int,
) -> tuple[dict[str, list[int]], dict]:
    """Exclude both completed endpoint experiments, then sample four new roles."""
    counts = (
        residual_fit_examples,
        direction_selection_examples,
        update_examples,
        validation_examples,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("all parameter-aware partition sizes must be positive")
    legacy, legacy_metadata = sample_endpoint_tsvc_partitions(
        dataset,
        calibration_examples=LEGACY_EXCLUSION["calibration_examples"],
        update_examples=LEGACY_EXCLUSION["update_examples"],
        validation_examples=LEGACY_EXCLUSION["validation_examples"],
        seed=LEGACY_EXCLUSION["seed"],
    )
    previous, previous_metadata = sample_fresh_answer_conditioned_partitions(
        dataset,
        residual_fit_examples=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION[
            "residual_fit_examples"
        ],
        direction_selection_examples=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION[
            "direction_selection_examples"
        ],
        update_examples=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION["update_examples"],
        validation_examples=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION[
            "validation_examples"
        ],
        seed=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION["seed"],
    )
    old_indices = [
        int(index)
        for partitions in (legacy, previous)
        for values in partitions.values()
        for index in values
    ]
    excluded_questions = {
        _normalized_question(dataset[index]["question"]) for index in old_indices
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
        raise ValueError(f"need {required} fresh question groups, found {len(groups)}")
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
            _normalized_question(dataset[index]["question"]) for index in values
        }
        for name, values in partitions.items()
    }
    names = tuple(partitions)
    for left_index, left in enumerate(names):
        if question_sets[left] & excluded_questions:
            raise RuntimeError("parameter-aware partition overlaps a completed run")
        for right in names[left_index + 1 :]:
            if question_sets[left] & question_sets[right]:
                raise RuntimeError("parameter-aware partitions overlap by question")
    return partitions, {
        "selection": "fresh_after_seed11_and_seed29_normalized_question_disjoint_v1",
        "eligible_fresh_question_groups": len(groups),
        "excluded_unique_questions": len(excluded_questions),
        "legacy_exclusion": {
            **LEGACY_EXCLUSION,
            "partition_sha256": legacy_metadata["partition_sha256"],
        },
        "answer_conditioned_exclusion": {
            **PREVIOUS_ANSWER_CONDITIONED_EXCLUSION,
            "partition_sha256": previous_metadata["partition_sha256"],
        },
        "excluded_indices_sha256": _sha256_json(sorted(old_indices)),
        "partition_sha256": {
            name: _sha256_json(values) for name, values in partitions.items()
        },
    }


def _shuffled_answer_batch(batch, permutation: torch.Tensor):
    teacher_fields = (
        "teacher_ids",
        "teacher_mask",
        "teacher_trace_start",
        "teacher_trace_end",
        "teacher_answer_start",
        "teacher_endpoint",
    )
    return replace(
        batch,
        **{
            name: getattr(batch, name).index_select(0, permutation)
            for name in teacher_fields
        },
    )


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
            "selection_indices": partitions["direction_selection"][
                :selection_processed
            ],
            "residual_moments": endpoint_moments_state(residual_moments),
            "alignment_moments": answer_alignment_moments_state(alignment_moments),
        },
        path,
    )


def _selection_summary(bases) -> dict:
    energy = bases.eigenvalues.double()
    totals = energy.sum(dim=-1).clamp_min(1e-30)
    indices_by_state = []
    selected_energy = []
    minimum_z = []
    minimum_cosine = []
    for state, rank_value in enumerate(bases.ranks.tolist()):
        rank = int(rank_value)
        indices = [
            int(value) for value in bases.selected_pc_indices[state, :rank].tolist()
        ]
        indices_by_state.append(indices)
        selected_energy.append(
            float(energy[state, indices].sum() / totals[state]) if indices else 0.0
        )
        if indices:
            minimum_z.append(float(bases.split_z_scores[:, state, indices].min()))
            minimum_cosine.append(
                float(bases.split_cosine_means[:, state, indices].min())
            )
        else:
            minimum_z.append(None)
            minimum_cosine.append(None)
    return {
        "status": (
            "candidate_selected"
            if bases.total_rank > 0
            else "no_stable_positive_parameter_cosines"
        ),
        "scope": "final_two_transformer_block_states_only",
        "candidate_states": list(bases.candidate_states),
        "candidate_pc_count": bases.candidate_pc_count,
        "hutchinson_probes": bases.hutchinson_probes,
        "rank_by_state": [int(value) for value in bases.ranks.tolist()],
        "active_states": list(bases.active_states),
        "total_rank": bases.total_rank,
        "selected_pc_indices_by_state": indices_by_state,
        "selected_residual_energy_fraction_by_state": selected_energy,
        "minimum_selected_split_z_by_state": minimum_z,
        "minimum_selected_split_cosine_by_state": minimum_cosine,
        "minimum_split_z": bases.minimum_split_z,
        "selection_fdr": bases.selection_fdr,
        "maximum_rank_per_state": bases.maximum_rank_per_state,
        "selection_unit": "disjoint_minibatch_parameter_gradient",
    }


def collect(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    if args.precision.casefold() not in {"float32", "fp32"}:
        raise ValueError("parameter-aware double-backward selection requires float32")
    if tuple(args.candidate_states) != PARAMETER_AWARE_CANDIDATE_STATES:
        raise ValueError("the registered contract requires candidate states 11 and 12")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("parameter-aware endpoint collection requires CUDA")
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

    data_cfg = load_config(str(cfg.endpoint_parameter_aware.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    partitions, sampling = sample_fresh_parameter_aware_partitions(
        dataset,
        residual_fit_examples=args.residual_fit_examples,
        direction_selection_examples=args.direction_selection_examples,
        update_examples=args.update_examples,
        validation_examples=args.validation_examples,
        seed=args.sampling_seed,
    )
    metadata = {
        "analysis": "official_codi_endpoint_parameter_aware_collection",
        "contract": "parameter_aware_colon_final_two_blocks_v1",
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
        "candidate_states": list(args.candidate_states),
        "candidate_pc_count": args.candidate_pc_count,
        "hutchinson_probes": args.hutchinson_probes,
        "minimum_split_z": args.minimum_split_z,
        "selection_fdr": args.selection_fdr,
        "maximum_rank_per_state": args.maximum_rank_per_state,
        "hidden_states": GPT2_STATE_COUNT,
        "hidden_size": GPT2_HIDDEN_SIZE,
        "precision": args.precision,
        "selection_attention_backend": "math_sdpa_on_cuda_default_on_cpu",
        "endpoint": "teacher and student answer-cue colon after six student latents and EOT",
        "selection_score": (
            "Hutchinson-normalized cosine between each rank-one residual-PC target "
            "gradient and the gold-answer LoRA-parameter gradient"
        ),
        "selection_boundary": (
            "positive mean, z threshold, and independent BH-FDR survival in both "
            "even/odd minibatch halves"
        ),
    }
    identity_keys = (
        "contract",
        "checkpoint_sha256",
        "official_source_revision",
        "dataset_fingerprint",
        "partitions",
        "fit_batch_size",
        "selection_batch_size",
        "sampling_seed",
        "random_basis_seed",
        "candidate_states",
        "candidate_pc_count",
        "hutchinson_probes",
        "minimum_split_z",
        "selection_fdr",
        "maximum_rank_per_state",
        "precision",
    )
    request_sha256 = _sha256_json({key: metadata[key] for key in identity_keys})
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
    if not parameters:
        raise RuntimeError("official CODI exposes no trainable parameters")
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
            raise RuntimeError("existing parameter-aware basis identity differs")
        validate_parameter_aware_bases(
            parameter_aware_bases_from_state(payload["bases"]),
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
            raise RuntimeError("parameter-aware resume request differs")
        fit_processed = int(state["fit_processed"])
        selection_processed = int(state["selection_processed"])
        if partitions["residual_fit"][:fit_processed] != state["fit_indices"]:
            raise RuntimeError("residual-fit resume prefix changed")
        if (
            partitions["direction_selection"][:selection_processed]
            != state["selection_indices"]
        ):
            raise RuntimeError("direction-selection resume prefix changed")
        residual_moments = endpoint_moments_from_state(state["residual_moments"])
        alignment_moments = answer_alignment_moments_from_state(
            state["alignment_moments"]
        )
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
        desc="Parameter-aware residual eigensystem",
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
            output = scorer(batch, return_answer_endpoint_hidden=True)
        student = output.student_answer_endpoint_hidden
        if student is None:
            raise RuntimeError("student colon endpoint is missing")
        update_endpoint_moments(residual_moments, student, teacher)
        fit_processed = end
        progress.update(len(indices))
        if (
            fit_processed < len(fit_indices)
            and fit_processed % args.save_every < len(indices)
        ):
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
    eigenvectors_device = eigenvectors.to(device=device, dtype=dtype)
    selection_indices = partitions["direction_selection"]
    progress = tqdm(
        total=len(selection_indices),
        initial=selection_processed,
        unit="examples",
        desc="Parameter-gradient PC scoring",
    )
    while selection_processed < len(selection_indices):
        end = selection_processed + args.selection_batch_size
        indices = selection_indices[selection_processed:end]
        batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in indices],
            bot_token_id=model.bot_id,
        ).to(device)
        teacher = extract_official_teacher_endpoint_targets(model, batch).all_hidden
        generator = torch.Generator(device="cpu").manual_seed(
            args.sampling_seed * 1_000_003 + selection_processed * 10_007 + 41
        )
        permutation = deterministic_derangement(
            len(indices), generator=generator, device=device
        )
        shuffled_batch = _shuffled_answer_batch(batch, permutation)
        # Efficient/flash SDPA backward does not implement the mixed derivative used
        # by ``parameter_gradient_cosines``.  Dispatch these graph-producing forwards
        # through math SDPA; the residual fit and utility retain their normal backend.
        with _math_sdpa_context(device), _amp_context(device, args.precision):
            output = scorer(batch, return_answer_endpoint_hidden=True)
            shuffled_output = scorer(shuffled_batch)
        student = output.student_answer_endpoint_hidden
        if student is None:
            raise RuntimeError("student endpoint is missing during parameter selection")
        answer_gradients = autograd_gradients(
            output.mean_loss, parameters, retain_graph=True
        )
        shuffled_answer_gradients = autograd_gradients(
            shuffled_output.mean_loss, parameters, retain_graph=False
        )
        candidate_losses, identities = residual_pc_candidate_losses(
            student,
            teacher,
            eigenvectors_device,
            candidate_states=args.candidate_states,
            candidate_pc_count=args.candidate_pc_count,
        )
        batch_index = selection_processed // args.selection_batch_size
        scores = parameter_gradient_cosines(
            candidate_losses,
            parameters,
            {
                "answer": answer_gradients,
                "shuffled_answer": shuffled_answer_gradients,
            },
            hutchinson_probes=args.hutchinson_probes,
            seed=args.probe_seed + batch_index,
        )
        actual = _dense_candidate_scores(scores["cosines"]["answer"], identities)
        shuffled = _dense_candidate_scores(
            scores["cosines"]["shuffled_answer"], identities
        )
        split_ids = torch.tensor([batch_index % 2], dtype=torch.long)
        update_answer_alignment_moments(
            alignment_moments, actual, shuffled, split_ids
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
            shuffled_batch,
            teacher,
            output,
            shuffled_output,
            student,
            answer_gradients,
            shuffled_answer_gradients,
            candidate_losses,
            scores,
            actual,
            shuffled,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress.close()

    bases = fit_parameter_aware_bases(
        eigenvalues,
        eigenvectors,
        alignment_moments,
        candidate_states=args.candidate_states,
        candidate_pc_count=args.candidate_pc_count,
        hutchinson_probes=args.hutchinson_probes,
        minimum_split_z=args.minimum_split_z,
        selection_fdr=args.selection_fdr,
        maximum_rank_per_state=args.maximum_rank_per_state,
        random_seed=args.random_basis_seed,
        residual_fit_count=len(fit_indices),
        direction_selection_examples=len(selection_indices),
    )
    validate_parameter_aware_bases(bases, require_candidate=False)
    selection_summary = _selection_summary(bases)
    payload = {
        "schema_version": PARAMETER_AWARE_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "metadata": metadata,
        "selection": selection_summary,
        "bases": parameter_aware_bases_to_state(bases),
    }
    _atomic_torch_save(payload, basis_path)
    _atomic_json(
        {
            **metadata,
            "state": "complete",
            "basis_file": basis_path.name,
            "basis_sha256": sha256_file(basis_path),
            "selection": selection_summary,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        manifest_path,
    )
    _atomic_json(
        {
            "schema_version": PARAMETER_AWARE_SCHEMA_VERSION,
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
        description="Collect parameter-aware official-CODI endpoint directions."
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
    parser.add_argument("--selection-batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--sampling-seed", type=int, default=41)
    parser.add_argument("--random-basis-seed", type=int, default=20260805)
    parser.add_argument("--probe-seed", type=int, default=314159)
    parser.add_argument("--candidate-states", type=int, nargs="+", default=list(PARAMETER_AWARE_CANDIDATE_STATES))
    parser.add_argument("--candidate-pc-count", type=int, default=64)
    parser.add_argument("--hutchinson-probes", type=int, default=8)
    parser.add_argument("--minimum-split-z", type=float, default=1.645)
    parser.add_argument("--selection-fdr", type=float, default=0.05)
    parser.add_argument("--maximum-rank-per-state", type=int, default=8)
    parser.add_argument("--parity-examples", type=int, default=4)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    args.candidate_states = tuple(args.candidate_states)
    if args.selection_batch_size < 2:
        parser.error("selection batch size must be at least two")
    if args.direction_selection_examples % args.selection_batch_size:
        parser.error("direction-selection examples must be divisible by its batch size")
    if args.direction_selection_examples // args.selection_batch_size < 4:
        parser.error("each selection half requires at least two minibatches")
    if args.save_every % args.selection_batch_size:
        parser.error("save-every must be divisible by the selection batch size")
    if not 0 < args.selection_fdr <= 1:
        parser.error("selection FDR must lie in (0, 1]")
    if not 0 < args.maximum_rank_per_state <= args.candidate_pc_count:
        parser.error("maximum rank must not exceed the candidate PC count")
    if not 0 < args.parity_examples <= args.residual_fit_examples:
        parser.error("parity examples must lie inside the residual-fit partition")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
