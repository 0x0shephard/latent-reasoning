"""Compatibility layer for the author-released CODI GPT-2 checkpoint.

This module intentionally mirrors the public CODI evaluation architecture instead of
loading the checkpoint into :class:`src.models.latent_lm.LatentCausalLM`.  The two models
use different special-token layouts, projection modules, LoRA structure, prompts, and
generation paths; treating them as interchangeable would make an accuracy comparison
invalid.

Reference implementation:
    https://github.com/zhenyi4/codi
Reference source revision:
    2c2314662c63e9f482ebc46614ffe9af17a241e5
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OfficialCODILoadReport:
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_tensors: int
    matched_tensors: int
    matched_numel_fraction: float
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_torch_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.casefold()
    if normalized == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32
    values = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in values:
        raise ValueError(f"unsupported dtype {name!r}")
    dtype = values[normalized]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 is not supported for the official CODI CPU path")
    if (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("this CUDA device does not support bfloat16; use dtype=auto")
    return dtype


class OfficialCODIGPT2(nn.Module):
    """The exact module topology used by the released GPT-2 checkpoint."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        lora_rank: int = 128,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        projection_dim: int = 768,
        projection_dropout: float = 0.0,
        projection_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model

        self.codi = backbone
        original_vocab_size = int(self.codi.config.vocab_size)
        self.pad_token_id = original_vocab_size
        self.bot_id = original_vocab_size + 1
        self.eot_id = original_vocab_size + 2
        self.codi.resize_token_embeddings(original_vocab_size + 3)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=["c_attn", "c_proj", "c_fc"],
            init_lora_weights=True,
        )
        self.codi = get_peft_model(self.codi, lora_config)

        hidden_size = int(self.codi.config.hidden_size)
        self.prj = nn.Sequential(
            nn.Dropout(float(projection_dropout)),
            nn.Linear(hidden_size, int(projection_dim)),
            nn.GELU(),
            nn.Linear(int(projection_dim), hidden_size),
        )
        if projection_layer_norm:
            # The official implementation adds this named module after constructing the
            # Sequential. Keeping the name ``ln`` preserves checkpoint keys exactly.
            self.prj.add_module("ln", nn.LayerNorm(hidden_size))

    @property
    def config(self):
        return self.codi.config

    def input_embeddings(self) -> nn.Module:
        base = self.codi.get_base_model()
        if not hasattr(base, "transformer") or not hasattr(base.transformer, "wte"):
            raise TypeError("official CODI GPT-2 requires transformer.wte embeddings")
        return base.transformer.wte

    def tie_weights(self) -> None:
        self.codi.tie_weights()


def build_official_codi_gpt2(
    *,
    base_model: str,
    base_revision: str,
    dtype: torch.dtype,
    settings: dict,
    token: str | None = None,
) -> tuple[OfficialCODIGPT2, object]:
    """Construct the released architecture before applying its state dictionary."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pretrained_kwargs = {
        "revision": base_revision,
        "torch_dtype": dtype,
    }
    if token:
        pretrained_kwargs["token"] = token
    backbone = AutoModelForCausalLM.from_pretrained(base_model, **pretrained_kwargs)
    model = OfficialCODIGPT2(
        backbone,
        lora_rank=int(settings["lora_rank"]),
        lora_alpha=int(settings["lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
        projection_dim=int(settings["projection_dim"]),
        projection_dropout=float(settings["projection_dropout"]),
        projection_layer_norm=bool(settings["projection_layer_norm"]),
    )

    tokenizer_kwargs = {
        "revision": base_revision,
        "model_max_length": int(settings["model_max_length"]),
        "padding_side": "left",
        "use_fast": False,
    }
    if token:
        tokenizer_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token_id != model.pad_token_id:
        raise RuntimeError(
            "official tokenizer/model special-token contract changed: "
            f"pad={tokenizer.pad_token_id}, expected={model.pad_token_id}"
        )
    return model, tokenizer


def download_official_checkpoint(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    token: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            token=token,
        )
    )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return path


def load_official_checkpoint(
    model: OfficialCODIGPT2,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None,
    minimum_matched_numel_fraction: float = 0.95,
) -> OfficialCODILoadReport:
    """Load safely and reject silent architecture/checkpoint incompatibility."""
    path = Path(checkpoint_path)
    observed_sha = sha256_file(path)
    if expected_sha256 and observed_sha != expected_sha256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: expected {expected_sha256}, "
            f"observed {observed_sha}"
        )

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise TypeError("official checkpoint must contain a non-empty tensor state dict")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("official checkpoint state dict contains non-tensor values")

    target = model.state_dict()
    shape_mismatches = [
        key
        for key, value in state.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    ]
    if shape_mismatches:
        preview = ", ".join(shape_mismatches[:5])
        raise RuntimeError(f"official checkpoint has shape mismatches: {preview}")

    matched = {
        key: value
        for key, value in state.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    checkpoint_numel = sum(int(value.numel()) for value in state.values())
    matched_numel = sum(int(value.numel()) for value in matched.values())
    matched_fraction = matched_numel / max(1, checkpoint_numel)
    if matched_fraction < minimum_matched_numel_fraction:
        unexpected = [key for key in state if key not in matched]
        raise RuntimeError(
            "official CODI checkpoint is incompatible with the adapter: "
            f"matched {matched_fraction:.2%} of checkpoint parameters; "
            f"sample unmatched keys={unexpected[:5]}"
        )

    required_fragments = ("prj.1.weight", "prj.3.weight", "lora_A", "lora_B", "wte.weight")
    absent = [
        fragment
        for fragment in required_fragments
        if not any(fragment in key for key in matched)
    ]
    if absent:
        raise RuntimeError(
            "official checkpoint did not load required components: " + ", ".join(absent)
        )

    incompatible = model.load_state_dict(state, strict=False)
    model.tie_weights()
    return OfficialCODILoadReport(
        checkpoint_path=str(path),
        checkpoint_sha256=observed_sha,
        checkpoint_tensors=len(state),
        matched_tensors=len(matched),
        matched_numel_fraction=matched_fraction,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )


def _normalized_official_questions(questions: Iterable[str]) -> list[str]:
    # Match the released test.py formatting exactly.
    return [str(question).strip().replace("  ", " ") for question in questions]


def _new_answer_endpoint_mask(
    generated: list[list[int]],
    cue_ids: list[int],
    already_applied: torch.Tensor,
) -> torch.Tensor:
    """Rows whose current input is the first exact generated cue-final token."""
    applied = already_applied.detach().cpu().tolist()
    return torch.tensor(
        [
            (not bool(applied[row]))
            and len(token_ids) >= len(cue_ids)
            and token_ids[-len(cue_ids) :] == cue_ids
            for row, token_ids in enumerate(generated)
        ],
        dtype=torch.bool,
        device=already_applied.device,
    )


@torch.inference_mode()
def generate_official_codi(
    model: OfficialCODIGPT2,
    tokenizer,
    questions: list[str],
    *,
    latent_iterations: int,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    kv_intervention=None,
    answer_endpoint_intervention=None,
    answer_cue: str = "The answer is:",
    force_answer_cue: bool = False,
    return_endpoint_metadata: bool = False,
) -> list[str] | tuple[list[str], dict]:
    """Greedy generation matching the released path, with optional causal KV edits."""
    if latent_iterations <= 0:
        raise ValueError("latent_iterations must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model.eval()
    outputs: list[str] = []
    endpoint_reached: list[bool] = []
    embedding = model.input_embeddings()
    normalized = _normalized_official_questions(questions)
    cue_ids = list(
        tokenizer(f" {answer_cue}", add_special_tokens=False)["input_ids"]
    )
    if not cue_ids:
        raise ValueError("answer cue must tokenize to at least one token")

    for start in range(0, len(normalized), batch_size):
        chunk = normalized[start : start + batch_size]
        batch = tokenizer(
            chunk,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to(device)
        bot = torch.full(
            (len(chunk), 1),
            model.bot_id,
            dtype=torch.long,
            device=device,
        )
        input_ids = torch.cat((batch["input_ids"], bot), dim=1)
        attention_mask = torch.cat(
            (batch["attention_mask"], torch.ones_like(bot)), dim=1
        )

        encoded = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = encoded.past_key_values
        latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
        for latent_position in range(latent_iterations):
            latent_output = model.codi(
                inputs_embeds=latent,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = latent_output.past_key_values
            if kv_intervention is not None:
                cache = kv_intervention(cache, latent_position)
            latent = model.prj(
                latent_output.hidden_states[-1][:, -1, :].unsqueeze(1)
            )

        finished = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        endpoint_applied = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in chunk]
        if force_answer_cue:
            forced = torch.tensor(
                [model.eot_id, *cue_ids], dtype=torch.long, device=device
            ).unsqueeze(0).expand(len(chunk), -1)
            endpoint_mask = torch.ones(len(chunk), dtype=torch.bool, device=device)
            context = (
                answer_endpoint_intervention.activate(endpoint_mask)
                if answer_endpoint_intervention is not None
                else nullcontext()
            )
            with context:
                decoded = model.codi(
                    inputs_embeds=embedding(forced),
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            cache = decoded.past_key_values
            endpoint_applied |= endpoint_mask
            next_token = decoded.logits[:, -1, : model.eot_id].argmax(dim=-1)
            for row, token_id in enumerate(next_token.tolist()):
                generated[row].append(int(token_id))
                if token_id == tokenizer.eos_token_id:
                    finished[row] = True
            token_embedding = embedding(next_token).unsqueeze(1)
            remaining_steps = range(1, max_new_tokens)
        else:
            eot_ids = torch.full(
                (len(chunk),),
                model.eot_id,
                dtype=torch.long,
                device=device,
            )
            token_embedding = embedding(eot_ids).unsqueeze(1)
            remaining_steps = range(max_new_tokens)

        for _ in remaining_steps:
            if bool(finished.all()):
                break
            if force_answer_cue:
                endpoint_mask = torch.zeros(
                    len(chunk), dtype=torch.bool, device=device
                )
            else:
                endpoint_mask = _new_answer_endpoint_mask(
                    generated, cue_ids, endpoint_applied
                )
            endpoint_applied |= endpoint_mask
            context = (
                answer_endpoint_intervention.activate(endpoint_mask)
                if answer_endpoint_intervention is not None and bool(endpoint_mask.any())
                else nullcontext()
            )
            with context:
                decoded = model.codi(
                    inputs_embeds=token_embedding,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            cache = decoded.past_key_values
            # The released script excludes only the final synthetic EOT id.
            next_token = decoded.logits[:, -1, : model.eot_id].argmax(dim=-1)
            for row, token_id in enumerate(next_token.tolist()):
                if not finished[row]:
                    generated[row].append(int(token_id))
                    if token_id == tokenizer.eos_token_id:
                        finished[row] = True
            token_embedding = embedding(next_token).unsqueeze(1)

        outputs.extend(
            tokenizer.decode(token_ids, skip_special_tokens=True)
            for token_ids in generated
        )
        endpoint_reached.extend(bool(value) for value in endpoint_applied.tolist())

    if len(outputs) != len(questions):
        raise RuntimeError("official CODI generation count mismatch")
    if return_endpoint_metadata:
        return outputs, {
            "answer_cue": answer_cue,
            "answer_cue_token_ids": cue_ids,
            "answer_cue_forced": bool(force_answer_cue),
            "endpoint_reached": endpoint_reached,
            "endpoint_reached_count": int(sum(endpoint_reached)),
            "endpoint_reached_fraction": float(sum(endpoint_reached) / len(endpoint_reached)),
        }
    return outputs
