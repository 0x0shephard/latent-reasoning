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
    ignored_prefixes: tuple[str, ...] = ()   # whole modules the architecture does not have (e.g. SIM-CoT's training decoder)
    ignored_tensors: int = 0

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


#: LoRA targets of the released checkpoints, per backbone family (official test.py).
GPT2_LORA_TARGET_MODULES = ("c_attn", "c_proj", "c_fc")
LLAMA_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")


class OfficialCODIGPT2(nn.Module):
    """The exact module topology used by the released checkpoints (GPT-2, and the
    LLaMA family through ``lora_target_modules`` and the embedding accessor)."""

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
        lora_target_modules: tuple[str, ...] = GPT2_LORA_TARGET_MODULES,
        question_special_tokens: bool = False,
        pad_aware_generation: bool = False,
        prompt_logits_last_only: bool = False,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model

        self.codi = backbone
        # The released LLaMA path tokenizes questions with the tokenizer's special
        # tokens (a BOS prefix); the GPT-2 path adds none.  Read by the generator.
        self.question_special_tokens = bool(question_special_tokens)
        # The released path drops the attention mask after the question, so with left
        # padding the pad positions become visible to the thoughts and the answer and
        # rotary positions shift.  GPT-2 tolerates this (and the released GPT-2 numbers
        # were produced that way); LLaMA does not.  With this flag the mask and explicit
        # positions are carried through every step, so a padded batch reproduces the
        # unpadded computation.
        self.pad_aware_generation = bool(pad_aware_generation)
        # The question pass only needs the last position's hidden state, but the backbone
        # computes logits for every position; over LLaMA's 128k vocabulary at batch 128
        # that is a 4 GB tensor.  With this flag the prompt pass asks for one position of
        # logits (``logits_to_keep=1``); hidden states are unaffected.  Off for GPT-2.
        self.prompt_logits_last_only = bool(prompt_logits_last_only)
        original_vocab_size = int(self.codi.config.vocab_size)
        self.pad_token_id = original_vocab_size
        self.bot_id = original_vocab_size + 1
        self.eot_id = original_vocab_size + 2
        try:
            # The released checkpoints overwrite the three new rows, so the covariance
            # based initialisation newer Transformers performs is wasted work.
            self.codi.resize_token_embeddings(original_vocab_size + 3, mean_resizing=False)
        except TypeError:  # older Transformers without the keyword
            self.codi.resize_token_embeddings(original_vocab_size + 3)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=list(lora_target_modules),
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
        base = official_codi_base_model(self)
        if hasattr(base, "transformer") and hasattr(base.transformer, "wte"):
            return base.transformer.wte                      # GPT-2
        embeddings = base.get_input_embeddings()             # LLaMA and other HF causal LMs
        if embeddings is None:
            raise TypeError("official CODI backbone exposes no input embeddings")
        return embeddings

    def tie_weights(self) -> None:
        self.codi.tie_weights()


def official_codi_base_model(model: OfficialCODIGPT2) -> nn.Module:
    """Return the causal LM below PEFT, or the already-merged causal LM.

    The author-compatible loader initially constructs a ``PeftModel``.  Optimized
    inference may subsequently call ``merge_and_unload()``, which replaces that
    wrapper with the ordinary Transformers causal LM.  Keeping this distinction in
    one helper prevents inference utilities from depending on either representation.
    """
    getter = getattr(model.codi, "get_base_model", None)
    return getter() if callable(getter) else model.codi


def prompt_logits_kwargs(model) -> dict:
    """Extra kwargs for the question pass: one position of logits when the model asks
    for it (LLaMA), nothing otherwise (the released GPT-2 path)."""
    return {"logits_to_keep": 1} if getattr(model, "prompt_logits_last_only", False) else {}


class _PadAwareStepper:
    """Carries the attention mask and explicit position ids across incremental
    forwards (``None`` inputs when the model is not pad-aware, i.e. the released path)."""

    def __init__(self, model, attention_mask: torch.Tensor | None):
        self.active = bool(getattr(model, "pad_aware_generation", False)) and attention_mask is not None
        if self.active:
            self.mask = attention_mask.to(torch.long)
            positions = (self.mask.cumsum(-1) - 1).clamp_min(0)
            self.prompt_positions = positions
            self.next_position = positions[:, -1:] + 1

    def prompt_kwargs(self) -> dict:
        return {"position_ids": self.prompt_positions} if self.active else {}

    def step_kwargs(self, new_tokens: int) -> dict:
        """Call once per incremental forward of ``new_tokens`` positions."""
        if not self.active:
            return {}
        ones = torch.ones(self.mask.shape[0], new_tokens, dtype=self.mask.dtype, device=self.mask.device)
        self.mask = torch.cat((self.mask, ones), dim=1)
        offsets = torch.arange(new_tokens, device=self.mask.device).unsqueeze(0)
        positions = self.next_position + offsets
        self.next_position = positions[:, -1:] + 1
        return {"attention_mask": self.mask, "position_ids": positions}


def prepare_backbone_config(config):
    """Compatibility shim: with torch < 2.5 Transformers 4.52 has no tensor-parallel
    styles and rejects any backbone that declares a TP plan (LLaMA).  The plan is
    irrelevant to single-device inference, so it is dropped there and left alone
    elsewhere.  GPT-2 declares none, so it is never touched."""
    try:
        from transformers.integrations import tensor_parallel as tp
    except ImportError:  # pragma: no cover - older Transformers
        return config
    if getattr(tp, "ALL_PARALLEL_STYLES", None) is None and getattr(config, "base_model_tp_plan", None):
        config.base_model_tp_plan = None
    return config


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
    from transformers import AutoConfig

    config = prepare_backbone_config(AutoConfig.from_pretrained(base_model, revision=base_revision, token=token or None))
    backbone = AutoModelForCausalLM.from_pretrained(base_model, config=config, **pretrained_kwargs)
    targets = settings.get("lora_target_modules")
    model = OfficialCODIGPT2(
        backbone,
        lora_rank=int(settings["lora_rank"]),
        lora_alpha=int(settings["lora_alpha"]),
        lora_dropout=float(settings["lora_dropout"]),
        projection_dim=int(settings["projection_dim"]),
        projection_dropout=float(settings["projection_dropout"]),
        projection_layer_norm=bool(settings["projection_layer_norm"]),
        lora_target_modules=tuple(str(m) for m in targets) if targets else GPT2_LORA_TARGET_MODULES,
        question_special_tokens=bool(settings.get("question_special_tokens", False)),
        pad_aware_generation=bool(settings.get("pad_aware_generation", False)),
        prompt_logits_last_only=bool(settings.get("prompt_logits_last_only", False)),
    )

    tokenizer_kwargs = {
        "revision": base_revision,
        "model_max_length": int(settings["model_max_length"]),
        "padding_side": "left",
        "use_fast": bool(settings.get("tokenizer_use_fast", False)),
    }
    if token:
        tokenizer_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        # GPT-2: the released code adds [PAD], which lands on the model's pad id.
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if tokenizer.pad_token_id != model.pad_token_id:
            raise RuntimeError(
                "official tokenizer/model special-token contract changed: "
                f"pad={tokenizer.pad_token_id}, expected={model.pad_token_id}"
            )
    # LLaMA: the tokenizer already has a pad id; the released code keeps it for
    # question padding (positions are masked), so no contract is imposed.
    tokenizer.official_codi_answer_space_token = bool(settings.get("answer_space_token", False))
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

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise TypeError("official checkpoint must contain a non-empty tensor state dict")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("official checkpoint state dict contains non-tensor values")

    target = model.state_dict()
    # Whole top-level modules the architecture does not define are training-time
    # auxiliaries (SIM-CoT's step decoder, CODI-1B's dynamic_cls head).  They are
    # dropped before matching and reported; anything under a module the model does
    # have is still held to the shape and coverage checks below.
    model_prefixes = {key.split(".")[0] for key in target}
    ignored_prefixes = tuple(sorted({key.split(".")[0] for key in state if key.split(".")[0] not in model_prefixes}))
    ignored_tensors = sum(1 for key in state if key.split(".")[0] in ignored_prefixes)
    state = {key: value for key, value in state.items() if key.split(".")[0] not in ignored_prefixes}
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

    required_fragments = ("prj.1.weight", "prj.3.weight", "lora_A", "lora_B")
    absent = [
        fragment
        for fragment in required_fragments
        if not any(fragment in key for key in matched)
    ]
    if not any("wte.weight" in key or "embed_tokens.weight" in key for key in matched):
        absent.append("input embeddings (wte.weight or embed_tokens.weight)")
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
        ignored_prefixes=ignored_prefixes,
        ignored_tensors=ignored_tensors,
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


def _set_output_head_answer_position(
    model: OfficialCODIGPT2, position: int | None
) -> None:
    """Tell an opt-in routed output head which visible answer token is next.

    Ordinary Hugging Face heads do not implement ``set_answer_position``, so this
    is a no-op for every existing experiment.  ``None`` marks prompt encoding and
    continuous-latent passes, where CODI computes but does not consume vocabulary
    logits.
    """
    base_model = official_codi_base_model(model)
    head = base_model.get_output_embeddings()
    setter = getattr(head, "set_answer_position", None)
    if setter is not None:
        setter(position)


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
    answer_state_observer=None,
    answer_logit_observer=None,
    latent_state_hook=None,
) -> list[str] | tuple[list[str], dict]:
    """Greedy generation matching the released path, with optional causal KV edits.

    ``answer_state_observer`` is an opt-in measurement hook. It receives the final
    normalized state used to predict each answer token, the active-row mask, and the
    zero-based answer-token position. The default path remains byte-for-byte
    unchanged and does not request hidden states.

    ``answer_logit_observer`` receives the corresponding vocabulary logits, active
    rows, and answer-token position.  It is intended for read-only fidelity audits;
    tensors are detached before the callback and generation decisions are unchanged.

    ``latent_state_hook(state, latent_position, chunk_start)`` is an opt-in
    intervention on the latent trajectory (ledger 105): it receives the last hidden
    state of each thought ``[B, D]`` before the projector and must return the state to
    project (the same tensor to observe only).  With the default ``None`` the path is
    byte-for-byte the released one.
    """
    if latent_iterations <= 0:
        raise ValueError("latent_iterations must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model.eval()
    outputs: list[str] = []
    endpoint_reached: list[bool] = []
    generated_token_counts: list[int] = []
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
            add_special_tokens=bool(getattr(model, "question_special_tokens", False)),
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

        stepper = _PadAwareStepper(model, attention_mask)
        _set_output_head_answer_position(model, None)
        encoded = model.codi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            **stepper.prompt_kwargs(),
            **prompt_logits_kwargs(model),
        )
        cache = encoded.past_key_values
        latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
        for latent_position in range(latent_iterations):
            _set_output_head_answer_position(model, None)
            latent_output = model.codi(
                inputs_embeds=latent,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                **stepper.step_kwargs(1),
            )
            cache = latent_output.past_key_values
            if kv_intervention is not None:
                cache = kv_intervention(cache, latent_position)
            slot_state = latent_output.hidden_states[-1][:, -1, :]
            if latent_state_hook is not None:
                slot_state = latent_state_hook(slot_state, latent_position, start)
            latent = model.prj(slot_state.unsqueeze(1))

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
            _set_output_head_answer_position(model, 0)
            with context:
                decoded = model.codi(
                    inputs_embeds=embedding(forced),
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=answer_state_observer is not None,
                    output_attentions=False,
                    return_dict=True,
                    **stepper.step_kwargs(int(forced.shape[1])),
                )
            cache = decoded.past_key_values
            endpoint_applied |= endpoint_mask
            if answer_state_observer is not None:
                answer_state_observer(
                    decoded.hidden_states[-1][:, -1, :],
                    ~finished,
                    0,
                )
            if answer_logit_observer is not None:
                answer_logit_observer(
                    decoded.logits[:, -1, : model.eot_id].detach(),
                    ~finished,
                    0,
                )
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

        # Opt-in flag used only by the margin-geometry all-position arms.  It
        # defaults to False, so every completed experiment keeps intervening on
        # the single answer-cue forward pass exactly as before.
        intervene_everywhere = bool(
            getattr(answer_endpoint_intervention, "applies_to_all_positions", False)
        )
        for answer_position in remaining_steps:
            if bool(finished.all()):
                break
            if intervene_everywhere:
                endpoint_mask = ~finished
            elif force_answer_cue:
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
            _set_output_head_answer_position(model, int(answer_position))
            with context:
                decoded = model.codi(
                    inputs_embeds=token_embedding,
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=answer_state_observer is not None,
                    output_attentions=False,
                    return_dict=True,
                    **stepper.step_kwargs(1),
                )
            cache = decoded.past_key_values
            if answer_state_observer is not None:
                answer_state_observer(
                    decoded.hidden_states[-1][:, -1, :],
                    ~finished,
                    int(answer_position),
                )
            if answer_logit_observer is not None:
                answer_logit_observer(
                    decoded.logits[:, -1, : model.eot_id].detach(),
                    ~finished,
                    int(answer_position),
                )
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
        generated_token_counts.extend(len(token_ids) for token_ids in generated)
        endpoint_reached.extend(bool(value) for value in endpoint_applied.tolist())

    _set_output_head_answer_position(model, None)

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
            "generated_token_counts": generated_token_counts,
            "generated_token_count": int(sum(generated_token_counts)),
        }
    return outputs
