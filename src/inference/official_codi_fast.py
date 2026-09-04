"""Faster inference path for the author-compatible CODI GPT-2 model.

The reference decoder deliberately follows the released implementation.  This
module provides an inference-only path that preserves its transformer, projector,
KV-cache, answer cue, and greedy-decoding semantics while avoiding vocabulary
projections whose logits are discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import torch
import torch.nn.functional as F

from src.models.official_codi import (
    OfficialCODIGPT2,
    _normalized_official_questions,
    official_codi_base_model,
)


@dataclass(frozen=True)
class PreparedCODIBatch:
    """One left-padded CPU batch plus its positions in the original request."""

    original_indices: tuple[int, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class PreparedCODIQuestions:
    """Tokenized questions ready for repeated, comparable timing runs."""

    batches: tuple[PreparedCODIBatch, ...]
    question_count: int
    preparation_seconds: float
    tokenization_seconds: float
    padded_prompt_tokens: int
    unpadded_prompt_tokens: int
    length_bucketed: bool


@dataclass(frozen=True)
class FastCODIGeneration:
    texts: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    generated_token_counts: tuple[int, ...]

    @property
    def generated_token_count(self) -> int:
        return int(sum(self.generated_token_counts))


def prepare_official_codi_batches(
    tokenizer,
    questions: Iterable[str],
    *,
    batch_size: int,
    length_bucketed: bool,
) -> PreparedCODIQuestions:
    """Tokenize once and optionally group similar prompt lengths together."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    normalized = _normalized_official_questions(questions)
    preparation_started = time.perf_counter()
    started = time.perf_counter()
    tokenized = tokenizer(
        normalized,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]
    tokenization_seconds = time.perf_counter() - started
    rows = [(index, [int(token) for token in ids]) for index, ids in enumerate(tokenized)]
    if length_bucketed:
        rows.sort(key=lambda item: (len(item[1]), item[0]))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("the official CODI tokenizer must define a padding token")
    batches: list[PreparedCODIBatch] = []
    padded_prompt_tokens = 0
    unpadded_prompt_tokens = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        maximum = max((len(ids) for _, ids in chunk), default=0)
        ids_tensor = torch.full((len(chunk), maximum), int(pad_id), dtype=torch.long)
        mask_tensor = torch.zeros((len(chunk), maximum), dtype=torch.long)
        for row, (_, ids) in enumerate(chunk):
            if ids:
                ids_tensor[row, maximum - len(ids) :] = torch.tensor(ids, dtype=torch.long)
                mask_tensor[row, maximum - len(ids) :] = 1
            unpadded_prompt_tokens += len(ids)
        padded_prompt_tokens += len(chunk) * maximum
        batches.append(
            PreparedCODIBatch(
                original_indices=tuple(index for index, _ in chunk),
                input_ids=ids_tensor,
                attention_mask=mask_tensor,
            )
        )
    preparation_seconds = time.perf_counter() - preparation_started
    return PreparedCODIQuestions(
        batches=tuple(batches),
        question_count=len(rows),
        preparation_seconds=float(preparation_seconds),
        tokenization_seconds=float(tokenization_seconds),
        padded_prompt_tokens=int(padded_prompt_tokens),
        unpadded_prompt_tokens=int(unpadded_prompt_tokens),
        length_bucketed=bool(length_bucketed),
    )


def merge_official_codi_lora_(model: OfficialCODIGPT2) -> OfficialCODIGPT2:
    """Merge the loaded PEFT adapter into GPT-2 weights in place."""
    merger = getattr(model.codi, "merge_and_unload", None)
    if not callable(merger):
        raise TypeError("official CODI does not currently contain a mergeable PEFT adapter")
    model.codi = merger(progressbar=False)
    model.tie_weights()
    return model


def numeric_vocabulary_candidates(
    tokenizer,
    *,
    vocabulary_stop: int,
    extra_characters: str = ".,+-$%/",
) -> torch.Tensor:
    """Return a tokenizer-semantic numeric shortlist for the GSM8K exploration.

    Selection depends only on individual token strings, never on evaluation states,
    labels, or model predictions. EOS is always included.
    """
    if vocabulary_stop <= 0:
        raise ValueError("vocabulary_stop must be positive")
    allowed = set("0123456789" + str(extra_characters))
    candidates = []
    for token_id in range(int(vocabulary_stop)):
        piece = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        stripped = piece.strip()
        if stripped and set(stripped) <= allowed:
            candidates.append(token_id)
    eos = tokenizer.eos_token_id
    if eos is None or not 0 <= int(eos) < int(vocabulary_stop):
        raise ValueError("EOS must be inside the eligible official CODI vocabulary")
    candidates.append(int(eos))
    unique = sorted(set(candidates))
    if len(unique) < 2:
        raise RuntimeError("numeric vocabulary shortlist is unexpectedly empty")
    return torch.tensor(unique, dtype=torch.long)


def _select_token(
    output_head,
    hidden: torch.Tensor,
    *,
    vocabulary_stop: int,
    candidate_token_ids: torch.Tensor | None,
    candidate_weight: torch.Tensor | None,
    candidate_bias: torch.Tensor | None,
) -> torch.Tensor:
    if candidate_token_ids is None:
        return output_head(hidden)[..., :vocabulary_stop].argmax(dim=-1)
    if candidate_weight is None:
        raise RuntimeError("candidate weight was not prepared")
    scores = F.linear(hidden, candidate_weight, candidate_bias)
    return candidate_token_ids.index_select(0, scores.argmax(dim=-1))


@torch.inference_mode()
def generate_official_codi_fast(
    model: OfficialCODIGPT2,
    tokenizer,
    prepared: PreparedCODIQuestions,
    *,
    latent_iterations: int,
    max_new_tokens: int,
    device: torch.device,
    answer_cue: str = "The answer is:",
    candidate_token_ids: torch.Tensor | None = None,
    answer_state_observer=None,
) -> FastCODIGeneration:
    """Decode forced-cue CODI answers without computing unused vocabulary logits."""
    if latent_iterations <= 0:
        raise ValueError("latent_iterations must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prepared.question_count <= 0:
        raise ValueError("at least one prepared question is required")

    model.eval()
    base = official_codi_base_model(model)
    transformer = getattr(base, "transformer", None)
    if transformer is None:
        raise TypeError("fast official CODI path requires a GPT-2 transformer body")
    embedding = model.input_embeddings()
    output_head = base.get_output_embeddings()
    answer_position_setter = getattr(output_head, "set_answer_position", None)

    def set_answer_position(position: int | None) -> None:
        if answer_position_setter is not None:
            answer_position_setter(position)

    set_answer_position(None)
    vocabulary_stop = int(model.eot_id)
    candidate_ids_device = None
    candidate_weight = None
    candidate_bias = None
    if candidate_token_ids is not None:
        candidate_ids_device = candidate_token_ids.to(device=device)
        candidate_weight = output_head.weight.index_select(
            0, candidate_ids_device
        ).to(dtype=output_head.weight.dtype)
        head_bias = getattr(output_head, "bias", None)
        if head_bias is not None:
            candidate_bias = head_bias.index_select(0, candidate_ids_device).to(
                dtype=output_head.weight.dtype
            )
    cue_ids = list(tokenizer(f" {answer_cue}", add_special_tokens=False)["input_ids"])
    if not cue_ids:
        raise ValueError("answer cue must tokenize to at least one token")

    result_tokens: list[tuple[int, ...] | None] = [None] * prepared.question_count
    result_texts: list[str | None] = [None] * prepared.question_count
    result_counts = [0] * prepared.question_count

    for batch in prepared.batches:
        input_ids = batch.input_ids.to(device=device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device=device, non_blocking=True)
        bot = torch.full(
            (input_ids.shape[0], 1), model.bot_id, dtype=torch.long, device=device
        )
        input_ids = torch.cat((input_ids, bot), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(bot)), dim=1)

        encoded = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        cache = encoded.past_key_values
        latent = model.prj(encoded.last_hidden_state[:, -1:, :])
        for _ in range(int(latent_iterations)):
            latent_output = transformer(
                inputs_embeds=latent,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            cache = latent_output.past_key_values
            latent = model.prj(latent_output.last_hidden_state[:, -1:, :])

        forced = torch.tensor(
            [model.eot_id, *cue_ids], dtype=torch.long, device=device
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
        cue_output = transformer(
            inputs_embeds=embedding(forced),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        cache = cue_output.past_key_values
        active = torch.ones(input_ids.shape[0], dtype=torch.bool, device=device)
        set_answer_position(0)
        if answer_state_observer is not None:
            answer_state_observer(cue_output.last_hidden_state[:, -1, :], active, 0)
        next_token = _select_token(
            output_head,
            cue_output.last_hidden_state[:, -1, :],
            vocabulary_stop=vocabulary_stop,
            candidate_token_ids=candidate_ids_device,
            candidate_weight=candidate_weight,
            candidate_bias=candidate_bias,
        )

        size = input_ids.shape[0]
        finished = torch.zeros(size, dtype=torch.bool, device=device)
        token_buffer = torch.full(
            (size, max_new_tokens), int(tokenizer.eos_token_id),
            dtype=torch.long, device=device,
        )
        counts = torch.zeros(size, dtype=torch.long, device=device)

        for answer_position in range(max_new_tokens):
            active = ~finished
            token_buffer[:, answer_position] = torch.where(
                active, next_token, token_buffer[:, answer_position]
            )
            counts += active.long()
            finished |= active & next_token.eq(int(tokenizer.eos_token_id))
            if answer_position + 1 >= max_new_tokens or bool(finished.all()):
                break
            token_output = transformer(
                inputs_embeds=embedding(next_token).unsqueeze(1),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            cache = token_output.past_key_values
            set_answer_position(answer_position + 1)
            if answer_state_observer is not None:
                answer_state_observer(
                    token_output.last_hidden_state[:, -1, :],
                    ~finished,
                    answer_position + 1,
                )
            next_token = _select_token(
                output_head,
                token_output.last_hidden_state[:, -1, :],
                vocabulary_stop=vocabulary_stop,
                candidate_token_ids=candidate_ids_device,
                candidate_weight=candidate_weight,
                candidate_bias=candidate_bias,
            )

        cpu_tokens = token_buffer.cpu()
        cpu_counts = counts.cpu().tolist()
        for row, original_index in enumerate(batch.original_indices):
            count = int(cpu_counts[row])
            ids = tuple(int(token) for token in cpu_tokens[row, :count].tolist())
            result_tokens[original_index] = ids
            result_counts[original_index] = count
            result_texts[original_index] = tokenizer.decode(ids, skip_special_tokens=True)

    set_answer_position(None)
    if any(value is None for value in result_tokens) or any(
        value is None for value in result_texts
    ):
        raise RuntimeError("fast CODI decoder did not restore every original row")
    return FastCODIGeneration(
        texts=tuple(str(value) for value in result_texts),
        token_ids=tuple(value for value in result_tokens if value is not None),
        generated_token_counts=tuple(int(value) for value in result_counts),
    )
