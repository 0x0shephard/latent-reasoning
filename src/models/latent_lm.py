"""Shared continuous-thought causal LM used by CODI and KaVa.

The architecture is deliberately identical across methods.  Only the configured
distillation loss changes.  Autoregressive latent generation is the primary controlled
mechanism; a parallel Jacobi fixed-point update is available as a separately labelled
ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from src.data.teacher_cache import LatentBatch, cache_to_tensors


BOT_TOKEN = "<bot>"
EOT_TOKEN = "<eot>"
LatentIntervention = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class StudentOutput:
    answer_loss: torch.Tensor
    hidden_endpoint: torch.Tensor
    latent_hidden: torch.Tensor
    latent_keys: torch.Tensor
    latent_values: torch.Tensor


def add_latent_tokens(tokenizer, model=None) -> tuple[int, int]:
    """Add the two phase markers deterministically and optionally resize a backbone."""
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [BOT_TOKEN, EOT_TOKEN]}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
    bot = tokenizer.convert_tokens_to_ids(BOT_TOKEN)
    eot = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    if bot == tokenizer.unk_token_id or eot == tokenizer.unk_token_id or bot == eot:
        raise ValueError("tokenizer failed to register distinct <bot>/<eot> tokens")
    return int(bot), int(eot)


def _position_ids(mask: torch.Tensor) -> torch.Tensor:
    positions = mask.long().cumsum(dim=-1) - 1
    return positions.masked_fill(mask == 0, 0)


class LatentCausalLM(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        bot_token_id: int,
        eot_token_id: int,
        latent_steps: int = 6,
        mechanism: str = "autoregressive",
        jacobi_iterations: int = 3,
        projection_dim: int | None = None,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_steps <= 0:
            raise ValueError("latent_steps must be positive")
        if mechanism not in {"autoregressive", "jacobi"}:
            raise ValueError("mechanism must be 'autoregressive' or 'jacobi'")
        if jacobi_iterations <= 0:
            raise ValueError("jacobi_iterations must be positive")
        hidden = int(backbone.config.hidden_size)
        middle = int(projection_dim or hidden)
        self.backbone = backbone
        self.bot_token_id = int(bot_token_id)
        self.eot_token_id = int(eot_token_id)
        self.latent_steps = int(latent_steps)
        self.mechanism = mechanism
        self.jacobi_iterations = int(jacobi_iterations)
        self.projection = nn.Sequential(
            nn.Dropout(projection_dropout),
            nn.Linear(hidden, middle),
            nn.GELU(),
            nn.Linear(middle, hidden),
            nn.LayerNorm(hidden),
        )

    @property
    def config(self):
        return self.backbone.config

    def _autoregressive_latents(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        intervention: LatentIntervention | None = None,
    ):
        initial = self.backbone(
            input_ids=question_ids,
            attention_mask=question_mask,
            position_ids=_position_ids(question_mask),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        lengths = question_mask.sum(dim=-1)
        batch_index = torch.arange(question_ids.shape[0], device=question_ids.device)
        last = initial.hidden_states[-1][batch_index, lengths - 1]
        latent_embedding = self.projection(last).unsqueeze(1)
        cache = initial.past_key_values
        all_hidden = []
        full_mask = question_mask
        for step in range(self.latent_steps):
            if intervention is not None:
                original = latent_embedding.squeeze(1)
                intervened = intervention(original, step)
                if (
                    intervened.shape != original.shape
                    or intervened.device != original.device
                    or intervened.dtype != original.dtype
                ):
                    raise ValueError(
                        "latent intervention must preserve state shape, device, and dtype"
                    )
                latent_embedding = intervened.unsqueeze(1)
            full_mask = torch.cat(
                [full_mask, torch.ones_like(full_mask[:, :1])], dim=-1
            )
            out = self.backbone(
                inputs_embeds=latent_embedding,
                attention_mask=full_mask,
                position_ids=(lengths + step).unsqueeze(-1),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = out.past_key_values
            layer_hidden = torch.stack(out.hidden_states[1:], dim=1).squeeze(2)
            all_hidden.append(layer_hidden)
            latent_embedding = self.projection(out.hidden_states[-1][:, -1]).unsqueeze(1)
        latent_hidden = torch.stack(all_hidden, dim=2)  # [B,L,M,D]
        keys, values = cache_to_tensors(cache)
        start = question_ids.shape[1]
        return (
            cache,
            full_mask,
            latent_hidden,
            keys[:, :, :, start : start + self.latent_steps],
            values[:, :, :, start : start + self.latent_steps],
        )

    def _jacobi_latents(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        intervention: LatentIntervention | None = None,
    ):
        # Initialize every parallel slot from the final <bot> activation, then repeatedly
        # update all slots with a causal full-sequence pass.  This is kept as an explicit
        # ablation; autoregressive generation is the controlled primary experiment.
        seed = self.backbone(
            input_ids=question_ids,
            attention_mask=question_mask,
            position_ids=_position_ids(question_mask),
            output_hidden_states=True,
            return_dict=True,
        )
        lengths = question_mask.sum(dim=-1)
        batch_index = torch.arange(question_ids.shape[0], device=question_ids.device)
        bot_hidden = seed.hidden_states[-1][batch_index, lengths - 1]
        latents = self.projection(bot_hidden).unsqueeze(1).expand(
            -1, self.latent_steps, -1
        ).contiguous()
        word_embeddings = self.backbone.get_input_embeddings()(question_ids)
        latent_mask = torch.ones(
            (question_ids.shape[0], self.latent_steps),
            dtype=question_mask.dtype,
            device=question_mask.device,
        )
        full_mask = torch.cat([question_mask, latent_mask], dim=-1)
        positions = _position_ids(full_mask)
        final = None
        for _ in range(self.jacobi_iterations):
            effective_latents = latents
            if intervention is not None:
                states = []
                for step in range(self.latent_steps):
                    original = latents[:, step]
                    intervened = intervention(original, step)
                    if (
                        intervened.shape != original.shape
                        or intervened.device != original.device
                        or intervened.dtype != original.dtype
                    ):
                        raise ValueError(
                            "latent intervention must preserve state shape, device, and dtype"
                        )
                    states.append(intervened)
                effective_latents = torch.stack(states, dim=1)
            final = self.backbone(
                inputs_embeds=torch.cat([word_embeddings, effective_latents], dim=1),
                attention_mask=full_mask,
                position_ids=positions,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            latents = self.projection(final.hidden_states[-1][:, -self.latent_steps :])
        assert final is not None
        start = question_ids.shape[1]
        layer_hidden = torch.stack(final.hidden_states[1:], dim=1)
        keys, values = cache_to_tensors(final.past_key_values)
        return (
            final.past_key_values,
            full_mask,
            layer_hidden[:, :, start : start + self.latent_steps],
            keys[:, :, :, start : start + self.latent_steps],
            values[:, :, :, start : start + self.latent_steps],
        )

    def latent_context(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        intervention: LatentIntervention | None = None,
    ):
        if self.mechanism == "autoregressive":
            return self._autoregressive_latents(
                question_ids, question_mask, intervention=intervention
            )
        return self._jacobi_latents(
            question_ids, question_mask, intervention=intervention
        )

    def forward_student(self, batch: LatentBatch) -> StudentOutput:
        cache, latent_mask, latent_hidden, latent_keys, latent_values = self.latent_context(
            batch.question_ids, batch.question_mask
        )
        combined_mask = torch.cat([latent_mask, batch.student_segment_mask], dim=-1)
        lengths = batch.question_mask.sum(dim=-1) + self.latent_steps
        segment_positions = lengths.unsqueeze(-1) + _position_ids(batch.student_segment_mask)
        out = self.backbone(
            input_ids=batch.student_segment_ids,
            attention_mask=combined_mask,
            position_ids=segment_positions,
            labels=batch.student_labels,
            past_key_values=cache,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if out.loss is None:
            raise ValueError("student forward did not produce answer loss")
        hidden = torch.stack(out.hidden_states[1:], dim=1)
        row = torch.arange(hidden.shape[0], device=hidden.device)
        endpoint = hidden[row, :, batch.student_endpoint, :]
        return StudentOutput(
            answer_loss=out.loss,
            hidden_endpoint=endpoint,
            latent_hidden=latent_hidden,
            latent_keys=latent_keys,
            latent_values=latent_values,
        )

    @torch.no_grad()
    def generate_from_prefix(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        prefix_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int,
        intervention: LatentIntervention | None = None,
    ) -> list[list[int]]:
        """Greedy answer generation after the shared latent block and discrete cue."""
        if max_new_tokens <= 0:
            return [[] for _ in range(question_ids.shape[0])]
        cache, latent_mask, _, _, _ = self.latent_context(
            question_ids, question_mask, intervention=intervention
        )
        prefix_mask = torch.ones_like(prefix_ids)
        lengths = question_mask.sum(dim=-1) + self.latent_steps
        full_mask = torch.cat([latent_mask, prefix_mask], dim=-1)
        positions = lengths.unsqueeze(-1) + _position_ids(prefix_mask)
        out = self.backbone(
            input_ids=prefix_ids,
            attention_mask=full_mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        next_token = out.logits[:, -1].argmax(dim=-1)
        generated = [[] for _ in range(question_ids.shape[0])]
        finished = torch.zeros(question_ids.shape[0], dtype=torch.bool, device=question_ids.device)
        for step in range(max_new_tokens):
            for batch_index, token in enumerate(next_token.tolist()):
                if not finished[batch_index]:
                    generated[batch_index].append(token)
            finished |= next_token.eq(eos_token_id)
            if bool(finished.all()) or step + 1 == max_new_tokens:
                break
            full_mask = torch.cat([full_mask, torch.ones_like(full_mask[:, :1])], dim=-1)
            token_position = lengths + prefix_ids.shape[1] + step
            feed = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            out = self.backbone(
                input_ids=feed.unsqueeze(-1),
                attention_mask=full_mask,
                position_ids=token_position.unsqueeze(-1),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            next_token = out.logits[:, -1].argmax(dim=-1)
        return generated
