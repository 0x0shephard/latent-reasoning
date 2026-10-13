"""Coconut (Hao et al. 2024) adapter for the audit (ledger §127 C).

Reproduces the released forward pass: the question, ``<|start-latent|>``, six
``<|latent|>`` placeholders and ``<|end-latent|>``; each latent position's input
embedding is the last hidden state of the position before it; text follows.  The
audit's hooks are the same as the CODI path's: a state hook that may replace a thought's
output state before it is fed forward, a K/V hook at each latent position, and K/V masks.
Decoding is batched with left padding, explicit position ids and a cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Sequence

import torch

from src.mech.cache_carrier import Trajectory, _last_position_kv, cache_to_legacy

ANSWER_PATTERN = re.compile(r"###\s*\$?(-?[\d,]*\.?\d+)")
NUMBER_PATTERN = re.compile(r"-?[\d,]*\.?\d+")


@dataclass
class CoconutSpec:
    repo_id: str = "UdeSA/coconut-0.2B-gsm8k-aug"
    revision: str = "741379d8425c1d0640ea92756bf3fa0c291532e9"
    subfolder: str = "coconut"
    weight_file: str = "pytorch_model.bin"
    n_latents: int = 6
    published_accuracy: float = 0.3457


def coconut_answer_number(text: str) -> float | None:
    """The number after ``###`` (the released answer format), else the last number."""
    m = ANSWER_PATTERN.findall(text)
    cands = m if m else NUMBER_PATTERN.findall(text)
    for c in reversed(cands):
        try:
            return float(c.replace(",", ""))
        except ValueError:
            continue
    return None


class CoconutGPT2(torch.nn.Module):
    """The released GPT-2 Coconut checkpoint with the audit's hooks."""

    family = "coconut"

    def __init__(self, base, tokenizer, *, latent_id: int, start_id: int, end_id: int, n_latents: int = 6):
        super().__init__()
        self.base, self.tokenizer = base, tokenizer
        self.latent_id, self.start_id, self.end_id, self.n_latents = latent_id, start_id, end_id, n_latents
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id

    @classmethod
    def from_hub(cls, spec: CoconutSpec = CoconutSpec(), *, dtype=torch.float32, token=None) -> "CoconutGPT2":
        import json
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

        tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, subfolder=spec.subfolder, revision=spec.revision, token=token)
        config = GPT2Config.from_pretrained(spec.repo_id, subfolder=spec.subfolder, revision=spec.revision, token=token)
        base = GPT2LMHeadModel(config)
        path = hf_hub_download(spec.repo_id, f"{spec.subfolder}/{spec.weight_file}", revision=spec.revision, token=token)
        state = torch.load(path, map_location="cpu", weights_only=True)
        extra = state.pop("embedding.weight", None)   # the Coconut class keeps a reference to wte under this name
        state = {k[len("base_causallm."):] if k.startswith("base_causallm.") else k: v for k, v in state.items()}
        if extra is not None and not torch.equal(extra, state["transformer.wte.weight"]):
            raise RuntimeError("Coconut embedding.weight differs from transformer.wte.weight")
        missing, unexpected = base.load_state_dict(state, strict=False)
        missing = [k for k in missing if k != "lm_head.weight"]   # tied to wte
        if missing or unexpected:
            raise RuntimeError(f"Coconut state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        base.tie_weights()
        added = json.loads(open(hf_hub_download(spec.repo_id, f"{spec.subfolder}/added_tokens.json", revision=spec.revision, token=token)).read())
        model = cls(base.to(dtype), tokenizer, latent_id=added["<|latent|>"], start_id=added["<|start-latent|>"], end_id=added["<|end-latent|>"], n_latents=spec.n_latents)
        model.eval()
        return model

    # ------------------------------------------------------------------ pieces
    def question_ids(self, question: str) -> list[int]:
        return self.tokenizer.encode(question + "\n", add_special_tokens=True) + [self.start_id]

    @property
    def embedding(self):
        return self.base.get_input_embeddings()

    def readout_matrix(self) -> torch.Tensor:
        return self.base.get_output_embeddings().weight.detach().float().cpu()

    def hidden_size(self) -> int:
        return int(self.base.config.n_embd)

    def n_layers(self) -> int:
        return int(self.base.config.n_layer)

    # ------------------------------------------------------------------ the latent pass
    @torch.inference_mode()
    def run_batch(self, questions: Sequence[str], *, device, state_hook: Callable | None = None, kv_hook: Callable | None = None,
                  mask_slots: Sequence[int] = (), max_new_tokens: int = 64, record_kv: bool = True) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[str]]:
        """One left-padded batch: six latent passes with hooks, then cached greedy decoding.
        Returns (states [B, S, D], keys [B, S, L, H, h], values, decoded texts)."""
        ids = [self.question_ids(q) for q in questions]
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, width - len(x):] = torch.tensor(x); attention[i, width - len(x):] = 1
        input_ids, attention = input_ids.to(device), attention.to(device)
        position_ids = (attention.cumsum(1) - 1).clamp_min(0)
        out = self.base(input_ids=input_ids, attention_mask=attention, position_ids=position_ids, use_cache=True, output_hidden_states=True)
        cache = out.past_key_values
        state = out.hidden_states[-1][:, -1, :]
        next_pos = position_ids[:, -1:] + 1
        states, keys, values = [], [], []
        latent_columns = []
        for s in range(self.n_latents):
            attention = torch.cat([attention, torch.ones((attention.shape[0], 1), dtype=attention.dtype, device=device)], 1)
            out = self.base(inputs_embeds=state.unsqueeze(1).to(self.base.dtype), attention_mask=attention, position_ids=next_pos, past_key_values=cache,
                            use_cache=True, output_hidden_states=True)
            cache = out.past_key_values
            latent_columns.append(attention.shape[1] - 1)
            if kv_hook is not None:
                cache = kv_hook(cache, s)
            if record_kv:
                k, v = _last_position_kv(cache); keys.append(k.float().cpu()); values.append(v.float().cpu())
            state = out.hidden_states[-1][:, -1, :]
            if state_hook is not None:
                state = state_hook(state, s)
            states.append(state.float().cpu())
            if s in mask_slots:
                attention[:, latent_columns[-1]] = 0   # hide this thought's K/V from every later position
            next_pos = next_pos + 1
        # the end-latent token, then greedy decoding with the cache
        end = torch.full((attention.shape[0], 1), self.end_id, dtype=torch.long, device=device)
        attention = torch.cat([attention, torch.ones_like(end)], 1)
        out = self.base(input_ids=end, attention_mask=attention, position_ids=next_pos, past_key_values=cache, use_cache=True)
        cache = out.past_key_values; next_pos = next_pos + 1
        token = out.logits[:, -1, :].argmax(-1)
        generated = [token]
        finished = token == self.eos_id
        for _ in range(max_new_tokens - 1):
            if bool(finished.all()):
                break
            attention = torch.cat([attention, torch.ones((attention.shape[0], 1), dtype=attention.dtype, device=device)], 1)
            out = self.base(input_ids=token.unsqueeze(1), attention_mask=attention, position_ids=next_pos, past_key_values=cache, use_cache=True)
            cache = out.past_key_values; next_pos = next_pos + 1
            token = out.logits[:, -1, :].argmax(-1)
            token = torch.where(finished, torch.full_like(token, self.eos_id), token)
            generated.append(token); finished = finished | (token == self.eos_id)
        gen = torch.stack(generated, 1).cpu()
        texts = []
        for row in gen.tolist():
            cut = row.index(self.eos_id) if self.eos_id in row else len(row)
            texts.append(self.tokenizer.decode(row[:cut], skip_special_tokens=True))
        st = torch.stack(states, 1)
        return st, (torch.stack(keys, 1) if record_kv else None), (torch.stack(values, 1) if record_kv else None), texts


def coconut_record(model: CoconutGPT2, questions: Sequence[str], *, batch_size: int, device, **kw) -> tuple[Trajectory, list[str]]:
    """Native decoding with every thought's state and K/V recorded."""
    S, K, V, T = [], [], [], []
    for start in range(0, len(questions), batch_size):
        s, k, v, t = model.run_batch(questions[start:start + batch_size], device=device, **kw)
        S.append(s); K.append(k); V.append(v); T += t
    return Trajectory(states=torch.cat(S), keys=torch.cat(K), values=torch.cat(V)), T


def coconut_generate_under(model: CoconutGPT2, questions: Sequence[str], *, batch_size: int, device, hidden_donors: dict[int, torch.Tensor] | None = None,
                           cache_edits=(), donor: Trajectory | None = None, mask_slots: Sequence[int] = (), max_new_tokens: int = 64) -> list[str]:
    """Native decoding under route swaps: donor states fed forward at the given slots, donor K/V
    patched at the given positions, or thoughts hidden by the attention mask."""
    from src.mech.cache_carrier import apply_edits

    texts = []
    for start in range(0, len(questions), batch_size):
        rows = slice(start, start + min(batch_size, len(questions) - start))

        def state_hook(state, s):
            if hidden_donors and s in hidden_donors:
                return hidden_donors[s][rows].to(state.device, state.dtype)
            return state

        def kv_hook(cache, s):
            return apply_edits(cache, s, list(cache_edits), donor, rows) if cache_edits else cache

        _, _, _, t = model.run_batch(questions[rows], device=device, state_hook=state_hook, kv_hook=kv_hook, mask_slots=mask_slots,
                                     max_new_tokens=max_new_tokens, record_kv=False)
        texts += t
    return texts
