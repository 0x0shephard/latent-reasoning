from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from src.inference.official_codi_fast import (
    generate_official_codi_fast,
    numeric_vocabulary_candidates,
    prepare_official_codi_batches,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    pieces = {
        0: "[PAD]",
        1: "[EOS]",
        2: "2",
        3: "[EOT]",
        4: ":",
        5: " word",
        6: " 17",
        7: ".",
    }

    def __call__(self, text, *, add_special_tokens=False, padding=False):
        del add_special_tokens, padding
        if isinstance(text, str):
            if text == " The answer is:":
                return {"input_ids": [4]}
            return {"input_ids": [2] * max(1, len(text.split()))}
        return {
            "input_ids": [[2] * max(1, len(value.split())) for value in text]
        }

    def decode(self, ids, *, skip_special_tokens=False, **kwargs):
        del kwargs
        values = []
        for token in ids:
            if skip_special_tokens and int(token) in {0, 1, 3}:
                continue
            values.append(self.pieces[int(token)])
        return "".join(values)


class IdentityTransformer(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, *, input_ids=None, inputs_embeds=None, **kwargs):
        del kwargs
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=((hidden,),))


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 2)
        self.transformer = IdentityTransformer(self.embedding)
        self.lm_head = nn.Linear(2, 8, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[2] = torch.tensor([1.0, 0.0])
            self.embedding.weight[4] = torch.tensor([0.0, 1.0])
            self.lm_head.weight.zero_()
            self.lm_head.weight[1] = torch.tensor([1.0, 0.0])
            self.lm_head.weight[2] = torch.tensor([0.0, 1.0])

    def get_output_embeddings(self):
        return self.lm_head

    def tie_weights(self):
        return None


class TinyCODI(nn.Module):
    bot_id = 2
    eot_id = 3

    def __init__(self):
        super().__init__()
        self.codi = TinyCausalLM()
        self.prj = nn.Identity()

    def input_embeddings(self):
        return self.codi.embedding


def test_preparation_length_buckets_reduce_padding_and_restore_indices():
    tokenizer = TinyTokenizer()
    questions = ["one two three four", "one", "one two", "one two three"]
    plain = prepare_official_codi_batches(
        tokenizer, questions, batch_size=2, length_bucketed=False
    )
    bucketed = prepare_official_codi_batches(
        tokenizer, questions, batch_size=2, length_bucketed=True
    )
    assert bucketed.padded_prompt_tokens < plain.padded_prompt_tokens
    assert sorted(index for batch in bucketed.batches for index in batch.original_indices) == [
        0,
        1,
        2,
        3,
    ]


def test_numeric_candidates_are_token_semantic_and_include_eos():
    candidates = numeric_vocabulary_candidates(TinyTokenizer(), vocabulary_stop=8)
    assert candidates.tolist() == [1, 2, 6, 7]


def test_fast_decoder_generates_and_restores_original_question_order():
    tokenizer = TinyTokenizer()
    prepared = prepare_official_codi_batches(
        tokenizer,
        ["longer question here", "short"],
        batch_size=1,
        length_bucketed=True,
    )
    result = generate_official_codi_fast(
        TinyCODI(),
        tokenizer,
        prepared,
        latent_iterations=2,
        max_new_tokens=4,
        device=torch.device("cpu"),
    )
    assert result.token_ids == ((2, 1), (2, 1))
    assert result.texts == ("2", "2")
    assert result.generated_token_counts == (2, 2)

