# Official CODI checkpoint validation

## Purpose

The pilot CODI and KaVa checkpoints were trained for one epoch with batch size four and
reached much lower accuracy than the published systems. Before interpreting additional
KV-subspace experiments, this gate tests whether the author-released CODI GPT-2 checkpoint
reproduces its published accuracy under the released inference protocol.

This is evaluation only. It does not update weights and it does not load the public
checkpoint into the pilot `LatentCausalLM`.

## Pinned contract

- CODI source revision `2c2314662c63e9f482ebc46614ffe9af17a241e5`
- Checkpoint `zen-E/CODI-gpt2`
- Checkpoint revision `fd641b3d3edc59e4f534b55588e906588c9e36bb`
- Checkpoint SHA256 `fd223b14932b8b66605c2a688e7d7193058ff2b40b2c6f6c1547c227d06d8417`
- GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Six latent iterations, projection dimension 768, LoRA rank 128
- Raw question input, greedy decoding, maximum 256 generated tokens
- Official benchmark counts
  - GSM8K 1,319
  - SVAMP 1,000, formed from train plus test
  - MultiArith 180
  - GSM-Hard 1,319

The adapter rejects checkpoint hash mismatches, tensor-shape mismatches, low state-dict
coverage, missing projection weights, missing LoRA weights, or benchmark-count drift.

## Run on Colab

Use [`notebooks/colab_official_codi_validation.ipynb`](../notebooks/colab_official_codi_validation.ipynb)
with a GPU runtime. The notebook writes predictions, summaries, manifests, and logs to:

```text
MyDrive/CODI_KAVA/outputs/official_codi_gpt2/
MyDrive/CODI_KAVA/logs/official_codi_gpt2/
```

The notebook runs in three explicit scopes:

1. A 32-example GSM8K loading diagnostic
2. The complete GSM8K primary accuracy gate
3. An opt-in complete four-benchmark evaluation after GSM8K passes

## Run from the command line

Quick diagnostic:

```bash
python -u -m src.eval.official_codi \
  --config configs/official_codi_gpt2.yaml \
  --datasets gsm8k \
  --limit 32 \
  --device cuda
```

Complete primary gate:

```bash
python -u -m src.eval.official_codi \
  --config configs/official_codi_gpt2.yaml \
  --datasets gsm8k \
  --limit 0 \
  --device cuda
```

Complete official benchmark suite:

```bash
python -u -m src.eval.official_codi \
  --config configs/official_codi_gpt2.yaml \
  --limit 0 \
  --device cuda
```

Set `HF_TOKEN` for authenticated Hugging Face downloads. The first run downloads the
406 MB CODI checkpoint and the pinned GPT-2 backbone. The notebook installs
`requirements-official-codi.txt`, which pins the released Transformers, PEFT, Datasets,
Accelerate, Hugging Face Hub, and Safetensors versions while retaining the Colab
runtime's CUDA-compatible Torch build.

## Gate interpretation

The primary gate compares complete GSM8K accuracy with the paper's 43.7 percent result
using the released last-number scorer and an absolute tolerance of three percentage
points. Each prediction is also evaluated with this repository's stricter Decimal-based
numeric exact-match scorer, and both correctness flags are saved in the JSONL output.

- `passed` means the official checkpoint and evaluator are compatible enough to proceed.
- `failed` means generated text, checkpoint loading, dependencies, and protocol must be
  reconciled before any more spectral analysis.
- `diagnostic_only_partial_evaluation` means the run used a limit and cannot establish
  reproduction.

Passing is not a new performance claim. After passing, Stage 1b and Stage 1c must be
recomputed from this checkpoint. A projector learned from the pilot checkpoint must never
be reused because its spectral coordinates are checkpoint-specific.

The follow-up workflow is documented in
[`OFFICIAL_CODI_KV_SUBSPACES.md`](OFFICIAL_CODI_KV_SUBSPACES.md) and implemented in
[`notebooks/colab_official_codi_kv_subspaces.ipynb`](../notebooks/colab_official_codi_kv_subspaces.ipynb).

The OOD outputs are preserved but are not currently formal reproduction gates. The CODI
paper lists 500 MultiArith evaluation examples, whereas the released `test.py` loads the
180-example `ChilleD/MultiArith` test split. That discrepancy must be reconciled before
comparing the generated MultiArith accuracy directly with the paper's table.
