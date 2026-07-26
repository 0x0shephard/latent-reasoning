# Official CODI KV subspace analysis

## Research purpose

The official `zen-E/CODI-gpt2` checkpoint reproduces the paper's complete GSM8K result
at 43.67 percent. This workflow therefore repeats the teacher–student KV diagnostic on a
competent, externally released CODI model rather than the undertrained pilot checkpoint.

The diagnostic asks whether the model's explicit-CoT teacher KV trajectory and latent
student trajectory contain example-paired, split-stable low-rank correspondence. It does
not claim that CODI was trained with KV supervision. R-KV is introduced only after
loading the checkpoint as an analysis-time method for choosing six teacher trace tokens.

## Sequence and alignment contract

The collector mirrors the released GPT-2 `icot` training path at source revision
`2c2314662c63e9f482ebc46614ffe9af17a241e5`.

- The question is used without a new prompt.
- The final whitespace-separated equation token is removed from the explicit CoT.
- Question, CoT, and answer are tokenized separately with the released 256-token cap.
- The teacher sees the question, retained CoT, `The answer is: N`, and EOS.
- The student sees the left-padded question and BOT followed by six recurrent projected
  latent states.
- R-KV scores explicit trace tokens using 10 percent answer attention and 90 percent key
  novelty.
- Six selected teacher positions are sorted chronologically.
- Teacher and student vectors are paired only at the same layer and KV head, with
  selected teacher positions aligned to latent positions zero through five.

The one-batch audit rejects shape mismatches, invalid selections, non-finite tensors, a
failed public-checkpoint accuracy gate, or checkpoint revision drift.

## Statistical design

The primary collection uses 2,000 seeded calibration examples. Whole extraction batches
alternate between two independent halves. For each layer, head, position, and KV kind,
the collector stores sufficient statistics for:

- the correctly paired teacher and student vectors
- four seeded, within-batch teacher derangements that preserve both marginals while
  destroying example identity

Stage 1b whitens teacher–student cross-covariance and tests canonical correlation and
split-stable subspaces against the derangement null. Stage 1c fits reduced-rank linear
maps on one half and evaluates held-out prediction on the untouched half. Position
identity remains explicit because pooling positions can mix distinct subspaces.

The unchanged preregistered rank-four gates require at least 60 percent of matched groups.
Stage 1b uses a canonical-correlation margin of 0.05 and a split-overlap margin of 0.10.
Stage 1c requires a held-out R-squared margin of 0.02 over shuffling, median held-out
R-squared of at least 0.05, and at least 80 percent retention of the full-rank map.

The independent confirmation uses 5,000 examples at seed one. It should be enabled only
after the primary result has been recorded.

## Run on Colab

Use
[`notebooks/colab_official_codi_kv_subspaces.ipynb`](../notebooks/colab_official_codi_kv_subspaces.ipynb)
with an A100 runtime. The complete GSM8K reproduction summary from
`colab_official_codi_validation.ipynb` must already exist in Drive.

The notebook runs:

1. dependency and unit-test checks
2. a one-batch alignment audit without allocating moment tensors
3. the 2,000-example seed-zero primary collection
4. the Stage 1b and Stage 1c CPU analyses
5. an opt-in 5,000-example seed-one confirmation

Durable artifacts are written to:

```text
MyDrive/CODI_KAVA/outputs/official_codi_kv_subspaces/
MyDrive/CODI_KAVA/reports/official_codi_kv_subspaces/
MyDrive/CODI_KAVA/logs/official_codi_kv_subspaces/
```

Incomplete collection state is saved atomically every 500 examples. Rerunning the same
cell with the same commit, seed, and configuration resumes from the durable prefix.

## Command-line equivalent

```bash
python -u scripts/collect_official_codi_kv_subspaces.py \
  --config configs/official_codi_gpt2.yaml \
  --reproduction-summary /path/to/full_gsm8k/summary.json \
  --output-dir /path/to/official_codi_kv_subspaces/n2000_seed0 \
  --examples 2000 \
  --batch-size 16 \
  --shuffle-repeats 4 \
  --save-every 500 \
  --precision bfloat16 \
  --device cuda \
  --seed 0

python scripts/analyze_kv_cross_subspaces.py \
  --statistics /path/to/official_codi_kv_subspaces/n2000_seed0 \
  --output /path/to/reports/official_codi_n2000_seed0_cross_subspace.json

python scripts/analyze_kv_reduced_rank.py \
  --statistics /path/to/official_codi_kv_subspaces/n2000_seed0 \
  --output /path/to/reports/official_codi_n2000_seed0_reduced_rank.json
```

## Interpretation boundary

A positive Stage 1c gate means a small position-conditioned linear subspace predicts
teacher KV information beyond shuffled example pairing in a paper-accuracy CODI
checkpoint. It does not establish answer causality or a downstream accuracy gain.
Projection versus full-target versus random-projection training under a matched compute
budget would still be necessary.

A negative result means this R-KV alignment and linear diagnostic did not isolate stable
transferable correspondence. It should not be reinterpreted as evidence that all latent
reasoning representations are unstructured.
