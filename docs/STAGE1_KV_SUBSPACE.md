# Stage 1 research protocol

## Objective

Determine whether the teacher-to-student KV residuals in the trained KaVa checkpoint
contain reproducible low-rank signal subspaces before spending compute on any new
distillation run.

This stage is calibration-only. It does not update model weights and it does not claim
that a discovered subspace causes higher answer accuracy.

## Exact object being measured

For every calibration example, transformer layer, KV head, and latent position, the
workflow constructs

```text
key residual   = compressed teacher key   - student latent key
value residual = compressed teacher value - student latent value
```

The teacher target is selected by the repository's existing R-KV implementation. Selected
trace tokens are sorted chronologically and paired with student latent positions zero
through five. Layers and KV heads are paired by identical index because both paths use
the same trained backbone.

Keys and values are analyzed independently. Statistics are kept at two granularities.

- Pooled positions produce one residual matrix for each layer and head.
- Position-resolved statistics produce one residual matrix for every layer, head, and
  latent position.

## Storage and decomposition

The default workflow does not retain the full activation matrix. It streams the exact
sufficient statistics for centered covariance

```text
number of rows
sum of rows
sum of row outer products
```

The covariance eigenvectors are exactly the right singular vectors that would be obtained
from an SVD of the centered residual matrix. For GPT-2, this reduces the durable artifact
from multiple gigabytes of raw teacher and student caches to roughly a few hundred
megabytes of restartable statistics. Float16 residual shards can be enabled when the raw
calibration residuals are needed for follow-up analyses.

## Stability and null controls

Examples are assigned deterministically to independent split halves. The analysis compares
their eigenvalue spectra and top-r subspaces using explained variance, effective rank,
spectrum cosine similarity, principal angles, and projection overlap.

Two nulls are accumulated from the same forward passes.

1. The shuffled null deranges teacher targets across examples within each batch. It
   preserves the teacher and student marginals while destroying correct example pairing.
2. The random null uses isotropic Gaussian residuals with exactly the same energy for
   each layer, head, and position. It measures the stability expected from dimensionality,
   sample size, and scale alone.

The predefined rank-four diagnostic gate is positive for a key or value subspace only
when all of the following hold across pooled layer-head groups.

- At least 60 percent of groups exceed both nulls in split overlap and exceed the random
  null in explained variance.
- Median split-overlap advantage over the stronger null is at least 0.10.
- Median rank-four explained-variance advantage over the random null is at least 0.05.

This gate is a transparent screening rule. The full spectra and position-level results
remain the primary evidence.

## Recommended execution

Use the completed seed-zero KaVa checkpoint at step 96,405. Start with 2,000 examples.
If the extraction and report pass validation, extend the same output directory to 5,000
examples. Sampling uses a deterministic permutation prefix, so the larger run continues
from the existing 2,000 examples rather than repeating them.

### GPU extraction

```bash
python -u scripts/collect_kv_subspaces.py \
  --config configs/kava.yaml \
  --checkpoint-root /content/drive/MyDrive/CODI_KAVA/outputs/kava \
  --output-dir /content/drive/MyDrive/CODI_KAVA/outputs/stage1_kv_subspaces \
  --examples 2000 \
  --batch-size 4 \
  --num-splits 2 \
  --save-every 250 \
  --precision auto
```

Re-run the identical command after an interrupted session. The script resumes from
`collection_state.pt`. To extend a completed pilot, change only `--examples 2000` to
`--examples 5000`.

Use `--save-residual-shards` only if the raw residuals are required. That choice is part
of the resume identity and cannot be changed halfway through one output directory.

### CPU spectral analysis

```bash
python scripts/analyze_kv_subspaces.py \
  --statistics /content/drive/MyDrive/CODI_KAVA/outputs/stage1_kv_subspaces \
  --output /content/drive/MyDrive/CODI_KAVA/reports/stage1_kv_subspaces.json
```

This step does not require a GPU. It writes JSON with every layer, head, position, and
null comparison, plus a compact Markdown report.

## Durable outputs

```text
outputs/stage1_kv_subspaces/
  collection_state.pt        present only while extraction is incomplete
  statistics.pt              complete streaming sufficient statistics
  collection_manifest.json   checkpoint, data, sample, alignment, and null provenance
  residual_shards/           optional float16 residual shards

reports/
  stage1_kv_subspaces.json    complete spectral results
  stage1_kv_subspaces.md      compact reader-facing result
```

## Decision after Stage 1

- A positive, reproducible gate justifies Stage 2 experiments that project or weight
  distillation targets using the stable directions.
- Stability only in values or only in keys narrows Stage 2 to that target type.
- Stability that is no better than shuffled pairing indicates that the dominant low-rank
  structure is marginal rather than example-specific.
- Failure against both nulls is still a useful result. It prevents an expensive retraining
  program based on an unsupported spectral assumption.
