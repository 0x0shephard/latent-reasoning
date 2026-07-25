# Stage 1d warm-started key-projection training

## Evidence entering Stage 1d

Stage 1c established that rank-four key directions were sufficient under every predefined
held-out prediction condition.

- Median rank-four key R² was 0.263.
- Every one of the 864 layer-head-position key groups had positive R² and exceeded its
  shuffled null.
- The median actual-minus-shuffled R² was 0.274.
- Rank four retained 82.7 percent of full-rank key prediction.

Values remained predictable but failed only the low-rank sufficiency condition. Rank four
retained 71.8 percent of full-rank value prediction, below the predefined 80 percent
threshold. Stage 1d therefore tests keys first and does not claim that values are noise.

## Research question

When starting from the same trained CODI model, does position-conditioned rank-four key
supervision transfer mathematical-reasoning performance more efficiently than a full key
target or a random rank-four target?

## Leakage and transfer control

The learned basis is estimated from the seed-zero KaVa calibration statistics. Every
Stage 1d arm starts from the completed seed-one CODI checkpoint. The trained student used
to discover the subspace is therefore not the student used in the downstream comparison.

The exported basis has shape

```text
[layer, head, latent position, key dimension, rank]
```

and is frozen throughout training. An independently sampled orthonormal rank-four basis
is stored in the same artifact as the dimensionality control.

## Four matched arms

| Arm | Additional KV loss | Purpose |
| --- | --- | --- |
| `codi_continue_seed1` | zero | Controls for 10,000 additional optimization steps |
| `key_full_seed1` | full-dimensional key MSE | Tests key supervision without spectral compression |
| `key_rank4_seed1` | learned rank-four key MSE | Tests the Stage 1c signal subspace |
| `key_random_rank4_seed1` | random rank-four key MSE | Controls for dimensionality reduction |

All arms use seed one, the same CODI model weights, a reset optimizer, identical batches,
10,000 steps, learning-rate schedule, CE objectives, hidden loss, R-KV token selection,
and checkpoint/evaluation code. The zero-KV continuation arm still runs teacher KV
extraction and R-KV compression, but multiplies the target loss by zero. This keeps the
teacher-forward and target-selection computation matched.

Projected losses compare basis coefficients rather than reconstructed 64-dimensional
vectors. Averaging over the four coefficients keeps their scale comparable to the
full-key mean-squared error.

## Projection export

This CPU step reuses the completed 5,000-example Stage 1b statistics.

```bash
python scripts/export_kv_projection.py \
  --statistics /content/drive/MyDrive/CODI_KAVA/outputs/stage1b_kv_cross_subspaces \
  --output /content/drive/MyDrive/CODI_KAVA/artifacts/stage1d_key_rank4_projectors.pt \
  --rank 4
```

The artifact records its source-statistics SHA-256, calibration checkpoint and example
count, ridge ratio, random seed, and orthonormality checks.

## Training

Use [`colab_stage1d_key_projection.ipynb`](../notebooks/colab_stage1d_key_projection.ipynb)
on an A100 when possible. Each arm is independently restartable and mirrored to Drive.
The first evaluation is capped at 200 examples per dataset.

The primary comparisons are

1. learned rank four versus random rank four for spectral selection
2. learned rank four versus full keys for target compression
3. learned rank four versus continued CODI for incremental KV benefit

If the learned projection is competitive in the capped evaluation, evaluate all four
completed checkpoints on the full datasets before making a performance claim.

## Interpretation boundary

This is a warm-started target-efficiency ablation, not a replacement reproduction of
CODI or KaVa. A positive result shows that the discovered key subspace transfers across
training seeds and provides useful incremental supervision. A final method claim would
still require end-to-end training under a paper-aligned capacity and training budget.
