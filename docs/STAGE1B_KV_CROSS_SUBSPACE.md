# Stage 1b paired KV cross-subspace protocol

## Why Stage 1b is required

The first 2,000-example analysis found highly stable low-rank residual covariance.
Rank-four split overlap was approximately 0.99 for both keys and values. The same
directions remained after cross-example teacher shuffling, however. Residual covariance
contains

```text
Cov(teacher) + Cov(student) - Cov(teacher, student) - Cov(student, teacher)
```

Shuffling removes the paired cross terms but leaves both large marginal covariance terms.
The first result therefore established stable marginal geometry, not stable
example-specific transfer.

Stage 1b measures the teacher–student cross term directly.

## Research question

After controlling for the teacher and student marginal covariance, do correctly paired
teacher and student KV trajectories share reproducible low-dimensional directions that
disappear when teacher examples are shuffled?

## Measurement

For every key and value layer-head group, the workflow accumulates

```text
count
sum(teacher)
sum(student)
sum(teacher teacherᵀ)
sum(student studentᵀ)
sum(teacher studentᵀ)
```

Statistics are retained after pooling six aligned positions and independently at each
position. The teacher tokens use the same R-KV selection and chronological alignment as
KaVa training.

The analysis centers all moments and constructs

```text
W = Cov(teacher)⁻¹ᐟ² Cov(teacher, student) Cov(student)⁻¹ᐟ²
```

A ridge ratio of `1e-4` stabilizes inverse square roots. SVD of `W` yields regularized
canonical correlations and paired teacher/student coefficient directions.

## Independent splits and shuffled null

Whole extraction batches alternate between two independent split halves. Every shuffle
therefore remains inside one split and cannot leak examples between stability partitions.

For each batch, four seeded teacher derangements are paired with the unchanged students.
Their moments are pooled into a lower-variance shuffled null. This preserves the teacher
and student distributions but removes correct example pairing.

The analysis compares the actual and shuffled canonical correlations as well as the
split-half projection overlap of both teacher and student canonical directions.

## Predefined gate

At rank four, a key or value target passes only when all conditions hold across pooled
layer-head groups.

- At least 60 percent of groups have stronger canonical correlation and more stable
  teacher and student directions than the shuffled null.
- The median paired-minus-shuffled canonical-correlation advantage is at least 0.05.
- The median of the weaker teacher/student split-overlap advantages is at least 0.10.

A positive gate supports a low-rank distillation ablation. It does not establish answer
causality or improved accuracy.

## GPU extraction

```bash
python -u scripts/collect_kv_cross_subspaces.py \
  --config configs/kava.yaml \
  --checkpoint-root /content/drive/MyDrive/CODI_KAVA/outputs/kava \
  --output-dir /content/drive/MyDrive/CODI_KAVA/outputs/stage1b_kv_cross_subspaces \
  --examples 2000 \
  --batch-size 4 \
  --num-splits 2 \
  --shuffle-repeats 4 \
  --save-every 500 \
  --seed 0 \
  --precision auto
```

Re-running the same command resumes atomically. A later 5,000-example confirmation can
extend the same deterministic sample prefix by changing only `--examples`.

## CPU analysis

```bash
python scripts/analyze_kv_cross_subspaces.py \
  --statistics /content/drive/MyDrive/CODI_KAVA/outputs/stage1b_kv_cross_subspaces \
  --output /content/drive/MyDrive/CODI_KAVA/reports/stage1b_kv_cross_subspaces.json
```

The analysis produces complete JSON and compact Markdown reports. A GPU is not required
after cross-moment extraction.

## Decision

- Both keys and values pass. Proceed to a keys-versus-values low-rank distillation
  ablation.
- Only one target passes. Restrict the first training experiment to that target.
- Correlations exceed shuffle but directions are unstable. Increase calibration data or
  improve regularization before training.
- Neither target passes. Do not pursue TSV-style KV projection in this setup. The stable
  geometry found in Stage 1 is marginal rather than paired.
