# Stage 1c cross-validated reduced-rank KV prediction

## Why Stage 1c is required

Stage 1b produced two different results depending on whether latent-position identity was
preserved.

- The original pooled gate failed. Only 45.1 percent of key groups and 41.0 percent of
  value groups jointly exceeded the shuffled correlation and stability baselines. The
  median split-overlap advantages were approximately zero.
- The position-resolved comparison was strong. Correct pairing beat the shuffled null in
  97.8 percent of key groups and 99.9 percent of value groups. Median canonical-correlation
  advantages were 0.705 for keys and 0.678 for values, while the weaker teacher/student
  split-overlap advantages were 0.490 and 0.486.

The pooled and position-resolved results are not contradictory. Pooling combines six
potentially different student-to-teacher mappings before estimating one subspace.
Position-resolved analysis permits each latent slot to have its own mapping.

Stage 1b is still recorded as a negative result under its predefined pooled gate. Stage 1c
is a new test whose primary position-conditioned analysis and thresholds are fixed before
its held-out prediction results are examined.

## Research question

Can a low-rank linear map learned from student KV states on one split predict the aligned
teacher KV states on an untouched split, when layer, head, and latent position are
preserved?

## Measurement

Stage 1c reuses the complete 5,000-example `statistics.pt` produced by Stage 1b. It does
not reload the model or collect activations.

For each key or value layer-head-position group, it fits

```text
teacher KV ≈ student KV × W
```

on split A and evaluates on split B. It then reverses the splits. The intercept and
low-rank map are learned only from the training split. Held-out performance is reported
as R² relative to predicting the training-split teacher mean.

The ridge reduced-rank solution is calculated for ranks 1, 2, 4, 8, and 16. A full-rank
ridge map provides an upper reference. The same procedure is applied to the within-split
teacher-shuffling null.

## Predefined position-conditioned gate

Rank four is the primary rank. A key or value target passes only if all conditions hold
across the 864 layer-head-position groups.

- At least 60 percent of groups have higher held-out R² than the shuffled null.
- At least 60 percent of groups have positive held-out R².
- The median actual-minus-shuffled held-out R² is at least 0.02.
- The median actual held-out R² is at least 0.05.
- The median rank-four R² is at least 80 percent of the full-rank R².

The pooled analysis remains in the report as a diagnostic negative control, but it does
not determine this gate.

## CPU command

No GPU is required.

```bash
python scripts/analyze_kv_reduced_rank.py \
  --statistics /content/drive/MyDrive/CODI_KAVA/outputs/stage1b_kv_cross_subspaces \
  --output /content/drive/MyDrive/CODI_KAVA/reports/stage1c_kv_reduced_rank.json
```

The command writes both JSON and Markdown reports atomically. It does not alter the
Stage 1b statistics or reports.

## Decision

- Both keys and values pass. Proceed to a small compute-matched training ablation with
  separate keys-only, values-only, and keys-plus-values projected targets.
- Only one target passes. Restrict the first training ablation to that target.
- Held-out prediction is positive but rank four retains too little full-rank signal.
  Increase the candidate rank without claiming low-rank sufficiency.
- Neither target passes. Do not train a TSV-inspired projection with this alignment.

Passing this gate establishes out-of-sample linear predictability, not answer causality
or downstream accuracy. Those require a later compute-matched training experiment.
