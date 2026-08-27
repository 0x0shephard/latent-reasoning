# Test-like correctness detector replication

Corrective replication of the only provisional result in the completed three-track
correctness experiment. This is a new contract; it does not modify or reinterpret the
preregistered gate after seeing new outcomes.

## Question

Does the Fisher correctness score add at least 0.01 test AUC beyond CODI's own output
margin when the detector is fitted and selected on questions drawn from the same
test-like population as its final evaluation?

The original detector was fitted on GSM8K train, where CODI was about 25 percentage
points more accurate than on GSM8K test. This replication uses only the 1,319 cached
GSM8K test states and partitions them before fitting anything.

## Frozen partition

| split | examples | purpose |
|---|---:|---|
| fit | 440 | Fisher direction and logistic weights |
| select | 440 | Fisher shrinkage and ridge strength |
| test | 439 | one final read of the frozen gate |

The partition seed is `20260827`. The runner saves the exact indices and a SHA-256 hash
of the complete partition. Every row comes from `evaluation_states`; use of the old
`calibration_states` field is outside this contract.

This reuses a dataset the project has already inspected, so it is a corrective
replication rather than a pristine preregistration. Its smaller test split will also
produce wider confidence intervals than the original 1,319-example gate.

## Primary gate

Primary probe: `fisher_plus_margin`. Baseline: `margin`.

The result passes only if all three conditions hold:

1. test AUC improvement is at least `+0.01`;
2. the paired question-bootstrap 95% lower bound is above zero;
3. both selected logistic fits carry valid optimization certificates.

The bootstrap uses 10,000 paired resamples and seed `20260827`. The original threshold
is retained unchanged.

## Checked solver

The replication uses deterministic full-batch L-BFGS with strong-Wolfe line search.
Weights and intercept are ridge-regularized, making the complete objective strongly
convex. Each fit exports:

- final objective;
- L2 and infinity norms of the final gradient;
- iteration and function-evaluation counts;
- a strong-convexity upper bound on the objective gap;
- the tolerances and a boolean convergence verdict.

Every ridge candidate must converge. The runner stops instead of selecting among a
partially optimized grid.

## Run

Against the completed margin-geometry cache:

```bash
python scripts/run_official_codi_correctness_detect_replication.py \
  --states colon_states.pt \
  --readout readout.pt \
  --output detect_replication.json

python scripts/analyze_official_codi_correctness_detect_replication.py \
  --sweep detect_replication.json \
  --output detect_replication_report.json
```

The dedicated Kaggle notebook is
[`kaggle_official_codi_correctness_detect_replication.ipynb`](../notebooks/kaggle_official_codi_correctness_detect_replication.ipynb).
It is CPU-only and requires the completed `colon_states.pt` and `readout.pt` exports.

## Interpretation boundary

A pass would replicate the small incremental detector signal on a test-like fitting
population. It would not make the detector operationally useful without a separate
utility analysis. A failure would retire the original `+0.0123` as not robust to the
calibration-population correction. Neither outcome changes the completed steer or
project nulls.

## Status

Implementation and synthetic end-to-end validation are complete. The real cached
states are not stored in this repository, so the 439-example result remains pending.
