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

Complete. The run is recorded in ledger §49 and its export is
`jonraza15/replication-of-codis-correctness-detector`.

## Completed result (2026-08-27): `test_like_detect_not_supported`

All checksums intact, and the gate report recomputes bit-identically from the raw
sweep export with the local analyzer. The partition SHA matches the frozen test-like
split shared with the contrastive covariance run (fit 44.1% / select 42.0% / test
40.1% correct), so no GSM8K-train state entered any fit.

| probe | features | select AUC | test AUC |
|---|---:|---:|---:|
| **fisher_plus_margin** (primary) | 2 | 0.8664 | **0.8942** |
| margin (baseline to beat) | 1 | 0.8589 | 0.8795 |
| full_state_plus_margin | 769 | 0.8493 | 0.8753 |
| full_state | 768 | 0.8455 | 0.8709 |
| fisher alone | 1 | 0.8288 | 0.8664 |
| mean_difference | 1 | 0.6537 | 0.7216 |

ΔAUC over the margin is **+0.0147** — above the 0.01 magnitude threshold and close to
the original +0.0123 — but the paired-bootstrap CI is **[−0.0080, +0.0389]**, so the
required positive lower bound fails and the gate does not pass.

The second §44 repair is conclusive in the other direction: every probe carries a full
convergence certificate (L-BFGS with strong Wolfe line search, gradient norms at or
below 7×10⁻⁸, strong-convexity objective-gap bounds at or below 8×10⁻¹⁵, every ridge
candidate converged). Probe under-optimization does not explain any reported AUC.

Interpretation under the frozen decision rule: the §43 detect pass is **retired as not
established**. The honest nuance is that this is a power-limited null, not evidence of
zero effect — the point estimate replicated in sign and size on a population where the
model behaves like it does at evaluation, but 439 test examples cannot bound an
increment this small away from zero. Any future attempt to establish it would need a
larger held-out test-like population, and no such attempt is currently planned. The
model's own margin remains the only confirmed correctness signal at the answer cue.
