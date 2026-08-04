# Official CODI parameter-aware endpoint spectral experiment

## Motivation

The completed seed-29 answer-conditioned experiment selected six residual PCs in the
final two transformer-block states, but its candidate update had median answer-gradient
cosine `-0.002559` and failed every held-out control. The activation-local score

\[
(u^\top r)(u^\top \nabla_S L_{\text{answer}})
\]

does not include the model Jacobian. This follow-up tests the narrower hypothesis that
residual PCs become useful when they are selected in the trainable LoRA-parameter
geometry itself.

This remains a TSV-C-inspired endpoint experiment. It is not unchanged weight-space
TSV-C and it does not modify the completed seed-11 or seed-29 artifacts.

## Fresh data contract

The code deterministically reproduces and excludes every normalized question from:

1. the corrected seed-11 endpoint experiment: 5,000 calibration, 256 update, and
   256 validation questions;
2. the seed-29 answer-conditioned experiment: 1,024 residual-fit, 1,024 selection,
   256 update, and 256 validation questions.

It then samples four new seed-41 partitions:

| Partition | Examples | Purpose |
| --- | ---: | --- |
| residual fit | 1,024 | fit fixed residual eigensystems |
| parameter selection | 1,024 | score candidate PC parameter gradients |
| update | 256 | form 64 equal-norm local updates |
| validation | 256 | measure paired held-out numeric-answer NLL |

Every partition contains one row per normalized question and is disjoint from all
other partitions and both completed experiments.

## Candidate residual PCs

Teacher and student states are gathered at the same colon in `The answer is:`. The
residual eigensystem is fitted independently at each of the 13 depth states:

\[
G_s=\frac1N\sum_i r_{i,s}r_{i,s}^{\top}
=U_s\Lambda_sU_s^{\top}.
\]

Based on the observed seed-29 result, the confirmatory candidate inventory is fixed to
the first 64 residual-energy PCs in states 11 and 12, the final two transformer-block
outputs. No earlier state is eligible. This produces 128 hypotheses.

For candidate PC `u`, the rank-one auxiliary loss uses exactly the native
SmoothL1/teacher-standard-deviation scaling applied to the projected residual:

\[
L_u(\theta)
=
\frac{\operatorname{SmoothL1}((u^\top r)u,0)}
{\operatorname{std}(T_s)}.
\]

Its induced parameter gradient is:

\[
g_u=\nabla_\theta L_u,
\qquad
g_A=\nabla_\theta L_{\text{answer}}.
\]

The selection statistic is the parameter-space cosine:

\[
c_u=\frac{g_u^\top g_A}{\lVert g_u\rVert\lVert g_A\rVert}.
\]

The numerator is computed exactly. All candidate numerators are obtained together by
introducing candidate weights `w` and differentiating

\[
\left\langle
\nabla_\theta\sum_u w_uL_u,
g_A
\right\rangle
\]

with respect to `w`.

## Deterministic parameter-norm sketch

Computing 128 complete LoRA gradients per selection minibatch would be unnecessarily
expensive. Candidate norm squares use eight deterministic Rademacher probes:

\[
\lVert g_u\rVert^2
\approx
\frac1K\sum_{k=1}^{K}(g_u^\top z_k)^2,
\qquad K=8.
\]

This is an unbiased Hutchinson estimator of squared norm. Probe seeds, count, candidate
inventory, and every resulting split statistic are stored in the basis artifact. The
smoke run uses fewer examples and candidates but the same code path.

## Split-stable selection

The 1,024 selection examples form 128 disjoint minibatches of eight. Minibatches, not
individual examples, are the registered statistical unit because each cosine is the
geometry of one minibatch parameter update. Even and odd minibatches form two fixed
halves.

A PC is retained only if both halves independently have:

- positive mean cosine;
- one-sample `z >= 1.645`;
- one-sided Benjamini-Hochberg survival at FDR 0.05 across all 128 hypotheses.

At most eight PCs may be retained per state. The shuffled-answer rank-matched basis is
formed from the same PC inventory using answer gradients obtained by pairing each
question/latent trajectory with a deterministically deranged gold answer inside the
selection minibatch.

If no PC survives, the experiment closes before utility with
`no_stable_parameter_aware_directions`.

## Held-out utility gate

If a candidate exists, the untouched update and validation partitions evaluate:

1. answer only;
2. full block target;
3. parameter-aware target;
4. energy-rank-matched target;
5. random-rank-matched target;
6. shuffled-answer-selection-rank-matched target;
7. parameter-aware basis with shuffled teacher pairing;
8. the parameter-aware orthogonal complement.

Every auxiliary gradient is matched to the full-block target gradient norm. Every
combined update is normalized to

\[
10^{-4}\lVert\theta\rVert_2.
\]

The parameter-aware candidate must beat all five primary controls with a positive
paired-update-batch bootstrap lower bound and Holm-adjusted one-sided sign-flip
`p < 0.05`. Its median held-out-answer gradient cosine must also be positive. Only a
complete pass authorizes a separately preregistered training study.

## Implementation

- mechanism: `src/mech/endpoint_parameter_aware.py`
- collection: `scripts/collect_official_codi_endpoint_parameter_aware.py`
- utility: `scripts/run_official_codi_endpoint_parameter_aware_utility.py`
- final analysis: `scripts/analyze_official_codi_endpoint_parameter_aware.py`
- configuration: `configs/official_codi_gpt2.yaml` under
  `endpoint_parameter_aware`
- Kaggle notebook: `notebooks/kaggle_official_codi_endpoint_parameter_aware.ipynb`

## Interpretation boundary

This is a one-step local utility test at one checkpoint. A negative result rejects this
registered residual-PC/Jacobian-sketch construction, not every possible nonlinear or
parameter-aware target. A positive result authorizes a new training experiment; it is
not itself an accuracy or convergence claim.
