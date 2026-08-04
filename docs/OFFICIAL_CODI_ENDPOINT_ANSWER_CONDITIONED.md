# Official CODI answer-conditioned endpoint spectral experiment

## Purpose

The completed corrected rank-77 experiment established two facts:

1. teacher-student residuals at the answer-cue colon are strongly low-rank;
2. the leading residual-energy directions do not produce specifically useful answer
   updates beyond answer-only, random, bottom, or shuffled controls.

This follow-up does not tune rank 77 on the completed validation set. It starts with
fresh partitions, excludes every normalized question used by the completed seed-11
experiment, removes the non-contextual embedding state from the primary target, and
selects residual PCs by answer-gradient alignment rather than residual energy.

This remains a **TSV-C-inspired activation experiment**, not unchanged weight-space
TSV-C.

## Endpoint and states

Teacher and student are both gathered at the colon in `The answer is:`:

```text
teacher = question + explicit CoT + "The answer is:" + answer + EOS
student = question + BOT + six latent steps + EOT + "The answer is:" + answer + EOS
```

The endpoint is one sequence position. The 13 tensors are depth snapshots at that same
position. State 0 is the embedding/positional output and states 1 through 12 are the 12
GPT-2 transformer-block states. State 0 is collected for diagnostics but is ineligible
for selection or utility.

## Four disjoint data roles

The full run uses one eligible row per normalized question:

| Partition | Examples | Purpose |
| --- | ---: | --- |
| residual fit | 1,024 | fit the per-state residual eigensystems |
| direction selection | 1,024 | score fixed residual PCs using answer gradients |
| update | 256 | form 64 local parameter updates |
| validation | 256 | measure paired held-out answer NLL |

Before sampling, the code exactly reproduces the completed corrected experiment's
seed-11 partitions (5,000 calibration, 256 update, 256 validation) and excludes all
5,512 normalized questions. The new sampling seed is 29.

## Residual eigensystem

On the residual-fit partition, for state `s`:

\[
r_{i,s}=S_{i,s}-\operatorname{sg}(T_{i,s})
\]

\[
G_s=\frac{1}{N}\sum_i r_{i,s}r_{i,s}^{\top}
=U_s\Lambda_sU_s^{\top}
\]

The full 768-dimensional residual PC basis is retained temporarily. No rank is chosen
from residual energy.

## Answer-conditioned score

On the separate direction-selection partition, differentiate the official numeric
answer NLL with respect to every raw decoder hidden-state tensor, then gather the
gradient at the same colon:

\[
a_{i,s}=\frac{\partial L_{\mathrm{answer},i}}{\partial S_{i,s}}
\]

For residual PC `u_{s,j}`, the per-example selection statistic is:

\[
x_{i,s,j}
=
(u_{s,j}^{\top}r_{i,s})
\,(u_{s,j}^{\top}a_{i,s})
\]

Positive expectation means that, under the local squared-loss approximation, the
teacher-matching activation gradient and answer-loss gradient point in compatible
directions. This is a screening proxy; the later parameter-update experiment remains
the decisive test.

The selection partition is deterministically divided by example order into even and
odd halves. A direction is retained only when both halves satisfy:

\[
\bar{x}_{s,j}>0
\qquad\text{and}\qquad
z_{s,j}=\frac{\bar{x}_{s,j}}{\operatorname{SE}(x_{s,j})}\ge1.645
\]

The one-sided normal-approximation p-values are then Benjamini-Hochberg corrected at
FDR 0.05 independently within each half across all 9,216 eligible block-direction
hypotheses. A direction must survive both corrected halves. At most 64 directions may
be retained per block. Rank is therefore dynamic and state-specific. If no block
contains a qualifying direction, the experiment ends with
`no_stable_answer_conditioned_directions`; utility is not run.

The split-z/FDR rule is a fixed screening boundary, not the final confirmatory test.
Confirmation occurs only on the untouched update/validation partitions.

## Matched bases and utility arms

For every selected block rank `r_s`, the artifact stores four orthonormal bases:

1. **answer conditioned** — residual PCs passing the split-stable answer score;
2. **energy rank matched** — the first `r_s` residual-energy PCs;
3. **random rank matched** — deterministic random orthonormal directions;
4. **shuffled-answer rank matched** — residual PCs ranked using answer gradients
   deranged within selection batches.

The held-out utility runner evaluates:

1. answer only;
2. full block target;
3. answer-conditioned target;
4. energy-rank-matched target;
5. random-rank-matched target;
6. shuffled-answer-rank-matched target;
7. answer-conditioned basis with shuffled teacher pairing;
8. answer-conditioned orthogonal complement.

Every auxiliary parameter gradient is norm-matched to the full block-target gradient.
Every combined parameter update is then normalized to:

\[
10^{-4}\lVert\theta\rVert_2
\]

The checkpoint is evaluated statelessly and never mutated.

## Decision rule

The answer-conditioned target must beat all five primary controls:

- answer only;
- energy rank matched;
- random rank matched;
- shuffled-answer rank matched;
- shuffled teacher.

For every comparison:

- the paired update-batch bootstrap 95% lower bound must be positive;
- the Holm-adjusted one-sided paired sign-flip p-value must be below 0.05.

The median candidate/held-out-answer gradient cosine must also be positive. Only this
complete pass produces `answer_conditioned_training_authorized`. A pass authorizes a
separately preregistered training study; it is not an accuracy claim.

## Implementation

- configuration: `configs/official_codi_gpt2.yaml` under
  `endpoint_answer_conditioned`
- mechanisms: `src/mech/endpoint_answer_conditioned.py`
- collection: `scripts/collect_official_codi_endpoint_answer_conditioned.py`
- utility: `scripts/run_official_codi_endpoint_answer_conditioned_utility.py`
- analysis: `scripts/analyze_official_codi_endpoint_answer_conditioned.py`
- Kaggle notebook: `notebooks/kaggle_official_codi_endpoint_answer_conditioned.ipynb`

## Interpretation limits

The selection score is activation-local and uses the residual-PC coordinate system. It
does not include the full parameter Jacobian and does not prove causality. The utility
screen addresses that limitation with actual parameter gradients and held-out answer
loss, but remains a one-step diagnostic at one checkpoint. Neither a positive nor a
negative result should be generalized to nonlinear subspaces, other checkpoints, or
long-run training without a new registered experiment.
