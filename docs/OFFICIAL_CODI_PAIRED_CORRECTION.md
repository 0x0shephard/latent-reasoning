# Same-question paired correction at CODI's answer cue

## Question

Can a question-conditioned additive edit learned from correct and wrong state-12
counterfactuals of the **same question** improve CODI's answers?

This repairs the main confound in the correct/wrong covariance experiment. That
experiment compared correct states from some questions with wrong states from other
questions, so correctness was mixed with question content, difficulty, answer value and
confidence. It also retained only 28 dimensions and destroyed the other 740.

## Why ordinary answer sampling is insufficient

State 12 at `The answer is:` is computed before the first answer token is selected.
Sampling several answer tokens from the same logits can produce correct and wrong text,
but every sample has the same pre-token state 12. Their paired state difference is
exactly zero.

The new collection therefore applies seeded, relative-RMS Gaussian perturbations at
state 11 during the forced answer-cue forward pass. State 11 propagates through the last
GPT-2 block and final layer norm, producing genuinely different state-12 vectors before
greedy token selection. The frozen relative-noise schedule is
`[0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60]` plus the unperturbed baseline.

These are local controlled counterfactuals of answer selection, not naturally sampled
alternative reasoning traces. That limitation must accompany any positive result.

## Frozen population

The same deterministic GSM8K-test partition used by the preceding corrective work is
retained:

| split | questions | use |
|---|---:|---|
| fit | 440 | PCA band, paired targets and correction weights |
| select | 440 | ridge, edit strength and confidence gate |
| test | 439 | one final analytic and exact-match read |

No noisy final-test state or noisy final-test correctness label is saved. Each fit or
selection question is eligible only if its frozen variants contain at least one correct
and one wrong greedy first token. Each eligible question contributes one equally
weighted pair:

```text
delta_q = mean(state12_correct | q) - mean(state12_wrong | q)
```

## Conditioned additive intervention

The state-12 accuracy band is fitted from the unperturbed fit states and fixed to PCs
4–31. The target is the paired difference projected into those 28 coordinates. A
multi-output ridge map predicts that target from the incoming state's band coordinates
and its top-two output margin.

At deployment:

```text
h' = h + alpha * U_4:32 * predicted_delta(h, margin)
```

The edit is applied only below a selected margin threshold. Unlike retention, this
preserves all 740 coordinates outside the band. Alpha includes zero, so selection can
honestly choose no intervention.

## Required controls and gate

- `baseline`: no edit;
- `conditioned`: the paired, question-conditioned map;
- `global_mean`: one unconditional average paired correction;
- `shuffled_target`: the same map fitted after permuting correction targets across
  questions.

The analytic gate passes only if the conditioned arm:

1. improves final-test first-token accuracy by at least one percentage point over
   baseline with a positive paired-bootstrap lower bound;
2. beats both the global and shuffled controls with positive lower bounds; and
3. selected a non-zero edit on a non-empty set of questions.

The optional generation tier repeats all four arms with real greedy decoding and
numeric exact-match scoring on the same 439 final questions. The analytic gate remains
primary because it exactly measures the state-12 first-token channel being edited.

This follow-up reuses a GSM8K-test population whose aggregate behavior was inspected
by earlier experiments. The frozen split prevents tuning on its final 439 rows inside
this run, but the result is an exploratory corrective test, not a pristine independent
confirmation.

## Run

Use
[`kaggle_official_codi_paired_correction.ipynb`](../notebooks/kaggle_official_codi_paired_correction.ipynb).
Attach the completed `colon_states.pt`, `readout.pt`, and official reproduction
`summary.json`. The collection and exact-generation tiers require a Kaggle GPU.

The notebook pins the package versions recorded by the official reproduction and
removes Kaggle's incompatible optional TorchAO package before importing PEFT.

## Status

Complete. The run is recorded in ledger §50 and its export is
`jonraza15/same-question-correct-to-wrong-correction-for-codi`.

## Completed result (2026-08-27): `not_confirmed`

All checksums intact; both the analytic gate report and the generation report
recompute bit-identically from the raw export with the local analyzers. The run was
pinned at commit `e67bb20`, used the frozen partition (`c8316e46…`) shared with the
detect replication and the contrastive covariance run, and never saved a noisy
final-test state.

Collection: eight noise levels plus the baseline over the 880 fit/select questions.
First-token correct share stayed between 0.431 and 0.453 at **every** noise level —
even 0.60 relative RMS at state 11 barely moves answer selection. Paired coverage was
therefore thin: 65 fit and 60 select questions produced both outcomes, and the
composition diagnostic shows both transition types present (fit: 25 baseline-correct /
40 baseline-wrong; select: 30 / 30), so the null is not explained by a
denoiser-only training set.

Fitting: the held-out target cosine was ≈ 0.016 at every ridge value (0.0161 at 0.01
down to −0.004 at 100), and MSE selection chose the maximum shrinkage, ridge 100. The
paired wrong-to-correct delta is essentially unpredictable from the incoming state's
band coordinates and margin. Selection then honestly chose α = 0.25 with a 10%
confidence gate for the conditioned arm (+0.45 points on select), and α = 0 — the
no-op — for the global-mean arm.

Final test read, 439 questions:

| arm | analytic first-token | edited fraction | exact match (decoded) |
|---|---:|---:|---:|
| baseline | 0.4009 | — | 0.4237 |
| conditioned | **0.4009** | 0.128 | **0.4237** |
| global_mean (selected α = 0) | 0.4009 | 0.0 | 0.4237 |
| shuffled_target | 0.4032 | 0.510 | 0.4260 |

The conditioned map edited 48 of 439 decoded states (RMS edit norm 0.207) and changed
**zero** predictions on both tiers; gold NLL and margin moved at the fourth decimal.
Gates: primary gain failed (0.00 points), specificity failed, non-trivial intervention
passed (the edit was honestly applied). Every bootstrap interval crosses zero.

Interpretation, bounded as frozen:

- A question-conditioned additive correction — ridge from band coordinates and margin,
  trained on same-question noise counterfactuals — is not a demonstrated correction
  mechanism. With §43 (global constant translation) and §47 (class-conditioned
  projection), every conditioning level of fixed linear state-12 editing tested by
  this project has now returned a null.
- Two independent failure causes were measured, either sufficient alone: controlled
  state-11 perturbations rarely flip the answer (≈ 14% of questions across eight noise
  levels), so pairs are scarce; and the flip direction that does exist is unpredictable
  out of sample from the observables the map is allowed to see.
- This does not rule out corrections conditioned on richer inputs (question text,
  earlier-layer states), non-linear maps, or counterfactuals from genuinely different
  reasoning traces rather than local noise. It rules out the strongest form the
  completed evidence had motivated.
