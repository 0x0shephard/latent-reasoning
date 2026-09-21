# Trajectory-level supervision with mechanism-selected targets

Preregistered as ledger §84. Contract `official_codi_trajectory_supervision_v1`.

## Question

Ledger §1 asked whether KaVa's KV-trajectory supervision improves on CODI's
endpoint distillation under matched conditions. Ledger §55 established what CODI's
latent trajectory holds: the odd slots (0-based 1, 3, 5) store the solution's
intermediate values, as an unordered set. This experiment asks the §1 question with
that finding as the selector.

> Does supervising the value-holding slots toward the teacher positions that emit
> intermediate values improve on endpoint-only CODI, on KaVa's redundancy-selected
> targets, and on controls that differ only in which positions or which slots are
> supervised?

## What each loss supervises

CODI's only distillation signal is the hidden state at the answer cue, matched to
the teacher's at its own cue after an explicit chain of thought. KaVa keeps that and
adds, for every latent slot, a key/value target taken from the teacher's trace cache
at a position chosen by R-KV, a redundancy-plus-attention score. Neither knows what
the slots are for. §55 does.

| arm | auxiliary target | teacher positions | student slots |
| --- | --- | --- | --- |
| `codi` | none | — | — |
| `kava` | K and V, L1 | R-KV, λ = 0.1, per layer and head | all six |
| `value_odd` | K and V, L1 | first token of each `<<…=v>>` result in the truncated trace | 1, 3, 5 |
| `random_odd` | K and V, L1 | seeded random trace positions, count matched per example | 1, 3, 5 |
| `value_even` | K and V, L1 | as `value_odd` | 0, 2, 4 |
| `recon_odd` | cross-entropy of the slot's own vocabulary readout toward the value's first token | as `value_odd` | 1, 3, 5 |

Because the store is unordered, slot-to-value assignment is a per-example
minimum-cost injective matching on the detached loss, exhaustive over the tiny
candidate sets. Examples with fewer values than slots leave surplus slots
unsupervised; examples with more let the matching choose.

`recon_odd` is the generative-objective comparison. It is also the arm §58 warns
about: it trains the readable component of the slot directly, and §58 showed the
readable component is not the load-bearing one. It is included to test that
warning, not in spite of it.

## Matched conditions

Every arm warm-starts from the frozen official checkpoint and shares:

- **data:** 8,192 GSM8k-Aug training rows (the released training distribution,
  `eq_only`), unique questions, digit-leading answers, at least two equations so the
  truncated trace carries at least one value; 256 further rows for the selection
  split; data seed 20260922; disjointness from GSM8K test verified by normalised
  question;
- **optimisation:** batch 8, one epoch of 1,024 steps, AdamW at 2e-5 constant, weight
  decay 0, gradient clip 1.0, float32, training seeds 1, 2, 3; trainable parameters
  are the released LoRA adapters and the projector, as in §30;
- **base objective:** student gold-answer NLL plus the official endpoint loss,
  smooth-L1 over all 13 states scaled by the teacher standard deviation;
- **auxiliary weighting:** the auxiliary term's gradient is rescaled every step to
  the endpoint term's gradient norm, so arms differ in *what* they supervise, not
  how strongly;
- **teacher path:** frozen, no teacher cross-entropy, for every arm;
- **decoding:** the official native protocol, no forced cue.

## Screen, then one test read

After `codi` seed 1 trains, its accuracy on the selection split must be within
3 points of the frozen checkpoint's on the same split. Failing that means the warm
start damaged the model and the run stops without reading the test set. On pass,
every arm and seed is evaluated once on the full 1,319-question GSM8K test, together
with the frozen checkpoint as reference.

## Gates

Paired bootstrap over questions on per-question seed-mean correctness, 10,000
resamples:

1. **H1, KaVa versus CODI:** `kava − codi` lower bound above zero.
2. **H2, selector:** `value_odd − kava` lower bound above zero **and**
   `value_odd − random_odd` lower bound above zero.
3. **H3, slot specificity:** `value_odd − value_even` lower bound above zero.
4. **H4, objective:** `recon_odd − value_odd`, reported two-sided, no gate.

The headline claim, that mechanism-selected trajectory supervision helps, requires
H2 and `value_odd − codi` above zero. Per-seed accuracies, the fraction of R-KV picks
that land on value tokens, assignment statistics, and gradient-scale records are
reported whatever the gates say.

## Expectations, stated in advance

Modest effects at best. §17 records that R-KV was not better than uniform selection
in the pilot. On equation-only traces the value tokens are plausibly among the least
redundant tokens, so KaVa's selector and the value selector may largely coincide;
the overlap diagnostic measures exactly that. A null here says that, at this budget,
the §1 question is not decided by which trajectory positions are supervised. The
1,024-step warm start is the cheap held-out gate §17 requires before any long
training run.

## Inputs and outputs

Attach the completed official CODI reproduction dataset containing
`official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json`. The checkpoint,
GSM8k-Aug, and GSM8K test download from pinned sources. No artifact from any earlier
experiment is required.

`notebooks/kaggle_codi_trajectory_supervision.ipynb` first runs a three-minute smoke
pass of the whole pipeline on sixteen rows, then the frozen protocol. Outputs under
`/kaggle/working/codi_trajectory_supervision`: `summary.json`, per-run records under
`runs/`, `trajectory_supervision.pt` with per-question correctness, and
`predictions.jsonl` with every model's test generations.

## Files

- implementation: `src/mech/trajectory_supervision.py`
- runner: `scripts/run_codi_trajectory_supervision.py`
- notebook builder: `scripts/build_kaggle_codi_trajectory_supervision_notebook.py`
- notebook: `notebooks/kaggle_codi_trajectory_supervision.ipynb`
- tests: `tests/test_trajectory_supervision.py`,
  `tests/test_trajectory_supervision_runner.py`,
  `tests/test_codi_trajectory_supervision_notebook.py`
