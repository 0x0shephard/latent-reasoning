# Official CODI endpoint TSV-C-inspired experiment

## Status

| Milestone | State |
| --- | --- |
| Scientific contract preregistered | complete |
| Endpoint extraction and spectral utilities | complete |
| Resumable utility runner and analyzer | complete |
| Kaggle notebook | complete |
| Local unit and static validation | complete |
| Official-checkpoint alignment smoke test | pending Kaggle GPU |
| 5,000-example calibration | pending Kaggle GPU |
| All-layer endpoint utility screen | pending calibration |
| Layer-11 endpoint utility screen | pending calibration |
| Final gate decision | pending both screens |

Runtime progress is stored under `outputs/official_codi_endpoint_tsvc`. Large basis and
batch files are intentionally not committed. This document and
`RESEARCH_CONTEXT_LEDGER.md` preserve the scientific state in Git.

## Motivation

The previous TSV-inspired study decomposed teacher and student key/value trajectories.
It found statistically stable low-rank structure, but learned KV directions were not
more causally useful than energy-matched random directions. Complete and sparse KV
targets also failed the held-out answer-loss utility gates.

CODI does not natively distil those KV tensors. Its published objective matches teacher
and student hidden states at one answer-cue endpoint across all transformer blocks. The
current experiment therefore moves the spectral test to CODI's actual supervision
location.

The original TSV-C method decomposes per-layer weight-difference matrices. This work
adapts only its truncated-SVD principle to activation residual matrices. Every result
must be described as **TSV-C-inspired activation filtering**, not unchanged TSV-C.

## Research question

> At the paper-accuracy official CODI checkpoint, do the leading singular directions of
> teacher-student endpoint hidden-state residuals produce specifically useful answer
> updates beyond answer-only, random, bottom-spectrum, and shuffled-pairing controls?

Two axes must not be confused:

- **Endpoint** is the trajectory-time location. The teacher state is taken at the final
  token of `The answer is:` and the student state after its sixth continuous latent step.
- **Layer** is transformer depth. The primary scope uses all 12 blocks. The secondary
  scope uses block 11 only at that same endpoint.

## Checkpoint and alignment gate

The experiment uses the author-released `zen-E/CODI-gpt2` checkpoint pinned at revision
`fd641b3d3edc59e4f534b55588e906588c9e36bb`. Its complete GSM8K evaluation must pass the
existing reproduction gate near the published 43.7 percent accuracy. A partial or failed
reproduction summary blocks collection.

For each example and block `l`:

```text
teacher_l = hidden state at the final token of "The answer is:"
student_l = hidden state after continuous latent iteration 6
R_l       = student_l - stop_gradient(teacher_l)
```

Hugging Face's embedding-state entry is excluded. The resulting tensors must both be
`[batch, 12, 768]`.

## Frozen calibration contract

- Calibration examples: 5,000
- Update examples: 256
- Held-out validation examples: 256
- Sampling seed: 11
- One row per normalized question
- Calibration, update, and validation question groups are disjoint
- Decomposition: uncentered, independently for every transformer block
- Rank: 77, equal to `ceil(0.10 × 768)`
- Random-basis seed: 20260803
- Precision: float32

For each block, streaming collection stores the uncentered residual Gram matrix
`R_l^T R_l`. Its eigendecomposition is equivalent to obtaining the right singular
vectors of the residual matrix without saving all 5,000 activation rows. The artifact
stores the top 77 directions, bottom 77 directions, a deterministic random orthonormal
rank-77 basis, the full spectrum, checkpoint identity, dataset fingerprint, exact split
indices, and request hashes.

The top directions capture the largest residual energy. This fact alone is descriptive
and cannot pass the experiment. Earlier work showed why stable or high-energy structure
must not be equated with useful reasoning signal.

## Endpoint loss

For a selected orthonormal basis `B_l`, the differentiable filtered residual is

```text
projected_l = R_l B_l B_l^T
```

The loss retains CODI's L1 and teacher-standard-deviation contract:

```text
L_projected = sum_l mean(abs(projected_l)) / std(teacher_l)
```

The complement replaces `projected_l` with `R_l - projected_l`. The full condition uses
`R_l` unchanged. Layers are never pooled when fitting or applying a basis.

## Conditions

Each scope is run in a separate resumable output tree.

1. `answer_only`
2. `full`
3. `learned_top77`
4. `random_rank77`
5. `bottom_rank77`
6. `shuffled_top77`
7. `complement`

The complement is diagnostic because it contains 691 dimensions and is not a matched
rank-77 budget. Full-target comparison distinguishes possible denoising from simple
compression but is not part of the primary four-control family.

## Equal-update-norm utility test

The 256 update examples form 64 batches of four. Each update batch is paired with one
disjoint validation batch of four. For every batch:

1. Compute the student gold-answer gradient.
2. Compute every endpoint auxiliary gradient.
3. Match each auxiliary gradient norm to the native full endpoint-gradient norm.
4. Add the answer and matched auxiliary gradients.
5. Normalize every total update to `1e-4` times the trainable parameter norm.
6. Apply the update statelessly with `torch.func.functional_call`.
7. Measure held-out gold-answer NLL on the paired validation batch.

The official checkpoint is never overwritten. The bootstrap and randomization unit is
the paired update batch, not the individual validation example.

## Predefined decision rule

The primary `endpoint_all_layers` gate passes only if `learned_top77` beats all four:

- answer-only
- random rank 77
- bottom rank 77
- shuffled top 77

For every comparison, the paired 10,000-sample bootstrap 95 percent lower bound must be
positive and its Holm-adjusted one-sided paired sign-flip p-value must be below 0.05.
The median learned-subspace gradient cosine with the held-out answer gradient must also
be positive.

The `endpoint_layer11` scope uses the same rule as a secondary localization test.

- Primary pass authorizes a separately preregistered compute-matched training study.
- Primary fail and secondary pass requires a fresh last-layer confirmation first.
- Both fail close this endpoint TSV-C definition without expensive training.

No rank, seed, scope, sample size, or gate threshold may be changed after results are
observed and then presented as confirmatory.

## Interpretation boundary

This is a local one-step optimization test at a final, already-trained checkpoint. A
positive gate means the learned endpoint subspace produces more useful local parameter
updates than the preregistered controls. It does not prove that the directions encode a
human-readable reasoning algorithm or that using them throughout training improves
exact-match accuracy.

A negative result applies only to this linear, rank-77, endpoint-residual definition. It
does not rule out nonlinear, example-conditional, or earlier-training-stage signal.

## Execution

Use `notebooks/kaggle_official_codi_endpoint_tsvc.ipynb`. The notebook verifies the
official accuracy gate, fits or resumes the basis, runs both scopes, combines the gate,
and exports checksummed artifacts. The two utility trees can resume independently from
an attached Kaggle output dataset.

Principal commands:

```bash
python -u scripts/collect_official_codi_endpoint_tsvc.py ...
python -u scripts/run_official_codi_endpoint_tsvc_utility.py --scope endpoint_all_layers ...
python -u scripts/run_official_codi_endpoint_tsvc_utility.py --scope endpoint_layer11 ...
python scripts/analyze_official_codi_endpoint_tsvc.py ...
```
