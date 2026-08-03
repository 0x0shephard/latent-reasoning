# Historical CODI pre-cue TSV-C-inspired diagnostic

> **Alignment correction recorded 2026-08-03.** This completed run paired the
> teacher's answer-cue colon with the student's sixth latent state *before* EOT and
> `The answer is:`. It therefore did **not** test the source-native CODI endpoint
> loss. Its artifacts and negative result remain valid only for that cross-location
> diagnostic. The corrected experiment is preregistered in
> `OFFICIAL_CODI_ENDPOINT_TSVC_CORRECTED.md` and uses a separate output tree.

## Status

| Milestone | State |
| --- | --- |
| Scientific contract preregistered | complete |
| Endpoint extraction and spectral utilities | complete |
| Resumable utility runner and analyzer | complete |
| Kaggle notebook | complete |
| Local unit and static validation | complete |
| Official-checkpoint alignment smoke test | complete |
| 5,000-example calibration | complete |
| All-layer endpoint utility screen | complete, gate failed |
| Layer-11 endpoint utility screen | complete, gate failed |
| Final diagnostic decision | pre-cue cross-location TSV-C not supported |

Runtime progress is stored under `outputs/official_codi_endpoint_tsvc`. Large basis and
batch files are intentionally not committed. This document and
`RESEARCH_CONTEXT_LEDGER.md` preserve the scientific state in Git.

The completed, checksummed Kaggle artifact is
[`jonraza15/official-codi-endpoint-tsv-c-inspired-experiment`](https://www.kaggle.com/datasets/jonraza15/official-codi-endpoint-tsv-c-inspired-experiment),
version 1, produced from repository commit
`8b70b0d95e6b28ce3dfc512929bd0ac942f8a427`. All files listed in its
`SHA256SUMS.txt` were independently verified after download on 2026-08-03.

## Completed historical diagnostic result

Both prespecified scopes failed their gates. The learned top-rank-77 endpoint
directions did not produce lower held-out answer loss than all required controls.

| Scope | Comparison | Mean learned advantage | 95% paired-bootstrap CI | Holm p |
| --- | --- | ---: | ---: | ---: |
| All layers | versus answer only | -0.000466 | [-0.001758, +0.000791] | 1.0000 |
| All layers | versus random rank 77 | -0.000302 | [-0.001253, +0.000385] | 1.0000 |
| All layers | versus bottom rank 77 | -0.000988 | [-0.002336, +0.000094] | 1.0000 |
| All layers | versus shuffled top 77 | -0.000032 | [-0.000129, +0.000052] | 1.0000 |
| Layer 11 | versus answer only | +0.000215 | [-0.000035, +0.000526] | 0.3508 |
| Layer 11 | versus random rank 77 | +0.000035 | [-0.000084, +0.000149] | 0.8696 |
| Layer 11 | versus bottom rank 77 | +0.000060 | [-0.000338, +0.000449] | 0.8696 |
| Layer 11 | versus shuffled top 77 | -0.000026 | [-0.000244, +0.000143] | 0.8696 |

The primary all-layer median cosine with the answer gradient was -0.005703. The
secondary layer-11 cosine was +0.000565, but none of its required loss comparisons
passed. Learned top-77 was also indistinguishable from the full target in both scopes.

Decision for this historical alignment only:

- do not launch the proposed compute-matched distillation training
- close this fixed rank-77 teacher-colon versus student-pre-cue definition
- retain the result as a preregistered negative finding
- do not tune rank, scope, or thresholds on these same samples and relabel the result
  as confirmatory

## Motivation

The previous TSV-inspired study decomposed teacher and student key/value trajectories.
It found statistically stable low-rank structure, but learned KV directions were not
more causally useful than energy-matched random directions. Complete and sparse KV
targets also failed the held-out answer-loss utility gates.

CODI does not natively distil those KV tensors. This historical experiment attempted to
move the spectral test toward CODI's supervision location, but it incorrectly used the
student state immediately after latent step six. A later source audit showed that the
released objective first decodes EOT and `The answer is:` and then matches student and
teacher at the colon. The corrected experiment implements that exact alignment.

The original TSV-C method decomposes per-layer weight-difference matrices. This work
adapts only its truncated-SVD principle to activation residual matrices. Every result
must be described as **TSV-C-inspired activation filtering**, not unchanged TSV-C.

## Research question

> At the paper-accuracy official CODI checkpoint, do leading singular directions of the
> teacher-colon versus student-pre-cue latent-six residual produce specifically useful
> answer updates beyond matched controls?

Two axes must not be confused:

- **Historical alignment** used the teacher at the final token of `The answer is:` and
  the student immediately after its sixth continuous latent step. These are different
  sequence locations and must not be called CODI's native matched endpoint.
- **Layer** is transformer depth. The primary scope uses all 12 blocks. The secondary
  scope uses block 11 only under the same historical cross-location pairing.

## Checkpoint and alignment gate

The experiment uses the author-released `zen-E/CODI-gpt2` checkpoint pinned at revision
`fd641b3d3edc59e4f534b55588e906588c9e36bb`. Its complete GSM8K evaluation must pass the
existing reproduction gate near the published 43.7 percent accuracy. A partial or failed
reproduction summary blocks collection.

For each example and block `l`:

```text
teacher_l = hidden state at the final token of "The answer is:"
student_l = hidden state after continuous latent iteration 6, before EOT/answer cue
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

The historical runner used an L1 teacher-standard-deviation-normalized diagnostic. The
released CODI GPT-2 run instead used SmoothL1, so this was another reason it could not
serve as a native-loss parity test:

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
