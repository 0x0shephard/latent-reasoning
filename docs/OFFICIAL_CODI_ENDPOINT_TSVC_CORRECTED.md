# Corrected official CODI answer-cue endpoint TSV-C experiment

## Status

| Milestone | State |
| --- | --- |
| Pinned official source audited | complete |
| Historical alignment error isolated | complete |
| Corrected student and teacher colon extraction | implemented |
| Native loss and gradient parity gate | implemented, Kaggle execution pending |
| Rank-77 calibration | implemented, execution blocked on parity |
| All-state primary utility screen | implemented, execution blocked on calibration |
| Layer-11 secondary utility screen | implemented, execution blocked on calibration |
| Combined decision | pending |

The completed historical run remains under `outputs/official_codi_endpoint_tsvc`.
Corrected artifacts use `outputs/official_codi_endpoint_tsvc_corrected`, so rerunning the
new contract cannot overwrite the old evidence.

## Why correction was necessary

The initial implementation paired two different sequence locations:

- teacher hidden states at the colon in `The answer is:`
- student hidden states immediately after the sixth continuous latent step

The pinned official CODI source does not distil those two locations. After generating
six latent states, it feeds the student EOT and answer sequence, finds the same answer
prompt, subtracts one from the first-answer-token index, and gathers the student hidden
state at the colon. It then pairs that state with the teacher colon.

The released loop also uses the complete Hugging Face hidden-state tuple. For GPT-2 this
means the embedding output plus 12 transformer-block outputs. It uses SmoothL1 loss,
divides each tuple-entry loss by the teacher activation standard deviation, and averages
over all 13 entries. The earlier diagnostic excluded the embedding state and used L1.

Pinned evidence:

- official endpoint gather and all-state loop in
  [`src/model.py`](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py#L306-L344)
- official GPT-2 run settings in
  [`train_gpt2_gsm8k-aug.sh`](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/scripts/train_gpt2_gsm8k-aug.sh)

Therefore its negative result cannot answer whether spectral filtering helps CODI's
native endpoint objective.

## Corrected research question

> At the paper-accuracy official CODI checkpoint, do the leading rank-77 singular
> directions of the source-native teacher-student answer-cue endpoint residual produce
> locally useful answer updates beyond answer-only, random, bottom-spectrum, and
> shuffled-pairing controls?

The method remains **TSV-C-inspired activation filtering**. TSV-C originally operates
on weight-difference matrices. Here its truncated-SVD idea is applied to paired
activation residuals, one hidden-state tuple entry at a time.

## Source-faithful endpoint contract

For each example, both sequences are reconstructed using the pinned CODI training
format.

```text
teacher = question + explicit CoT + "The answer is:" + answer + EOS
student = question + BOT + six latent steps + EOT + "The answer is:" + answer + EOS
```

The gathered endpoint is the colon token in both sequences. For hidden-state tuple
entry `s`:

```text
T_s = stop_gradient(teacher hidden state at teacher colon)
S_s = student hidden state at student colon after six latents and EOT
R_s = S_s - T_s
```

The primary scope `endpoint_all_states` uses tuple entries 0 through 12. Entry 0 is the
embedding state and entries 1 through 12 are transformer blocks 0 through 11. The
secondary scope `endpoint_layer11` uses tuple entry 12 only. Bases are always fitted
independently; states are never pooled.

## Native parity gate

Before writing any calibration moment, four deterministic calibration examples must
pass all of the following:

- teacher and student endpoint token IDs are identical
- teacher and student tensors are finite and shaped `[B, 13, 768]`
- teacher states are detached
- the direct released-loss reconstruction and the generic full-target loss differ by
  at most `1e-7`
- their trainable-parameter gradients have relative L2 error at most `1e-6`
- their gradient cosine is at least `0.999999`

The released objective reconstructed by the gate is

```text
L_native = mean_s SmoothL1(S_s, T_s) / std_unbiased(T_s)
```

A failed parity gate blocks calibration and every downstream claim.

## Frozen calibration contract

- Official checkpoint `zen-E/CODI-gpt2`
- Checkpoint revision `fd641b3d3edc59e4f534b55588e906588c9e36bb`
- Required full GSM8K reproduction accuracy near 43.67 percent
- 5,000 calibration questions
- 256 update questions
- 256 paired held-out validation questions
- normalized-question-disjoint partitions
- sampling seed 11
- rank 77, equal to `ceil(0.10 × 768)`
- random orthonormal basis seed 20260803
- uncentered per-state residual Gram eigendecomposition
- no pooling across hidden-state tuple entries

For each state, the collector accumulates `R_s^T R_s`. The artifact stores top, bottom,
and deterministic random rank-77 bases, the full spectra, exact partitions, checkpoint
and dataset identities, request hashes, and the parity report.

## Held-out equal-update-norm utility screen

The 256 update examples form 64 batches of four and are paired with 64 disjoint
validation batches. Both scopes evaluate:

1. answer-only update
2. native full endpoint target
3. learned top-rank-77 target
4. random orthonormal rank-77 target
5. bottom rank-77 target
6. learned top-rank-77 target with shuffled teacher pairing
7. orthogonal complement as a non-budget-matched diagnostic

For each batch, the runner computes the gold-answer and auxiliary gradients, matches
every auxiliary norm to the native full-endpoint gradient norm, combines gradients,
normalizes every total update to `1e-4 × ||theta||`, applies it statelessly, and measures
gold-answer NLL on the paired held-out batch. The official checkpoint is never changed.

## Predefined decision rule

The primary gate passes only if learned top-rank-77 beats answer-only, random rank 77,
bottom rank 77, and shuffled teacher pairing. Every paired-bootstrap 95 percent lower
bound must be positive and every Holm-adjusted one-sided paired sign-flip p-value must
be below 0.05. The median cosine with the held-out answer gradient must also be positive.

- Primary all-state pass authorizes a separately preregistered training experiment.
- Primary fail with layer-11 pass requires a fresh layer-11 confirmation.
- Both fail close this corrected fixed-rank endpoint definition.

Comparison with the full target is reported separately to distinguish compression from
denoising. It is not one of the four primary controls.

## Implementation and execution

- Configuration `configs/official_codi_gpt2.yaml` under `endpoint_tsvc_corrected`
- Extraction and losses `src/mech/official_codi_target_utility.py` and
  `src/mech/endpoint_tsvc_corrected.py`
- Collection `scripts/collect_official_codi_endpoint_tsvc_corrected.py`
- Utility `scripts/run_official_codi_endpoint_tsvc_corrected_utility.py`
- Analysis `scripts/analyze_official_codi_endpoint_tsvc_corrected.py`
- Kaggle notebook `notebooks/kaggle_official_codi_endpoint_tsvc_corrected.ipynb`

Large runtime outputs, reports, and logs live in separate corrected trees. Batch files,
manifests, request hashes, basis hashes, and checksums make interruption and resume
auditable.

## Interpretation boundary

This is a one-step local optimization test at the final official checkpoint. Passing
shows that a fixed learned subspace produces more useful local updates than the controls.
It does not by itself show long-run training or exact-match improvement. Failure applies
only to this source-native, linear, fixed-rank residual definition.
