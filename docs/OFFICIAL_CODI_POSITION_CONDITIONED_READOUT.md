# Official CODI position-conditioned low-rank readout experiment

## Status

Implemented and awaiting Kaggle execution. No result is claimed before the locked
full-GSM8K run completes.

## Motivation

The confirmed PC 4–31 result retained 0.3806 numeric exact match against a 0.4337
forced-cue baseline. That run compressed only the state that selected the first
visible answer token. The original head was restored for all later token decisions.

Using the same fixed endpoint head at every visible position is a different and much
stronger intervention. The previous fixed global rank-32 result of 0.0508 shows that
the colon subspace cannot simply be reused throughout the answer. This experiment
tests whether later positions instead occupy different small readout spaces.

## Frozen position buckets

| Bucket | Zero-based visible answer positions |
| --- | --- |
| `p0` | 0 |
| `p1` | 1 |
| `p2` | 2 |
| `p3_5` | 3 through 5 |
| `p6_plus` | 6 and later |

Bucket boundaries, ranks, epochs, and gates are frozen before the test outputs are
read. Position zero uses the confirmed 28-dimensional PC 4–31 band. Every later
bucket uses rank 32 selected from that bucket's own training-state covariance by
readout-aware score. Learned arms distil the frozen original vocabulary logits.

## Leakage control

GSM8K training questions are deterministically divided into:

- 1,024 basis-fitting and clean-distillation questions;
- 256 question-disjoint selection questions;
- 256 question-disjoint compressed-rollout recovery questions.

The 1,319 GSM8K test questions are used only after every head is frozen. No test
activation, token, label, rank, epoch, bucket boundary, or stopping decision enters
fitting or selection.

## Locked arms

| Arm | Token 0 | Later tokens | Purpose |
| --- | --- | --- | --- |
| `full` | full | full | Accuracy and latency baseline |
| `first_token_pc4_31_then_full` | fixed rank 28 | full | Reproduce the 0.3806 local result |
| `same_pc4_31_everywhere` | fixed rank 28 | same fixed rank 28 | Naive global extension |
| `fixed_position_local` | fixed rank 28 | bucket-specific fixed rank 32 | Test fixed local geometry |
| `learned_position_local` | learned rank 28 | bucket-specific learned rank 32 | Test clean-trajectory distillation |
| `learned_position_local_onpolicy` | learned rank 28 | bucket-specific learned rank 32 | Correct states visited after compressed decisions |
| `permuted_position_local_onpolicy` | same learned rank 28 | permuted learned rank-32 experts | Equal-storage and equal-online-rank later-position specificity control |
| `learned_global_r32` | learned rank 32 | same learned rank 32 | Equal online-rank, trajectory-aware global control |
| `learned_global_r64` | learned rank 64 | same learned rank 64 | Higher-rank trajectory-aware global control |

The local arm stores five experts but executes only one at each answer step. Its
online arithmetic is therefore rank 28 or 32, while its stored parameter count is
larger than a single global head. Both quantities are reported.

The global controls receive the exact union of the bucket training states. Their
initial bases are selected from the trajectory-wide covariance by readout-aware
score, so the local method is not compared against an endpoint-only global baseline.

## Intervention contract

For bucket `b`, the offline construction is

```text
A_b = W U_b
b_b = W mu_b
```

and the online readout is

```text
c_t     = U_b^T (h_t - mu_b)
logits  = A_b c_t + b_b
token   = argmax(logits)
```

The decoder explicitly sets the zero-based answer position before every visible
answer forward pass. Prompt encoding and continuous-latent passes are marked
inactive. Their vocabulary logits are unused; all-position compressed arms use a
compressed inactive head so those unnecessary projections are compressed as well.

## Outcomes

The notebook records:

- numeric exact match and paired per-question outcomes;
- full-head top-1 agreement in every position bucket;
- first-token and baseline reproduction gates;
- wall time, examples per second, microseconds per question, and microseconds per
  visible generated token;
- exact generated-token counts and EOS-sensitive sequence lengths;
- isolated head latency, theoretical MACs, and unique stored parameter counts.

The primary arm is `learned_position_local_onpolicy`. Position locality is supported
only if the run is complete and valid, the primary arm is faster than full CODI, and
it beats both `same_pc4_31_everywhere` and `learned_global_r32`, with a positive lower
bound for the paired improvement over global rank 32. It must also beat the cyclically
permuted expert assignment, which contains the same learned parameters and uses the
same online rank at each step but sends positions 1 and later to the wrong expert.

## Files

- reusable router: `src/mech/position_conditioned_readout.py`
- decoder routing: `src/models/official_codi.py`
- notebook builder: `scripts/build_kaggle_codi_position_conditioned_readout_notebook.py`
- Kaggle notebook: `notebooks/kaggle_codi_position_conditioned_readout.ipynb`
- tests: `tests/test_position_conditioned_readout.py` and
  `tests/test_kaggle_eigenspace_readout_notebooks.py`
