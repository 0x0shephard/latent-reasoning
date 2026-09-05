# CODI U28-to-global-rank96 bridge experiment

## Research question

Does the trajectory-trained global rank-96 vocabulary head preserve and use the
same 28-dimensional activation subspace (PCs 4–31 at the post-`ln_f` answer-colon
state) that supported the successful first-token compression experiment?

The experiment does not train another head. It connects two frozen results:

- the answer-colon covariance basis, fitted from 2,048 GSM8K training states;
- the trajectory-whitened, margin-distilled global rank-32/64/96 head, fitted in the
  completed global-head experiment.

The complete 1,319-question GSM8K test is opened only for locked analysis and
generation. No rank, basis, threshold, or weight is selected from test outcomes.

## Why the comparison uses the row span

The global head evaluates `Up(Down(h))`. At rank 96, `Down.weight` has shape
`[96, 768]`; its row span is the part of hidden-state space visible to the head.
Individual rows are not identifiable because an invertible change of the 96
bottleneck coordinates can be canceled inside `Up`. The row span is invariant, so it
is the valid object to compare with the 28-column basis `U28`.

For an orthonormal row-space basis `Q_r`, the primary overlap statistic is

```text
capture(U28 by Q_r) = ||Q_r^T U28||_F^2 / 28.
```

It equals one if the global head can see every direction in `U28`. An isotropic
rank-96 subspace captures 96/768 = 0.125 in expectation. The notebook compares the
learned rank-32 and rank-96 spaces with 200 seeded random spaces at each rank.

## Locked arms

| Arm | First answer token | Later answer tokens |
| --- | --- | --- |
| `dense_full` | dense head | dense head |
| `pc4_31_first_then_dense` | fixed U28 head | dense head |
| `pc4_31_remove_first_then_dense` | dense head after deleting U28 | dense head |
| `leading_pc0_3_first_then_dense` | variance-dominant four-PC control | dense head |
| `random_matched28_r{0..3}_first_then_dense` | selected-orthogonal energy-matched rank-28 control | dense head |
| `pc4_31_every_answer_position` | fixed U28 head | the same fixed U28 head |
| `global_rank32` | learned global rank 32 | learned global rank 32 |
| `global_rank96` | learned global rank 96 | learned global rank 96 |
| `global_rank96_retain_pc4_31_at_first` | rank 96 after retaining only U28 | rank 96 |
| `global_rank96_remove_pc4_31_at_first` | rank 96 after deleting U28 | rank 96 |

The retain/remove edits occur before the LM head and only at zero-based answer
position 0. Any later sequence changes are causal consequences of the first changed
token. The all-position U28 arm directly tests—and is expected to challenge—the
assumption that a colon-local space transfers to the rest of the answer trajectory.

## Measurements

1. Principal-angle overlap and reference capture versus a 200-subspace null.
2. Cached first-token gold accuracy, dense top-1 agreement, logit regret, and margin
   error on the exact state where U28 was discovered.
3. Dense-teacher agreement on `p0`, `p1`, and `p2+` states collected from 256
   question-disjoint training trajectories.
4. Full greedy numeric exact match for every locked arm, with paired bootstrap
   intervals and per-question correct-to-wrong/wrong-to-correct transitions.

The experiment calls a shared mechanism supported only if all three conditions hold:

- learned rank 96 captures U28 above the random 95th percentile;
- deleting U28 costs rank 96 at least 10 percentage points of full-sequence accuracy;
- retaining U28 preserves at least 70% of rank-96 full-sequence accuracy.

## Kaggle inputs

Attach exactly these completed inputs:

1. corrected official CODI reproduction, containing
   `official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json`;
2. `jonraza15/codi-answer-colon-margin-geometry`, containing
   `colon_states_seed89/colon_states.pt` and `readout.pt`;
3. `jonraza15/trajectory-whitened-global-low-rank-lm-head`, containing
   `global_low_rank_head.pt` and the adjacent `summary.json`.

Use a Kaggle GPU, enable Internet, and run all cells in
`notebooks/kaggle_codi_28_to_global96_bridge.ipynb`. Results are written to
`/kaggle/working/codi_28_to_global96_bridge`, including one JSONL file per arm,
`summary.json`, and `bridge_summary.png`.

This is a mechanistic experiment, not another timing experiment. The one-pass arm
times are saved only as bookkeeping and must not be reported as benchmark speedups.
