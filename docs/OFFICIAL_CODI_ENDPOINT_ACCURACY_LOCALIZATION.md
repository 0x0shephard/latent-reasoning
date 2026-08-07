# Frozen-checkpoint CODI answer-colon accuracy localization

## Question and preregistered scope

The earlier 82-arm experiment established that jointly removing the six
answer-conditioned directions lowered forced-cue GSM8K accuracy by 1.744 percentage
points and jointly removing the six parameter-aware directions lowered it by 2.502
points. No individual direction passed. Its random controls were rank matched but not
activation-magnitude matched.

This follow-up asks two narrower questions:

1. Does either joint effect exceed random subspaces that remove the same amount of
   calibrated activation energy at both endpoint states?
2. If its joint parent passes that stronger gate, can the effect be localized to a
   state or direction, or does it require interaction among directions?

Answer-conditioned and parameter-aware are the only confirmatory selector families
because they passed the previous joint causal gate. Energy is retained as a
preregistered negative-control joint arm, not promoted to another localization family.

## Frozen endpoint intervention

All arms load the identical official checkpoint and evaluate the same 1,319 GSM8K
questions in the same order. They force the same collector endpoint:

```text
question -> BOT -> z1 ... z6 -> EOT + "The answer is:" -> generated answer
```

At the cue colon, for an orthonormal basis `U_s`, the hook replaces its centered
coordinates by their fresh calibration mean:

\[
h'_s = h_s - U_sU_s^\top(h_s-\mu_s).
\]

No parameter changes. State 11 is the output of GPT-2 block 10, and state 12 is the
output of block 11 followed by `ln_f`. The hook fires only on the colon forward. Exact
cue reach must be paired across every arm and at least 95%; the expected value is 100%.

## Fresh covariance and activation-energy matching

The calibration collector samples 1,024 fresh eligible training questions using seed
71. It excludes all 10,632 normalized questions used by the three selector experiments
and the seed-53 retention-training partition. It records the student endpoint mean and
the full 768-by-768 covariance at states 11 and 12.

For a rank-three basis, its calibration projection energy is

\[
E_s(U)=\operatorname{tr}(U^\top C_s U)
      =\mathbb E\|U U^\top(h_s-\mu_s)\|_2^2.
\]

Every random control is method specific. At each state, it constructs independent
orthonormal spectral subspaces on opposite sides of the selected target energy and
orthogonally interpolates between them. Consequently:

- random and selected bases have the same rank;
- `E_s(U_random)` matches `E_s(U_selected)` independently at states 11 and 12;
- no per-example or post-outcome tuning is used;
- the intervention remains an unscaled orthogonal removal (`alpha=1`);
- normalized squared overlap with the selected basis must not exceed 0.20.

This avoids the extrapolation that would result from multiplying a low-energy random
projection by a factor larger than one. The smoke gate requires relative calibration
energy error at most `2e-5`.

## Registered 232 arms

- One unmodified forced-cue baseline.
- One energy joint negative control.
- Two confirmatory joint arms: answer-conditioned and parameter-aware.
- Four state-only arms: two states for each confirmatory method.
- Twelve single-direction arms: six per confirmatory method.
- Twelve joint-minus-one arms: remove five directions while retaining one selected
  direction, six per confirmatory method.
- Two hundred activation-energy-matched random joint arms: 100 independently seeded
  controls for each confirmatory method.

The single arm asks whether removing one direction is sufficient to hurt accuracy. The
joint-minus-one arm asks the complementary question: when the other five are removed,
does retaining this direction rescue accuracy relative to removing all six? Together
they can distinguish individual necessity, rescue within a group, and nonlinear or
redundant group effects without running all 63 subsets per method.

## Confirmatory statistics

For each joint selector the analysis computes paired accuracy loss, a paired bootstrap
95% interval, a one-sided exact McNemar test, deterministic even/odd-half losses, and an
empirical p-value against its 100 matched random controls.

A joint subspace passes only when:

1. its accuracy loss is positive in both deterministic halves;
2. the bootstrap lower bound is positive;
3. its McNemar p-value survives Holm correction across the two selectors;
4. its empirical matched-random p-value survives Holm correction across the two
   selectors.

With 100 controls, the smallest raw empirical p-value is `1/101 = 0.00990`, so both
selectors can survive the two-test Holm family even when both land at the null boundary.

State and direction claims are hierarchical: they are confirmatory only if their joint
parent passes. State tests are Holm corrected over two states within a method. Single
necessity and joint-rescue tests are separately Holm corrected over six directions
within a method. A direction is called an **accuracy-core direction** only when it both
passes the single-removal necessity test and significantly rescues the joint ablation.

## Interpretation boundaries

This experiment can establish causal necessity only conditional on the fixed answer
cue. It cannot claim that native cue-free decoding visits the same colon. Calibration
energy matching controls intervention magnitude in expectation on fresh training data;
the report also records realized evaluation RMS, because calibration and GSM8K need not
match perfectly.

A passing joint parent with no passing direction is evidence for a distributed,
redundant, or nonlinear subspace effect—not evidence that every direction is individually
critical. A directional projection hook also does not reduce transformer width or skip
blocks, so this experiment makes no inference-speed claim.

## Run and resume

Use
[`notebooks/kaggle_official_codi_endpoint_accuracy_localization.ipynb`](../notebooks/kaggle_official_codi_endpoint_accuracy_localization.ipynb).
It runs the source tests, discovers or downloads the three immutable selector artifacts,
fits fresh covariance, smoke-tests cue reach and matching, evaluates every arm, performs
the paired analysis, and exports hashes plus logs. `RESUME_INPUT` restores a previous
export. Optional deterministic arm sharding is available for Kaggle wall-clock limits;
analysis refuses to run until all 232 full arms are present.

Implementation entry points:

- `src/mech/endpoint_accuracy_localization.py`
- `src/eval/official_codi_endpoint_accuracy_localization_analysis.py`
- `scripts/collect_official_codi_endpoint_activation_stats.py`
- `scripts/run_official_codi_endpoint_inference_ablation.py`
- `scripts/analyze_official_codi_endpoint_accuracy_localization.py`
