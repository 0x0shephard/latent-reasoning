# Confirmatory parameter-aware state-12 answer-colon ablation

## Preregistered question

The 232-arm discovery experiment found that removing the parameter-aware rank-three
subspace at state 12 lowered forced-cue GSM8K accuracy by 1.516 percentage points, with
a paired bootstrap interval above zero. The joint six-direction parameter-aware result
was stronger than 96 of 100 activation-energy-matched random controls, but its empirical
`p=0.0495` became `0.0990` after correcting the two discovery selectors. Matching fitted
on the broad arithmetic calibration distribution also transported poorly to GSM8K test
at state 12: median random intervention RMS was about 21% larger than selected RMS.

This experiment freezes one new primary hypothesis before reading new outcomes:

> At the forced answer-cue colon, removing the three previously selected
> parameter-aware state-12 directions lowers frozen-checkpoint GSM8K accuracy more than
> removing selected-orthogonal rank-three state-12 subspaces with the same GSM8K-train
> calibration projection energy.

No answer-conditioned, energy, state-11, or individual-direction hypothesis is tested.
There is therefore one primary statistical test and no selector multiplicity correction.

## Independent calibration population

The matching distribution is the official `openai/grade-school-math` GSM8K **train**
JSONL pinned at revision
`3101c7d5072418e28b9008a6636bde82a006892c`. The collector requires exactly 7,473 raw
train examples, normalizes questions with the repository's frozen question normalizer,
and aborts if any normalized train question appears in the 1,319-example GSM8K test
split.

It deterministically selects 2,048 unique eligible train questions with seed 73. It
records source-index, selected-question, and test-question hashes. Test labels and test
activations are not used for calibration. GSM8K-train answers are used only to construct
valid source-format batches; the fitted quantities are the frozen student's state-12
answer-colon mean and covariance.

## Frozen endpoint and selected basis

Every arm loads the same author checkpoint and consumes:

```text
question -> BOT -> z1 ... z6 -> EOT + "The answer is:" -> generated answer
```

The intervention fires only when the fixed cue colon is consumed. State 12 is the final
GPT-2 block output after `ln_f`, exactly matching the tensor on which the parameter-aware
basis was selected. For the state-12 mean `mu`, covariance `C`, and orthonormal rank-three
basis `U`, the ablation is

\[
h' = h-UU^\top(h-\mu).
\]

`alpha=1`: coordinates inside the basis are replaced by their ordinary calibration
mean. Model parameters never change.

The primary basis contains the frozen parameter-aware state-12 residual PCs 9, 10, and
32 from the completed seed-41 source artifact.

## Five hundred matched controls

For each deterministic random replicate, the implementation constructs a rank-three
subspace inside the exact orthogonal complement of the selected basis. It eigendecomposes
the covariance restricted to that 765-dimensional complement and interpolates between
random lower- and higher-energy spectral subspaces so that

\[
\operatorname{tr}(U_{random}^\top C U_{random})
=
\operatorname{tr}(U_{selected}^\top C U_{selected}).
\]

The control remains an unscaled orthogonal removal. Every arm records calibration energy
error and selected overlap. The smoke gate requires relative energy error at most
`2e-5`, normalized selected overlap at most `0.20`, state-12 rank exactly three, and
exact forced-cue reach.

There are 502 full-test arms:

- one baseline;
- one parameter-aware state-12 primary ablation;
- 500 matched-random state-12 ablations generated with seed `20260808`.

With 500 controls, the smallest plus-one empirical p-value is
`1/501 = 0.001996`. The null resolution is therefore sufficient for a stable single
0.05-level comparison.

## Accuracy decision rule

The primary subspace is confirmed only when every condition passes:

1. paired accuracy loss is positive in both deterministic even/odd test halves;
2. the paired bootstrap 95% lower bound is positive;
3. the one-sided exact McNemar p-value is at most 0.05;
4. the plus-one empirical p-value against all 500 matched controls is at most 0.05;
5. maximum calibration relative energy error is at most `2e-5`;
6. maximum normalized selected overlap is at most `0.20`;
7. median random evaluation RMS divided by selected evaluation RMS lies in
   `[0.90, 1.10]`.

The final RMS rule is a preregistered transport gate. Calibration matching is performed
without test activations, but the causal comparison is not declared confirmatory if that
matching fails to transport to the actual evaluation population. This prevents a larger
random perturbation from masquerading as evidence against selector specificity.

The report status is one of:

- `confirmed`: statistical, calibration, and RMS-transport gates all pass;
- `not_confirmed`: the primary accuracy or matched-null evidence fails;
- `evaluation_magnitude_transport_failed`: accuracy evidence passes but realized
  magnitude comparability does not.

## Interpretation boundary

Confirmation supports one rank-three state-12 group conditional on the fixed answer cue.
It does not make PCs 9, 10, and 32 individually necessary, does not establish the same
effect in native cue-free decoding, and does not produce inference speedup. A projection
hook adds work; structural compression would require a separately trained and evaluated
narrower model.

## Kaggle workflow

Run
[`notebooks/kaggle_official_codi_parameter_state12_confirmation.ipynb`](../notebooks/kaggle_official_codi_parameter_state12_confirmation.ipynb).
It validates source tests, resolves the immutable bases, proves train/test question
disjointness, fits calibration statistics, smoke-tests the contract, evaluates all 502
paired arms, applies the single decision rule, and writes a checksummed resumable export.
Deterministic sharding and `RESUME_INPUT` are available if one Kaggle session is
insufficient.

Implementation entry points:

- `src/mech/endpoint_state12_confirmation.py`
- `src/eval/official_codi_parameter_state12_confirmation_analysis.py`
- `scripts/collect_official_codi_parameter_state12_confirmation_stats.py`
- `scripts/run_official_codi_endpoint_inference_ablation.py`
- `scripts/analyze_official_codi_parameter_state12_confirmation.py`

