# Exact-match confirmation of the accuracy-bearing colon-state PC band

## Status

| Milestone | State |
| --- | --- |
| Analytic discovery (first-token) | complete |
| Band construction and registry arms | implemented |
| Preregistered exact-match gates | frozen |
| Kaggle execution | pending |

Writes to `outputs/official_codi_endpoint_band_confirmation`. The completed
margin-geometry artifacts are read-only inputs and are never modified.

## What is being confirmed

The margin-geometry analytic tier measured, on held-out **first-token** accuracy at
the forced answer cue (parity 1.0 against the released decoder, float32, PCs fit on
2,048 GSM8K-train questions and applied to the 1,319 test questions):

| subspace | dims | % of variance | retain → frac. of baseline | remove → Δ |
| --- | ---: | ---: | ---: | ---: |
| PC 0–3 | 4 | **82.31%** | **0.067** | 1.74 pp |
| PC 4–15 | 12 | 7.49% | 0.506 | 13.57 pp |
| **PC 4–31** | **28** | **11.31%** | **0.859** | **32.22 pp** |
| PC 0–31 | 32 | 93.62% | 0.926 | 32.75 pp |
| PC 32–767 | 736 | 6.38% | 0.222 | 3.11 pp |

Baseline first-token accuracy 0.4208.

The finding is a dissociation between variance rank and answer contribution. The
leading principal component alone carries 66.3% of the variance and 6.3% of the
accuracy; removing it costs 0.15 points. The accuracy lives in an interior band that
any variance-ranked criterion ranks *down*.

This is the first positive result in the project, which is exactly why it needs
confirming on the outcome the rest of the work is stated in.

## Why the analytic result is not sufficient on its own

The analytic tier is exact for the **first answer token** only. It is a real outcome —
the released decoder's greedy argmax at the colon — but GSM8K answers can be multiple
tokens, and the project's other results are numeric exact match under full greedy
decoding. Two things could differ:

1. a subspace could preserve the first digit while corrupting later ones;
2. the retention intervention persists for the whole generation, not just one token,
   because the projection is applied every time the cue forward pass runs.

## Preregistered gates

Frozen before any exact-match outcome was read. All three must pass.

| Gate | Requirement | Discovery value (first-token) |
| --- | --- | ---: |
| Sufficiency | retain PC 4–31 ≥ **0.70** of baseline | 0.859 |
| Dissociation | retain PC 0–3 ≤ **0.20** of baseline, and primary − control has a positive paired bootstrap lower bound | 0.067 |
| Necessity | remove PC 4–31 costs ≥ **20 points**, positive bootstrap lower bound, exact McNemar `p ≤ 0.05` | 32.22 pp |

The thresholds are looser than the discovery values because exact match is strictly
harder than the first token alone. A run that lands between the threshold and the
discovery value still confirms the claim; one that falls below does not.

A fourth condition applies to the whole run: the forced-cue baseline must sit within
1.5 points of the reproduction gate. This exists because the previous run's generation
arms used `--precision auto`, which resolves to emulated bfloat16 on T4-class GPUs and
moved the baseline from 43.29% to 40.41%. Precision is now pinned to float32 in the
config, the runner default, and the notebook, and the baseline arm asserts its own
accuracy before any comparison is made.

## Arms

Twelve full-GSM8K greedy decodes:

- `baseline`
- retention of PC 0–3, 4–15, 4–31, 0–31, 32–767
- removal of PC 4–31 and PC 0–3
- four random rank-28 retention controls

Random controls are **descriptive**. The specificity null was established
analytically with 200 energy-matched replicates; rebuilding it through generation
would cost 200 full decodes for a comparison the analytic tier already settles.

## Bands are appended, not inserted

`build_margin_arm_registry` appends band targets after every existing target and
seeds their controls past the existing range, so adding them leaves the names and
bases of the already-exported margin-geometry arms bit-identical.
`tests/test_endpoint_band_confirmation.py` pins that.

## Interpretation limits

Retention is an intervention, not a decomposition: the complement is replaced by the
calibration mean, which pushes the state off-manifold. Under the stricter control of
filling the complement from a different question, the primary band retained 0.728 of
baseline rather than 0.859 — weaker but still a majority. The band was also stable
across disjoint calibration halves (retention 0.834 and 0.852; mean principal-angle
cosine 0.979 and 0.968).

A positive result establishes where the answer information sits at the forced answer
cue of this frozen checkpoint. It does **not** license a distillation target, and it
makes no inference-speed claim: a projection hook adds work and does not narrow
GPT-2's width or skip a block. Results are bounded to official CODI GPT-2, state 12,
the forced cue, linear subspaces, and GSM8K.

## Implementation

- configuration: `configs/official_codi_gpt2.yaml` under `endpoint_band_confirmation`
  and the `confirmation_bands` entry of `endpoint_margin_geometry`
- bands: `build_band_subspace` and `band_variance_share` in `src/mech/endpoint_margin_geometry.py`
- arms: `scripts/run_official_codi_endpoint_margin_generation.py` (unchanged; band arms resolve by name)
- gates: `src/eval/official_codi_endpoint_band_confirmation_analysis.py`
- analysis CLI: `scripts/analyze_official_codi_endpoint_band_confirmation.py`
- notebook: `notebooks/kaggle_official_codi_endpoint_band_confirmation.ipynb`
- tests: `tests/test_endpoint_band_confirmation.py`
