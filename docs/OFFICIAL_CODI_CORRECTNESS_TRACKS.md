# Correctness geometry at CODI's answer cue

Three preregistered tracks asking whether the part of the state that **predicts** being
right has anything to do with the part that **produces** the right answer.

Background: [`RESEARCH_CONTEXT_LEDGER.md`](RESEARCH_CONTEXT_LEDGER.md) §41 (exploratory
findings) and §42 (this design). The band result it builds on is §40 and
[`OFFICIAL_CODI_ENDPOINT_BAND_CONFIRMATION.md`](OFFICIAL_CODI_ENDPOINT_BAND_CONFIRMATION.md).

## The claim being tested

The band experiment found PCs 4–31 — 28 of 768, 11.3% of the colon-state variance —
carry 87.8% of exact-match accuracy. PCs 0–3 hold 82.3% of the variance and 6.1% of the
accuracy: they shift all 50,257 logits together, which cannot change an argmax.

Splitting the same states by correctness puts the correctness direction **97% inside
PCs 0–3**. So the signal that predicts being right sits almost entirely in the
directions that cannot change an answer. Three consequences follow, one per track.

## Tracks and gates

| track | question | primary arm | passes if |
|---|---|---|---|
| **detect** | does the state say anything about correctness the output doesn't? | `fisher_plus_margin` | ΔAUC over margin-only ≥ 0.01, bootstrap lower bound > 0 |
| **steer** | can one fitted global offset inside the band improve answers? | `margin_band` | gain ≥ 1.0 pt, lower bound > 0, and above the best matched random direction *in the same band* |
| **project** | does building from correct examples only help? | `correct_only` @ rank 28 | advantage over class-blind ≥ 1.0 pt, lower bound > 0 |

Every gate is stated against the thing that would otherwise explain the result, not
against chance. The model's own margin scores AUC 0.874, so a detector at 0.70 is a
failure however far above chance it sits.

**The steer gate is expected to fail**, and saying so in advance is the point. A pass
would be the project's first accuracy-improving intervention; a failure rules out this
single global translation, not a question-conditioned or learned steering map.

## Split discipline

```
fit    (1024)  every direction, probe and steering vector
select (1024)  every hyperparameter: ridge, Fisher shrinkage, alpha, rank
test   (1319)  GSM8K test, read once per arm
```

Nothing is chosen on test. This closes the standing caveat on the band experiment,
whose boundaries (4, 32) were read off test-set curves.

## Running it

Both tiers are driven by the Kaggle notebook, which pins the reproducing environment,
repairs the peft/torchao dispatch, and runs the source tests before anything else:

```bash
python scripts/build_kaggle_official_codi_correctness_tracks_notebook.py
```

Attach the completed margin-geometry export (`colon_states_seed89/`) and a reproduction
summary, then **Save Version → Save & Run All**.

Locally, against an existing colon-state cache:

```bash
python scripts/run_official_codi_correctness_tracks.py --states colon_states.pt --readout readout.pt --output tracks.json
```

```bash
python scripts/analyze_official_codi_correctness_tracks.py --sweep tracks.json --output report.json
```

The analytic tier is model-free and takes well under a minute at full scale. The
generation tier is three full-GSM8K greedy decodes (~2h) confirming the steer track on
numeric exact match; α comes from the analytic export and is never re-tuned there.

## Arms

**detect** — nine probes: `margin` (the baseline), `mean_difference`, `fisher`,
`lift_band` (PC 0–3), `accuracy_band` (PC 4–31), `full_state`, and the three
`*_plus_margin` combinations. Only the combinations answer the question that matters.

**steer** — `mean_difference_global` (the arm §41 already ran), `mean_difference_band`,
`fisher_band`, `margin_global`, `margin_band` (primary), plus 8 matched random
directions inside the band and 8 in the full space. `band_profile` reports how much of
each vector lies in PC 0–3 versus PC 4–31, which is what ties the outcome to the
mechanism.

**project** — `class_blind`, `correct_only` and `incorrect_only` bases at ranks 4, 12,
28, 32, 64, 128, with principal-angle cosines between the class-blind and correct-only
bases at each rank.

## Two exactness notes

Steering is a constant translation and the readout is linear, so

```
(h + alpha*v) W^T  =  h W^T  +  alpha * (W v)
```

which turns ~150 `[1319, 768] x [768, 50257]` float64 products into two. Retention uses
the matching low-rank form `mu W^T + ((h - mu) U)(U^T W^T)`. Both are tested against the
dense computation for bit-identical outcomes — these are algebra, not approximations
traded for speed.

`roc_auc` uses midranks rather than a double `argsort`, which breaks ties by position
and would score a fully tied probe at whatever the question order dictates instead of
0.5. The bootstrap resamples with replacement, so ties are the normal case there.

## Scope

Official CODI GPT-2, state 12, forced answer cue, linear subspaces, GSM8K, frozen
weights throughout. No distillation target and no inference-speed claim.

---

## Results (completed run)

Kaggle export `codi-that-predicts-the-right-answer`. Both tiers ran. Full detail in
[`RESEARCH_CONTEXT_LEDGER.md`](RESEARCH_CONTEXT_LEDGER.md) §43.

| track | verdict | headline |
|---|---|---|
| detect | **PASS**, at the edge | ΔAUC +0.0123 over margin-only, CI lower bound 3×10⁻⁵ |
| steer | **FAIL**, decisively | exact match unchanged: 0.43366 both; a random band direction did better |
| project | **FAIL**, as predicted | −0.076 pts; mean principal-angle cosine 0.9826 |

The environment reproduces the checkpoint again: baseline exact match 0.43366 against
a re-decoded gate of 0.43594 (drift 0.0023, 100% cue coverage, float32).

**The steer null is the substantive result.** Confining the global steering vector to
PC 4–31 was the strongest constant-offset arm tested, and on full GSM8K greedy decoding
it changed nothing at all — while a matched random direction in the same band scored
marginally higher. This does not test a question-conditioned correction: averaging
answer-specific margin gradients across different gold answers can cancel them.

**Known limitation.** Split base rates are fit 67.5% / select 66.3% / test 42.1%
correct: calibration comes from GSM8K train, which CODI was trained on. The nulls are
robust to this; the narrow detect pass is not, and should be replicated with directions
fitted on held-out test-like questions.

## Post-run audit

The historical §41 steering record did not preserve normalization metadata, so its
alpha values are not numerically comparable with this run's explicitly unit-normalized
vectors. The completed detector also recorded no optimizer convergence diagnostic.
That makes a convergence-audited rerun appropriate, but the lower held-out AUC of the
768-feature probe does not itself prove non-convergence: extra features can hurt
generalization even when the ridge-logistic training objective is optimized.

The durable corrected interpretation is therefore bounded: constant global steering
failed; correct-only projection failed; and detection passed its preregistered gate but
remains provisional because it was fitted on the model's training distribution.
