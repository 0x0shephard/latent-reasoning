# Latent value injection

## Question

Are the intermediate values that §55 decoded from CODI's latent workspace
**causally consumed** by the computation — and can wrong answers be **repaired**
by writing the correct values in?

This is the causal tier the §55 confirmation deliberately deferred, and the
upgrade the field's newest methodology work demands: decodable patterns alone do
not establish that a model *uses* them. Unlike every completed endpoint edit
(§43–§52), this intervention acts *inside the latent loop*, at the measured value
slots, so it propagates through block 11's KV contribution, the recurrent thought
chain, and everything downstream.

## Intervention

At state 11 of the value-slot latent passes (thoughts 1, 3, 5), add a value
token's unit-normalized readout direction, scaled by beta times the state's RMS:

```text
h <- h + beta * rms(h) * W[token] / ||W[token]||
```

Three arms share identical slot masks and scale and differ only in which value
they write:

| arm | value written per slot |
|---|---|
| `gold` (repair) | the k-th gold intermediate of the question |
| `offset` (corruption) | the k-th gold intermediate plus one |
| `random` (control) | a seeded draw from the numeric token pool |

Questions with no `<<…=v>>` intermediate are generated but excluded from both
gates. The `baseline` arm applies nothing.

## Frozen protocol

The population, split, and seed are the standard frozen partition
(fit 440 / select 440 / test 439, seed `20260827`, partition SHA `c8316e46…`),
built directly from the pinned GSM8K test file (SHA `3730d312…`). Outcomes are
numeric exact match under the released forced-cue greedy protocol.

Beta is chosen from the frozen grid `{0.5, 1, 2, 4}` **on the select split** by
one criterion — gold-minus-random recovery on baseline-wrong injectable rows,
ties to the smaller beta — and every arm inherits it. The corruption arm is never
run on select. The test split is read once per arm.

## Frozen gates

| gate | rows | passes if |
|---|---|---|
| **values causally used** | baseline-correct, injectable | damage(offset) − damage(random) ≥ **5 points**, positive paired-bootstrap lower bound |
| **values repairable** | baseline-wrong, injectable | recovery(gold) − recovery(random) ≥ **3 points**, positive paired-bootstrap lower bound |

Statuses: `values_used_and_repairable`, `values_used_not_repairable`,
`values_repairable_only` (unexpected — recheck before claiming),
`value_injection_not_supported`. This is a pristine preregistration: no injection
has ever been run and no threshold was tuned on any outcome.

Expectation stated in advance: the corruption gate is the likely pass (breaking is
easier than fixing); the repair gate is the coin flip. `values_used_not_repairable`
would itself be a sharp result — the workspace is causally consumed but a wrong
run is corrupted beyond its value slots.

## Efficiency measurement (companion protocol)

The same notebook runs the two findings-derived efficiency routes as a
protocol-frozen **measurement study** (no hypothesis gates):

- **latent-budget sweep**: full-GSM8K accuracy and wall clock at
  `latent_iterations ∈ {3, 4, 5, 6}` — the model was trained at M = 6, so the
  accuracy cost of inference-time truncation is the open measurement;
- **rank-k readout microbenchmark**: median latency of the full lm_head
  projection versus the factorized rank-{28, 32, 64} readout. The accuracy side
  is already measured (§40 retention: rank 32 keeps 94.4% of exact match).

## Run

Use
[`kaggle_official_codi_value_injection.ipynb`](../notebooks/kaggle_official_codi_value_injection.ipynb).
Attach the official reproduction `summary.json`; the pinned solutions file is
downloaded and hash-checked in-notebook. Kaggle GPU required. Roughly 13 short
generation runs (select sweep) plus 4 test arms plus the efficiency sweep — one
session.

## Status

Complete — **`value_injection_not_supported`**, both gates failed. Recorded in
ledger §58; the companion efficiency measurements are in §59.

## Completed result (2026-08-28): `value_injection_not_supported`

Kaggle export `jonraza15/writing-values-two-efficiency-routes`, pinned at commit
`bb8631d`; all checksums intact and the gate report recomputes bit-identically.
Beta selection chose 2.0 (the select curve peaked at +1.6 points of gold-over-random
recovery — already nearly nothing). Every injection arm applied 1,128 identically
scaled edits (edit RMS norm 18.2, i.e. twice the state RMS at three slots each).

| gate | rows | observed |
|---|---|---|
| values causally used | 183 baseline-correct | offset **0.9945** vs random **0.9945** — difference **exactly 0.00**, CI [−1.6, +1.6] |
| values repairable | 248 baseline-wrong | gold 3/248 vs random 1/248 — **+0.8 points**, CI [0.0, +2.0] |

The headline is the corruption null: writing plausible *wrong* values into the
value slots, at twice the state's own RMS, left **99.45% of correct answers
untouched** — indistinguishable from random numeric tokens. The latent computation
is almost completely robust to additive readout-direction edits at the value
slots.

Interpretation, bounded as frozen:

- Under **this additive injection** the decoded workspace values are not shown to
  be causally consumed. The result does not contradict §52's Phase-3 finding that
  the latent states as a whole are causally necessary (zeroing/shuffling them
  costs many points); it localizes the robustness: the *vocabulary-aligned
  component* that makes the values readable is not the component the computation
  runs on. The decodable values behave as a readable shadow of a redundant,
  distributed code.
- This is the strongest evidence in the project for the field's newest
  methodological warning (decodable ≠ used): the same values that pass four
  preregistered decoding gates (§55) resist causal manipulation through their own
  readout directions.
- Open, and honestly stated: a different injection form (replacement instead of
  addition, multi-token values, earlier layers, or editing the KV entries the
  later passes attend to) could still succeed. No such follow-up is planned this
  semester.
