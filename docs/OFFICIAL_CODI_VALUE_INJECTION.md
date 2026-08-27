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

Implementation and synthetic validation are complete. No injection has been run;
the repository makes no empirical claim about either gate.
