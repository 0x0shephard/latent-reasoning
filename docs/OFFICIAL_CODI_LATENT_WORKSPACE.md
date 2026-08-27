# Latent-workspace confirmation

## Question

Are CODI's six latent thoughts a measurable arithmetic workspace — carrying the gold
solution's intermediate values in fixed value slots, with less correct content on
wrongly answered questions, and with the endpoint's wrong answers traceable to the
workspace's own numbers?

This freezes the §53 exploration into gates. The instrument is the model's own
vocabulary projection: each thought's `ln_f` state is decoded through the frozen
readout (top-5), and the numeric tokens are compared by exact string match against
the `<<…=v>>` intermediate values of the pinned GSM8K test solutions. No probe is
fitted, no weight is updated, and no state is edited; a value longer than one GPT-2
token can never match, so every recovery figure is a conservative lower bound.

## Inputs and population

- the completed §52 trajectory export (`[1319, 6, 13, 768]`, commit `40ddfc0`,
  parity-gated on the pinned environment), which also supplies the live first-token
  correctness labels;
- the frozen readout export;
- `openai/grade-school-math` `test.jsonl` at revision `3101c7d5…`, pinned by SHA-256
  (`3730d312…`) and matched to the cached question order by normalized text.

The frozen partition is unchanged (seed `20260827`); the final **439 test questions
are read once**. Every threshold below was frozen from the §53 fit/select
observations before that read. The §53 exploration did include the test rows in its
aggregates, so this is a corrective-lineage confirmation in the sense of §42, not a
pristine preregistration.

## Frozen gates

| gate | passes if | fit/select observation |
|---|---|---|
| content | mean recovery of gold intermediates beats a seeded derangement null by ≥ 10 points, positive paired-bootstrap lower bound | 33% vs 5% |
| structure | gold hits at even thoughts ≤ 10% of all hits, and every odd thought's hit rate ≥ 0.30 | 0.4% even share; odd rates ≈ 0.6 |
| correct/wrong gap | recovery on correct questions beats wrong by ≥ 5 points, positive two-sample bootstrap lower bound | ≈ 16 points |
| faithful readout | on wrong questions, the model's own first answer token appears among the thought numbers ≥ 4 points more often than the gold first token, positive paired lower bound | 19.5% vs 11.0% |

`workspace_confirmed` requires all four. The thought-to-step **alignment table is
descriptive only** — the fit/select observation is that step order is *not*
preserved (intermediate k does not preferentially occupy value slot k), and no gate
depends on it.

## What the gates mean

- **content + structure**: the thoughts are a legible store of intermediate values
  in fixed slots — the CODI paper's qualitative decoding claim, made quantitative
  against a matched null on held-out questions.
- **correct/wrong gap**: the correctness information at the trajectory is *content*
  (did the model compute the right quantities), not confidence geometry.
- **faithful readout**: wrong final answers follow from the workspace's own wrong
  numbers — the failure originates in the latent computation, not in the endpoint's
  reading of it. Together with the gap, this explains the complete §43–§52 null
  lattice: the trajectory never holds the final answer, and a wrong workspace holds
  genuinely wrong values rather than a removable overlay.

## Run

CPU-only and deterministic:

```bash
python scripts/run_official_codi_latent_workspace.py \
  --trajectory <latent_trajectory.pt> \
  --states <colon_states.pt> --readout <readout.pt> \
  --solutions <gsm8k test.jsonl at 3101c7d5> \
  --output latent_workspace.json
```

`scripts/analyze_official_codi_latent_workspace.py` recomputes the report from the
written summary and artifact.

## Status

Complete — **`workspace_confirmed`**, all four gates passed. Recorded in ledger §55.

## Completed result (2026-08-27): `workspace_confirmed`

Run locally (CPU, deterministic) on the checksummed §52 export at preregistration
commit `5c1e326`; the report recomputes bit-identically with the analyzer, and the
summary, artifact, and report are pinned in-repo under
[`artifacts/latent_workspace/`](../artifacts/latent_workspace/) with SHA-256 sums.
One read of the frozen 439-question final split (424 scored; 15 questions have no
`<<…=v>>` annotation).

| gate | threshold | observed | CI |
|---|---|---|---|
| content | ≥ +10 pts over derangement null | **31.0% vs 5.6% → +25.3 pts** | [+22.5, +28.3] |
| structure | even share ≤ 10%, odd rates ≥ 0.30 | even share **0.14%** (1 hit / 719); odd rates 0.62 / 0.54 / 0.54 | — |
| correct/wrong gap | ≥ +5 pts | **41.7% vs 22.6% → +19.2 pts** | [+14.3, +24.1] |
| faithful readout | ≥ +4 pts | own **19.3%** vs gold **10.4% → +8.8 pts** | [+2.8, +14.9] |

Hits per thought on the final split: `[0, 263, 0, 228, 1, 227]` — the value-slot
structure is essentially exact. The descriptive alignment table confirms the §53
observation that step order is *not* preserved (step 1 matches other slots more
often than its own, 0.36 vs 0.20): the workspace is an unordered value store.

Confirmed conclusion, bounded to official CODI GPT-2, GSM8K, top-5 exact-match
decoding at each thought's `ln_f` state:

> CODI's six latent thoughts are a measurable arithmetic workspace. The three odd
> thoughts store the solution's intermediate values (recovering 31% of gold
> intermediates verbatim against a 5.6% matched null, a conservative lower bound);
> wrongly answered questions carry 19 points less of the correct content; and the
> model's wrong answers appear in its own workspace numbers significantly more
> often than the gold answers do. Errors originate in the latent computation, and
> the endpoint faithfully reads them out — which is why no endpoint-side detection
> or correction beyond the model's own margin was ever possible (§43–§52).
