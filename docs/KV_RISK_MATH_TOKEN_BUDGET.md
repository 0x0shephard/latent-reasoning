# MATH-500 token-budget sensitivity screen

## Why this screen exists

The third repaired KV-compression-risk run was numerically valid:

- the mandatory preflight passed
- the T4 used float32
- full-cache custom decoding matched `transformers.generate`
- all logits and entropy diagnostics were finite
- no generation was classified as degenerate

The fixed 64-example dataset screen then produced:

| Dataset | Accuracy | Median tokens | Length-limited | Original gate |
| --- | ---: | ---: | ---: | --- |
| GSM8K | 78.125% | 405.0 | 0 of 64 | trace too short |
| MATH-500 | 50.000% | 1,948.5 | 31 of 64 | accuracy too low |
| AIME 2024 | 3.333% | 2,048.0 | 30 of 30 | accuracy and sample size |

No dataset passed the preregistered selection rule. The primary compression
pilot therefore did not run.

MATH-500 is the only candidate for which the rejection may have been caused by
the generation ceiling rather than just the model. Almost half of its screen
ended at exactly 2,048 tokens. This experiment distinguishes those explanations
before changing the model or beginning an expensive compression sweep.

## Fixed comparison

The paired diagnostic keeps these fixed:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- pinned model and MATH-500 revisions
- float32 precision
- prompt and chat template
- greedy decoding
- the exact same 64 MATH-500 questions
- symbolic grader

The only changed variable is:

- maximum generation length from 2,048 to 4,096 tokens

The valid third-run dataset
`jonraza15/kv-compression-risk-pilot-export-third-run` is a required read-only
reference. It is not resumed or modified.

## Decision sequence

1. Re-evaluate the same 64 MATH-500 questions at 4,096 tokens.
2. Apply the unchanged eligibility rule:
   - accuracy from 60% to 85%
   - median generated length at least 512 tokens
   - at least 150 unused examples remain
3. If the paired diagnostic fails, stop.
4. If it passes, evaluate a fresh disjoint 64-question set at 4,096 tokens.
5. Authorize the 150-question compression pilot only if the fresh set also
   passes the unchanged rule.

The report additionally records paired accuracy change, incorrect-to-correct and
correct-to-incorrect flips, recovery among previously length-limited examples,
and a paired bootstrap confidence interval.

If the 4,096-token screen remains ineligible and at least 20% of examples still
hit the new ceiling, the result is classified as still truncation-bound. If it
is ineligible without substantial remaining truncation, model capacity or the
prompt is the more likely bottleneck. This classification does not alter the
eligibility gate.

## Kaggle execution

Run:

`notebooks/kaggle_kv_risk_math_token_budget.ipynb`

Attach:

`jonraza15/kv-compression-risk-pilot-export-third-run`

Use a T4 or newer accelerator with Internet enabled. The diagnostic deliberately
uses float32 even on newer GPUs because the 2,048-token reference was generated
in float32. Use Save Version and Run All. Exit code 42 means atomic records were
saved and the exported tree can be attached to a later run.

## Interpretation boundary

This experiment tests whether a generation-length ceiling caused the MATH-500
screen rejection. It does not measure KV-compression failure, establish that risk
is predictable, or compare compression policies.

## Completed result

The 4,096-token diagnostic completed on all 64 paired questions:

- accuracy increased from 50.000% to 57.8125%
- length-limited examples fell from 31 to 20
- seven answers changed from incorrect to correct
- two answers changed from correct to incorrect
- the paired accuracy interval was -1.5625 to +17.1875 percentage points
- all 2,048-token censored sequences were exact prefixes of their continuations
- all previously completed EOS generations reproduced exactly

The predefined decision was `candidate_cap_still_binding`. The fresh
confirmation did not run because the 60% accuracy gate was not reached. The
final bounded extension is documented in
`docs/KV_RISK_MATH_TOKEN_BUDGET_8192.md`.
