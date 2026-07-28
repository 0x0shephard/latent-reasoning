# Final MATH-500 8,192-token eligibility screen

## Motivation

The paired 4,096-token diagnostic was valid and complete:

| Metric | 2,048 tokens | 4,096 tokens |
| --- | ---: | ---: |
| Accuracy | 32 / 64 | 37 / 64 |
| Accuracy rate | 50.000% | 57.8125% |
| Length-limited examples | 31 / 64 | 20 / 64 |
| Median generated tokens | 1,948.5 | 1,948.5 |

The paired gain was 7.8125 percentage points with a 95% bootstrap interval
from -1.5625 to +17.1875 points. Seven answers changed from incorrect to
correct and two changed from correct to incorrect.

Every 2,048-token length-limited sequence was an exact prefix of its
4,096-token counterpart. Every sequence that had already reached EOS was
reproduced exactly. The result therefore reflects deterministic continuation,
not a stochastic rerun difference.

The unchanged eligibility gate still failed because 37 of 64 correct answers
is 57.8125%, below the 60% minimum. At least 39 correct answers are required.
The cap also remained binding for 20 questions. This justifies one final
extension before closing the 1.5B model configuration.

## Fixed design

Input dataset:

`jonraza15/math-500-token-budget-sensitivity-screen`

The experiment holds fixed:

- model and revision
- float32 precision
- prompt and chat template
- greedy decoding
- symbolic grader
- exact 64 MATH-500 questions
- original 60% to 85% accuracy gate
- minimum 512-token median reasoning length
- requirement that 150 disjoint questions remain

Only the 20 questions that ended at the 4,096-token limit are regenerated with
an 8,192-token maximum. The other 44 completed greedy generations are reused.
Every regenerated sequence must preserve the entire 4,096-token reference as
an exact prefix.

This is preregistered as the final token-cap escalation for the 1.5B setup.

## Decision

The 20 extended records are composed with the 44 completed records to recover
the full 64-question screen.

If the composed screen reaches all original eligibility criteria, a fresh,
disjoint 64-question confirmation is run at 8,192 tokens. The 150-question
compression-risk pilot is authorized only if that confirmation also passes.

If the composed screen fails, or if the fresh confirmation fails, close this
1.5B model configuration. Do not raise the token cap again.

## Durability

Run:

`notebooks/kaggle_kv_risk_math_token_budget_8192.ipynb`

The runner writes one atomic record per extended or confirmation question.
Exit code 42 means the exported output is a valid resume source. A pending
report is written before confirmation begins, so the extension result remains
inspectable even when confirmation requires another Kaggle session.

## Interpretation boundary

This experiment only determines whether MATH-500 becomes an eligible operating
point for the later risk pilot. It does not measure cache-compression failure,
failure nesting, stochastic noise, or predictability.
