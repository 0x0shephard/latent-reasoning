# KV-compression risk pilot

## Status

This is a cheap, inference-only gate. It replaces the proposal-first plan.
No risk predictor or large downstream study should be built unless this pilot
passes its preregistered checks.

## Question

Is KV-compression failure a stable, problem-specific property that could be
predicted, or is it mainly sampling noise and ordinary problem difficulty?

## Fixed design

- Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Model and dataset revisions: pinned in `configs/kv_risk_pilot.yaml`
- Candidate datasets: GSM8K, MATH-500, and AIME 2024
- Primary decoding: greedy
- Primary pilot size: 150 questions
- Cache conditions: full, full repeat, and 90%, 50%, 25%, and 10% requested
  retention
- Compressor: a simple heavy-hitter plus recent-token policy
- Prompt tokens: always retained
- Compression scope: generated reasoning tokens only
- Secondary noise check: three temperature seeds on 30 fixed questions
- Screen and pilot questions: disjoint

The simple compressor is intentional. This pilot tests whether meaningful risk
variation exists, not whether a new eviction policy beats the literature.

## Dataset selection

First run full-cache greedy inference on a fixed screen from every candidate
dataset. A dataset is eligible only when:

1. full-cache accuracy lies between 60% and 85%;
2. median generated reasoning length is at least 512 tokens; and
3. at least 150 unused questions remain for the disjoint pilot.

If multiple datasets qualify, choose the one whose accuracy is closest to
72.5%. Break exact ties by preferring the longer median trace and then the
lexicographically smaller dataset name. AIME 2024 is measured but its public
30-question set cannot supply the 150-example primary pilot.

## Primary outcomes

For every budget, report:

- correct-to-incorrect and incorrect-to-correct flips;
- full-cache repeat disagreements;
- exact-match accuracy;
- generated length;
- requested and realized generated-cache retention;
- retained KV token-steps;
- point-biserial correlation between full-cache size and failure;
- failure containment and reversal across adjacent budgets.

For a looser budget `b1` and tighter budget `b2`, define the failure set as
full-cache-correct questions that become incorrect. Adjacent containment is:

`|F_b1 intersect F_b2| / |F_b1|`.

## Difficulty baseline

The preregistered difficulty baseline uses question length and dataset
difficulty when available. A second diagnostic adds entropy from the first 32
full-cache generated tokens. Completed trace length is reported only as a
post-hoc diagnostic and is not a valid pre-generation router feature.

## Go decision

Proceed to risk-predictor development only if all primary conditions hold:

1. At least one middle budget has a correct-to-incorrect rate from 5% to 25%.
2. In the stochastic subset, within-seed compression disagreement is at least
   twice the full-cache seed-to-seed noise floor.
3. Mean adjacent failure-set containment is at least 0.70.
4. Cross-validated difficulty-only AUROC remains below 0.80.
5. An oracle selective policy can save at least 20% of KV token-memory while
   keeping compression-induced accuracy loss within one percentage point.

If a denominator is zero or a confidence estimate cannot be computed, the
corresponding gate is inconclusive rather than passed.

## Interpretation boundary

A positive pilot supports building a risk predictor. It does not establish that
any proposed online feature is sufficient, that guarantees transfer across
datasets, or that average per-request savings translate directly to batched
serving peak-memory savings.

## Kaggle execution

Run `notebooks/kaggle_kv_compression_risk_pilot.ipynb` with a T4 or newer GPU
and Internet enabled. Use Save Version and Run All so the browser can be
closed. The runner writes one atomic JSON record per condition and question.
If the wall-clock guard returns exit code 42, save the exported output as a
Kaggle dataset, attach it to a new run, and execute the notebook again. Verified
records are skipped, while incompatible resumes are rejected by identity hash.
