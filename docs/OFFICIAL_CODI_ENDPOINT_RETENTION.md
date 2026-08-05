# Official CODI endpoint rank-matched retention experiment

## Question

The three completed endpoint experiments rank residual directions in different ways:

1. **Energy:** largest eigenvalues of the teacher–student residual Gram matrix.
2. **Answer-conditioned:** residual PCs with split-stable local alignment to the
   gradient of gold-answer loss.
3. **Parameter-aware:** residual PCs whose induced trainable-parameter gradient is
   split-stably aligned with the gold-answer parameter gradient.

This follow-up asks whether the directions selected by each rule are sufficient for
accuracy, and whether they are more useful than the directions that the rule discards.

## Important meaning of “remove the other directions”

The residual is a training target, not a separable component of the released student's
inference state. Therefore this experiment filters the **teacher auxiliary residual**
during fine-tuning. It does not replace a 768-dimensional student hidden state by a
three-dimensional state at inference. The latter would erase ordinary language-model
features and would not test TSV-C.

For a frozen orthonormal basis \(U_s\) at state \(s\), with residual
\(r_s=S_s-\operatorname{sg}(T_s)\):

\[
r_s^{\mathrm{selected}}=U_sU_s^\top r_s,
\qquad
r_s^{\mathrm{complement}}=(I-U_sU_s^\top)r_s.
\]

The teacher remains detached. Every selected/complement auxiliary parameter gradient
is rescaled batch-by-batch to have the same norm as the full two-block auxiliary
gradient. Thus a method cannot win merely because its raw auxiliary gradient is larger.

## Fair comparison

All three methods receive the same capacity: the first three directions supplied by
that completed selector at hidden states 11 and 12. State 0 and states 1–10 are excluded.
The answer-conditioned experiment selected exactly three directions at both states, so
rank three is the largest comparison that does not grant another method extra capacity.

The eight training arms are:

- answer loss only;
- full residual target at states 11 and 12;
- selected-only target for energy, answer-conditioned, and parameter-aware;
- complement-only target for energy, answer-conditioned, and parameter-aware.

Every arm uses the released checkpoint, the same 512 fresh normalized questions, the
same order within each of three training seeds, AdamW at \(10^{-5}\), one epoch, and the
same full 1,319-example GSM8K evaluation. All 10,632 questions used anywhere in the
three completed experiments are excluded from the new training partition.

## Registered decisions

The primary sufficiency comparison is selected-only minus the full two-block target.
A selector is called non-inferior only when the lower endpoint of a paired hierarchical
95% bootstrap interval is above -1 accuracy point. The bootstrap resamples training
seeds and then paired GSM8K questions.

The primary concentration comparison is selected-only minus that selector's own
complement. A useful selector should retain accuracy close to the full target and should
outperform its complement. Reporting only a small selected-only accuracy drop is not
enough: answer-only might perform equally well, which would mean the auxiliary target
was unnecessary rather than compressed successfully. The report therefore also gives
paired selected-minus-answer-only and full-target-minus-answer-only intervals.

## Expected result before observing this experiment

At equal rank, parameter-aware is the strongest prior candidate because it selects in
the space in which optimization actually happens. Answer-conditioned is second because
it uses answer-loss information locally. Energy is least targeted to accuracy. This is
only a preregistered hypothesis: the earlier one-step utility screens were negative, so
all three selected-only arms may be indistinguishable from answer-only, or complements
may do as well as selections.

## Inference-speed boundary

This experiment cannot make generation faster. Every trained arm still evaluates all
12 GPT-2 blocks at width 768, and the residual projection is absent during generation.
Throughput is timed as a guardrail and should be statistically the same across arms.
Actual speedups require a second experiment that changes inference computation—for
example a low-rank adapter implementation, structured width pruning followed by
distillation, block skipping, or early exit—and then measures latency at matched
accuracy and batch size.

Implementation:

- `src/mech/endpoint_retention.py`: frozen rank matching and residual filters.
- `scripts/run_official_codi_endpoint_retention.py`: one resumable arm/seed run.
- `scripts/analyze_official_codi_endpoint_retention.py`: paired hierarchical analysis.
- `tests/test_endpoint_retention.py`: invariants and synthetic analysis tests.
