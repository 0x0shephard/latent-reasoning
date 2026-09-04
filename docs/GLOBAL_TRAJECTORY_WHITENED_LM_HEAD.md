# Trajectory-whitened global low-rank LM head

## Status

Implemented and awaiting the locked Kaggle run. No accuracy or speed result is
claimed until the generated `summary.json` and per-arm JSONL records are inspected.

## Primary question

Can one shared low-rank vocabulary head replace the official CODI head at every
visible answer-generation position, retain at least 98% of the reproduced GSM8K
exact-match accuracy, and reduce normalized end-to-end generation time?

This differs from the earlier endpoint and position-conditioned experiments:

- no test state is used for fitting or model selection;
- the fitted states cover every visible answer position;
- one head is shared across positions;
- the initialization minimizes activation-weighted logit reconstruction error;
- clean and compressed-policy states are both used for distillation;
- rank 32, 64, and 96 are ordered prefixes of the same trained head.

## Method

For final-normalized states with mean `mu` and regularized covariance

```text
C = E[(h - mu)(h - mu)^T] + lambda I = S S^T,
```

the head objective is

```text
E ||(W - W_hat)(h - mu)||^2 = ||(W - W_hat) S||_F^2.
```

The implementation obtains a randomized truncated decomposition of `W S` without
materializing that large matrix. The two inference factors implement

```text
W_hat = W_up W_down
```

and the output bias exactly preserves the full head at `mu`. Singular directions
are stored in descending order, making ranks 32, 64, and 96 nested prefixes.

The frozen full head then teaches each prefix with three signals:

1. forward KL divergence over the full vocabulary;
2. cross-entropy toward the full head's top token;
3. a hinge loss protecting the full head's token-ranking margin.

After clean training, the rank-64 head generates on a disjoint recovery population.
The states reached under its own mistakes are collected and labelled by the frozen
full head for a second training round.

## Locked populations

- 1,024 GSM8K training questions for clean trajectory fitting;
- 256 disjoint training questions for validation and threshold selection;
- 256 disjoint training questions for on-policy recovery;
- all 1,319 GSM8K test questions for the final evaluation.

At most 4,096 clean fit states, 1,024 selection states, and 2,048 on-policy states
are retained through seeded subsampling. The complete test is opened only after the
head and adaptive threshold have been frozen.

## Final arms

| Arm | Meaning |
| --- | --- |
| `full` | Original official CODI vocabulary head |
| `whitened_margin_onpolicy_r32` | Cheapest nested prefix |
| `whitened_margin_onpolicy_r64` | Primary fixed-rank candidate |
| `whitened_margin_onpolicy_r96` | Quality-recovery prefix |
| `adaptive_r32_r64` | Rank 32 with validation-selected low-margin fallback to 64 |

The adaptive threshold must reach 98% teacher top-token agreement on the selection
states when possible. That is only a selection statistic; the locked autoregressive
test determines whether sequence-level accuracy is retained.

## Gates and limitations

The primary rank-64 arm passes only if:

1. the untouched full head reproduces the official baseline;
2. rank 64 retains at least 98% of that run's exact-match accuracy; and
3. rank 64 improves microseconds per visible generated token.

The notebook reports isolated-head timing separately. Arithmetic reduction is not
treated as runtime evidence. Both the full and compressed arms use the same
transformer-body fast path, so neither computes prompt or latent vocabulary logits
that are discarded. A mandatory decoded-string parity gate compares this path with
the released decoder before fitting. The ordinary PyTorch two-GEMM head is a portable reference
implementation, not a fused deployment kernel, so this run can establish the quality
ceiling and portable speed before kernel specialization.

The factorization code is model-independent, but the primary notebook's causal task
adapter is CODI-specific. A companion notebook applies the identical fitter to pinned
Qwen2.5-Math-1.5B-Instruct trajectories. The numeric CODI factors themselves are never
copied into Qwen's hidden coordinate system. A third unrelated family and non-math
evaluation are still required before making a broad generalization claim.

## Files

- core implementation: `src/mech/global_low_rank_head.py`
- notebook builder: `scripts/build_kaggle_global_low_rank_head_notebook.py`
- Kaggle notebook: `notebooks/kaggle_global_low_rank_lm_head.ipynb`
- Qwen notebook builder: `scripts/build_kaggle_global_low_rank_head_qwen_notebook.py`
- Qwen companion notebook: `notebooks/kaggle_global_low_rank_lm_head_qwen.ipynb`
- unit tests: `tests/test_global_low_rank_head.py`
