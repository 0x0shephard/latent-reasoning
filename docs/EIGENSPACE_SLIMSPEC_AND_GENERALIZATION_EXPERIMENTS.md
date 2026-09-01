# Eigenspace readout: distillation and cross-model generalization

## Experiment 1: eigenspace-initialized learned CODI head

Question: can SlimSpec-style logit distillation recover the accuracy lost by CODI's
fixed eigenspace projection while retaining its low-rank computation?

The official CODI checkpoint, latent computation, tokenizer, full vocabulary, and
greedy decoding protocol remain frozen. A rank-32 head is initialized as

```text
W_down = U_32^T
W_up   = W U_32
```

with the centring terms required to reproduce
`mu W^T + ((h - mu) U_32)(W U_32)^T`. The two factors are then trained to match the
full head's logits at answer-generation states. A random-basis rank-32 head receives
the same training and operation budget. Rank 64 is a recovery arm.

Primary comparisons:

- learned eigenspace rank 32 versus fixed eigenspace rank 32;
- learned eigenspace rank 32 versus learned random-basis rank 32;
- each compressed head versus the unchanged full head.

Primary support requires at least 98% baseline exact-match retention, no loss versus
the fixed rank-32 arm, and an end-to-end speed improvement. The notebook additionally
reports first-token agreement, KL divergence, paired bootstrap intervals, isolated
head latency, end-to-end latency, and MAC counts.

Notebook: `notebooks/kaggle_codi_eigenspace_distilled_readout.ipynb`

Required Kaggle inputs:

- completed official CODI reproduction `summary.json`;
- completed margin-geometry `colon_states.pt`;
- matching `readout.pt`.

## Experiment 2: frozen readout generalization to Qwen

Question: does the eigenspace-selection principle work outside CODI on a conventional
autoregressive model?

The target is `Qwen/Qwen2.5-Math-1.5B-Instruct`, with hidden width 1,536. Fit and
selection rows come from GSM8K train; test rows come from GSM8K test. The experiment
fits an eigendecomposition on prompt-endpoint states and compares, at equal rank:

- leading-variance eigenvectors;
- the literal CODI-inspired rule that skips four leading directions;
- a portable readout-aware score: eigenvalue times relative-logit energy;
- seeded random orthonormal bases.

Ranks 32 and 64 distinguish an absolute-rank transfer from a width-fraction transfer.
The primary rank is 64 because it is approximately the Qwen-width counterpart of
CODI rank 32/768.

Generalization is supported only if the rank-64 readout-aware head retains at least
95% of baseline exact match, preserves at least 90% prompt-endpoint first-token
agreement, beats the matched random arm by at least five points with a positive
paired-bootstrap lower bound, and accelerates the isolated head by at least 3x. Fewer
than 50 correct baseline test answers makes the accuracy comparison inconclusive.

Notebook: `notebooks/kaggle_eigenspace_readout_generalization_qwen.ipynb`

## Interpretation

Experiment 1 tests whether the eigenspace is a useful initialization for a learned
factorization. Experiment 2 tests whether the post-training selection principle is
portable. They answer different questions and should not be merged into one success
claim. A positive result in one does not rescue a failed gate in the other.
