# CODI task-aware, causally protected xKV experiment

## Question

Can CODI's independently confirmed, answer-sensitive K/V directions improve the
quality–storage frontier of xKV, and do calibration-fitted attention sensitivities
help beyond ordinary cross-layer SVD?

This experiment is deliberately stricter than the exploratory layerwise runs. It
separates discovery, confirmation, compression calibration, and final evaluation.

## Frozen evidence and new data splits

The split is deterministic and excludes every GSM8K test question from training
calibration. In sampling order it uses:

1. 2,432 training questions from the original native-cache discovery:
   1,024 covariance fit, 1,024 task discovery, 256 rank selection, and 128 causal
   screening.
2. The next 512 training questions from the completed layer-11 confirmation.
3. The next 512 unseen training questions to confirm the frozen layer-2 `(K8,V48)`
   and layer-3 `(K16,V64)` hypotheses.
4. The next 512 unseen training questions to fit answer-gradient feature weights
   and group utilities.
5. Untouched GSM8K test questions for all reported compression results.

The runner reconstructs these boundaries and verifies question hashes. A prior
confirmation artifact must name the exact source artifact SHA-256.

## Early-layer confirmation

Layers 2 and 3 are co-primary frozen hypotheses. For each layer, the experiment
removes its discovered K/V subspace and compares the answer-NLL damage with eight
energy-matched random subspaces. A layer passes only when:

- its Bonferroni-corrected bootstrap interval is strictly above zero;
- at least three of four disjoint folds have positive specificity;
- retaining only the subspace preserves at least 95% dense first-token agreement;
- retain-only answer NLL rises by no more than 0.10.

Layer 11 is imported only from the preceding independent 512-question
confirmation. Failed early layers are excluded from causal protection. The strict
combined claim passes only if layers 2, 3, and 11 all pass.

## Compression methods

For each xKV layer group, concatenate keys and values into

`X_g in R^(tokens x (2 * group_layers * hidden_width))`.

The ordinary xKV reference applies a rank-`r` truncated SVD to `X_g`.

### Answer-Fisher weighted xKV

On the separate calibration split, estimate a fixed feature sensitivity

`s_j = sqrt(E[(dL / dX_j)^2])`.

Then factorize `X_g diag(s)` and undo the scale in the decoder. This minimizes
weighted reconstruction error. Because the loss gradient with respect to K/V
passes through QK scores, softmax, and value aggregation, it is an
attention-sensitive proxy. It is not an exact KQ-SVD implementation and is not
fitted on test answers.

### Sparse causal residual

Putting every protected feature direction inside xKV's shared token factor would
spend 136 rank units for layers 2 and 3 alone. Instead, the method applies xKV to
the bulk cache and stores exact correction coordinates only for CODI's six latent
rows:

`X_final = X_xKV + M_latent (X - X_xKV) U U^T`.

`U` is fixed model metadata. Per request, the cache stores only
`(X - X_xKV)U` for the six latent rows. The storage report includes these residual
coordinates but does not repeatedly charge the fixed basis to every request.

### Layer-adaptive rank

The full arm distributes the same total grouped rank budget according to
calibration gradient utility, subject to each group's matrix-rank cap. Diminishing
returns prevent the complete budget from collapsing into one group.

## Arms and decisions

The notebook evaluates ranks 16, 32, and 48 for:

- independent per-layer SVD;
- ordinary grouped xKV SVD;
- answer-Fisher xKV;
- causal-residual xKV;
- the full weighted, adaptive, causally protected method.

At rank 32 it also evaluates 20 energy-matched random protected-residual controls.
Symmetric INT8 and INT4 dense-cache controls and INT8/INT4 factor quantization are
reported separately; these are quality/storage proxies, not KIVI kernels.

The full method is a same-rank quality win only when the paired bootstrap interval
for `NLL_ordinary_xKV - NLL_full` is above zero. Because causal residuals use extra
bits, the report also gives the exact full-to-ordinary cache-bit ratio. A credible
claim additionally requires the causal basis to beat the matched-random residuals
at the focal rank and the quality–storage curve to dominate at comparable storage.

## Metrics

Teacher-forced evaluation reports answer NLL, answer-perplexity proxy, first-token
accuracy, dense top-1 agreement, top-5 overlap, KL divergence from dense logits,
and gold-margin change. Greedy generation reports GSM8K accuracy and exact sequence
agreement with dense CODI. Every arm records modeled cache bits and compression
ratio.

## What this experiment cannot establish

The implementation reconstructs dense caches so the unmodified Transformers model
can consume them. Timing includes SVD plus dense reconstruction. It therefore does
not support a speed claim. A production follow-up must implement reduced-coordinate
attention, packed residual storage, and fused INT4/INT8 kernels, then benchmark
prefill, time-to-first-token, decode latency, peak memory, and energy.

## Kaggle input

The only mandatory input is the completed official CODI reproduction dataset that
contains:

`official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json`.

If available, also attach the earlier `direct_cache_task_subspaces.pt`, previous
confirmation `summary.json`, and `confirmed_native_kv_artifact.pt`. Otherwise the
notebook deterministically reruns those predecessors. When it regenerates the
source artifact, it also regenerates the dependent confirmation so SHA lineage is
valid.

## Outputs

- `summary.json`: inspectable gates, metrics, storage, paired comparisons, warnings.
- `task_aware_protected_xkv.pt`: bases, weights, per-example deltas, controls, and
  generated outputs.
- `predictions.jsonl`: question-level dense and compressed generations.
- `codi_task_aware_protected_xkv.zip`: complete Kaggle audit trail.
