# A Global Low-Rank Vocabulary Head for CODI

**Research report and experiment design**

**Date:** 3 September 2026

**Audience:** Muhammad Jon Raza and collaborators working on CODI inference

## Scope and direct answer

This report asks whether the original CODI vocabulary head can be replaced at every
**visible answer-generation step** by a low-rank head that improves end-to-end inference
time while preserving the model's GSM8K accuracy. Speculative decoding is excluded.
CODI's transformer, six continuous latent-reasoning passes, tokenizer, and full 50,257-token
vocabulary remain unchanged.

The problem is only partially solved in the literature. Low-rank, activation-aware,
task-weighted, quantized, hierarchical, and retrieval-based output layers are established,
but no general technique guarantees that an aggressively low-rank, verifier-free replacement
will preserve the complete output of an arbitrary pretrained language model. For CODI, the
existing learned rank-64 head is already a promising global visible-token head: it scored
42.002% against the full model's 43.366%, accelerated the isolated head by about 4.59x, and
accelerated the measured end-to-end run by about 1.122x. The open problem is to close the
remaining 1.364-point accuracy gap and establish the speedup under a rigorous latency protocol.

The recommended next design is a **trajectory-aware, activation- and decision-weighted,
nested low-rank head**:

1. collect final-normalized hidden states at every visible answer position, not only the colon;
2. initialize a rank-64 head with the known answer subspace plus an activation-whitened
   correction fitted to trajectory-wide logit error;
3. distil token distributions and ranking margins from the original head;
4. add on-policy states produced by the compressed model itself;
5. train nested rank blocks so easy tokens use rank 32 and uncertain tokens add rank 32 or 64;
6. optionally correct a small set of numerically and structurally critical vocabulary rows;
7. deploy with fused, quantized low-rank kernels and measure normalized tokens per second.

## 1. The exact problem

Let the final normalized CODI hidden state for visible output position \(t\) be

\[
h_t\in\mathbb{R}^{768}.
\]

The original head is

\[
z_t=W h_t,\qquad W\in\mathbb{R}^{50{,}257\times768}.
\]

Ignoring bias and argmax, this requires

\[
50{,}257\times768=38{,}597{,}376
\]

multiply-accumulate operations per visible token. A rank-\(r\) replacement is

\[
\hat z_t=b+A_rB_r(h_t-\mu),
\]

with

\[
B_r\in\mathbb{R}^{r\times768},\qquad
A_r\in\mathbb{R}^{50{,}257\times r}.
\]

Its approximate MAC count is

\[
r(768+50{,}257)=51{,}025r.
\]

| Head | MACs per visible token | Arithmetic reduction vs full |
|---|---:|---:|
| Full | 38,597,376 | 1.00x |
| Rank 32 | 1,632,800 | 23.64x |
| Rank 64 | 3,265,600 | 11.82x |
| Rank 96 | 4,898,400 | 7.88x |
| Rank 128 | 6,531,200 | 5.91x |

These are operation counts, not end-to-end speedups. The transformer and CODI latent passes
are unchanged, the two low-rank matrix products are sequential, and all 50,257 output scores
are still produced.

## 2. Why the problem cannot have a universally exact low-rank solution

If \(W\) has numerical rank \(q\), an exact factorization \(W=A_rB_r\) for every possible
hidden state requires \(r\ge q\). With \(r<q\), there is always some hidden vector for which
the reconstructed logits differ. Rank 768 can reproduce the original head exactly but offers
no useful inner-dimension reduction. Rank 32 or 64 must therefore be approximate unless all
hidden states encountered in the deployment distribution occupy a correspondingly small
task subspace.

This is why the local CODI finding does not automatically solve the global problem. Colon
states occupy a narrow semantic regime: the model has completed its latent reasoning and is
about to commit to an answer. Later visible positions must additionally represent number
continuations, signs, decimal points, punctuation, explanation tokens, and termination. The
global state distribution is broader.

The classical softmax-bottleneck literature also warns that restricting the effective output
rank limits the family of contextual probability distributions a language model can express.
The original result formulated language modeling as a matrix-factorization problem and used
mixtures to recover higher-rank distributions
([Yang et al., ICLR 2018](https://arxiv.org/abs/1711.03953)). This does not prove that CODI
requires rank 768 on GSM8K, but it explains why aggressive global rank reduction can damage
generation even when a local answer decision is low-dimensional.

## 3. What the completed CODI experiments establish

The local confirmation used the pinned official CODI checkpoint and all 1,319 GSM8K test
questions. The full forced-cue baseline was 43.366%. Retaining PCs 4-31 at the answer cue
scored 38.06%, or 87.8% of baseline; retaining PCs 0-31 scored 40.94%, or 94.4% of baseline.
The isolated rank-32 readout benchmark was 117 microseconds versus 1,327 microseconds for the
full head, but this was a component benchmark and not an end-to-end result. See the
[endpoint-band confirmation](../OFFICIAL_CODI_ENDPOINT_BAND_CONFIRMATION.md) and
[research ledger](../RESEARCH_CONTEXT_LEDGER.md).

The subsequent learned global visible-token experiment showed:

| System | GSM8K exact match | Relative to full | Head latency | End-to-end speed |
|---|---:|---:|---:|---:|
| Full CODI | 43.366% | 100% | 637 us | 1.000x |
| Fixed eigen rank 32 | 5.080% | 11.7% | -- | -- |
| Learned random rank 32 | 22.517% | 51.9% | -- | -- |
| Learned eigen rank 32 | 40.334% | 93.0% | 128 us | 1.136x |
| Learned eigen rank 64 | 42.002% | 96.85% | 139 us | 1.122x |

The interpretation has three parts:

- low-rank shape supplied the computational reduction;
- answer-eigenspace initialization supplied a large quality advantage over random
  initialization at the same rank;
- training was necessary because a fixed endpoint basis did not cover the full visible
  generation trajectory.

The results do not show that eigen initialization itself makes a matrix product faster. At a
fixed rank, initialization changes learned quality, not the runtime shapes.

## 4. What related work contributes

### 4.1 Activation-aware and truncation-aware factorization

Vanilla SVD minimizes weight reconstruction error \(\lVert W-\hat W\rVert_F\), even though
the model only uses \(W\) on its actual activation distribution. ASVD rescales or whitens
weights using calibration activations and reports that activation distribution and layer
sensitivity are central to successful low-rank compression
([Yuan et al., 2023](https://arxiv.org/abs/2312.05821)). SVD-LLM makes this relationship
explicit through activation whitening and a closed-form recovery update, and reports real
generation-speed improvements when applied across LLM matrices
([Wang et al., ICLR 2025](https://arxiv.org/abs/2403.07378)).

For a head-only approximation, let centered trajectory states have covariance

\[
C_h=\mathbb{E}[(h-\mu)(h-\mu)^T],\qquad C_h=LL^T.
\]

Then the activation-weighted squared-logit error is

\[
\mathbb{E}\lVert(W-\hat W)(h-\mu)\rVert_2^2
=\lVert(W-\hat W)L\rVert_F^2.
\]

Therefore, a better closed-form starting point than weight-only SVD is to truncate the SVD of
\(WL\), then transform back by \(L^{-1}\). This is directly applicable to CODI's trajectory
states.

### 4.2 Task- and loss-weighted factorization

Fisher-Weighted SVD argues that equal reconstruction error is the wrong objective because
different parameters affect task loss differently. It weights reconstruction by estimated
Fisher importance and reports better task preservation than ordinary SVD
([Hsu et al., ICLR 2022](https://arxiv.org/abs/2207.00112)). The authors explicitly note that
this approach is particularly appropriate for task-specific models, which fits CODI on GSM8K.

For CODI, the corresponding lesson is to weight:

- states near answer commitment;
- teacher top-1 and top-k ranking margins;
- numerical tokens, signs, decimal separators, punctuation, and EOS;
- examples whose full-head margins are small and are therefore easy to flip.

### 4.3 Adaptive rank

Rank need not be constant. NAACL 2024 work learns rank allocation rather than imposing a
uniform rank across operations
([Gao et al., 2024](https://aclanthology.org/2024.naacl-long.13/)), and LREC 2026 work learns
masks over singular directions to select rank under a compression objective
([Sundrani et al., 2026](https://aclanthology.org/2026.lrec-1.787/)). FLRC further reports that
generation quality improves when more rank is allocated to early generated tokens and rank is
reduced later
([Qiu et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.755/)). These methods are
not head-specific solutions, but they support a nested rank schedule for CODI.

### 4.4 Whole-width structured compression

SliceGPT rotates transformer representations and removes low-importance rows and columns,
producing smaller dense matrices rather than sparse matrices. It reports real hardware gains
and up to 25% parameter removal with high retained zero-shot performance on some tested models
([Ashkboos et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/316648eb8b4ffb6010f531b07848c300-Paper-Conference.pdf)).
This is relevant if head-only optimization reaches its ceiling: shrinking the residual width
can accelerate both CODI's transformer and its head, but it is a substantially more invasive
experiment because the continuous latent projector and all 12 blocks must be transformed.

### 4.5 Alternatives that avoid dense full-vocabulary scoring

Adaptive softmax groups tokens by frequency and avoids calculating every tail score for every
example, but it changes the output architecture and normally must be used during model training
([Grave et al., ICML 2017](https://arxiv.org/abs/1609.04309)). Hierarchical softmax has similar
architectural and retraining costs.

Maximum inner-product search treats \(\arg\max_j w_j^Th\) as a retrieval problem. Earlier work
established approximate inference using MIPS for large output spaces
([Mussmann and Ermon, ICML 2016](https://proceedings.mlr.press/v48/mussmann16.html)). A recent
preprint replaces output projection with an HNSW vector index and reports up to 82% higher
batch-one CPU throughput for a small Gemma model, with over 99% index recall at a higher search
depth; its experiments are CPU-focused and approximate, so it is not yet direct evidence for
CODI on a T4 GPU
([Loretz and Hochreiter, 2026](https://arxiv.org/abs/2608.27460)).

VQ-Logits is another recent preprint that clusters output embeddings into a small codebook and
reports up to 6x logit-computation speed with a perplexity tradeoff
([Shao et al., 2025](https://arxiv.org/abs/2505.10202)). It is a useful baseline but is less
natural for preserving fine distinctions among numerical tokens.

### 4.6 Quantization and low-rank-plus-residual heads

Quantization reduces memory traffic without forcing a low-dimensional functional map. GPTQ
reported 3-4-bit post-training quantization with end-to-end speedups on its tested hardware
([Frantar et al., MLSys 2023](https://arxiv.org/abs/2210.17323)), while AWQ uses activation-aware
scaling and a hardware-specific runtime
([Lin et al., MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)).

ARCHead is a particularly relevant August 2026 preprint: it combines a quantized low-rank core,
group-wise INT4 residuals, and an activation-metric low-rank correction. It reports 3.7-3.9x
head-storage reduction and near-baseline perplexity, but less than 2% throughput change in its
measurements
([Kocabay et al., 2026](https://arxiv.org/abs/2608.02703)). This is evidence that storage
compression and inference acceleration must be evaluated separately.

## 5. Recommended CODI architecture

### 5.1 Global trajectory-aware base

Collect \(h_t\) after `ln_f` for **every visible answer token** under the original CODI head.
Include the first answer token, all number pieces, punctuation, and the EOS-decision state.
Construct a regularized covariance

\[
C_h=\frac{1}{N}\sum_t(h_t-\mu)(h_t-\mu)^T+\epsilon I.
\]

Initialize the first 32 dimensions with the established PC 0-31 basis because it retained
94.4% of local exact-match accuracy. Fit the next 32 dimensions to the residual trajectory
logit error using activation-whitened SVD. In words:

```text
rank 0-31   = preserve the known answer-commitment geometry
rank 32-63  = repair vocabulary behavior over all later answer positions
```

This hybrid is more defensible than selecting all 64 dimensions solely by hidden-state
variance.

### 5.2 Nested adaptive rank

Parameterize the head in ordered blocks:

\[
\hat z^{(32)}=b+A_1B_1(h-\mu),
\]

\[
\hat z^{(64)}=\hat z^{(32)}+A_2B_2(h-\mu),
\]

\[
\hat z^{(96)}=\hat z^{(64)}+A_3B_3(h-\mu).
\]

First calculate rank 32. If the top-1/top-2 logit margin is safely above a calibrated threshold,
emit the token. Otherwise add the next rank block. EOS and the first answer token can be forced
to at least rank 64. This remains a global low-rank head: it never calls the original full output
matrix, but it spends more rank on difficult positions.

The primary deployment target should be full-model-quality rank 64 with an average activated
rank below 48. Fixed rank 64 remains the control.

### 5.3 Critical-token residual

Define a small set \(S\) containing:

- every token used in training-set gold numeric answers;
- sign, decimal, comma, fraction, bracket, and boxed-answer tokens;
- EOS and newline tokens;
- the most frequent teacher top-k tokens over answer trajectories.

For \(j\in S\), store the exact residual row

\[
R_j=W_j-(A_rB_r)_j
\]

and correct only those logits:

\[
\hat z_j\leftarrow\hat z_j+R_j(h-\mu).
\]

With rank 64 and \(|S|=512\), the approximate cost is

\[
64(50{,}257+768)+512(768)=3{,}658{,}816
\]

MACs, still about 10.55x below the full head's arithmetic. This branch directly protects the
tokens most likely to determine numeric exact match and correct termination. It is an optional
enhancement; the pure low-rank arm must remain in the experiment so its contribution is visible.

### 5.4 Training loss

Freeze the complete CODI backbone and train only the head. For teacher distribution \(p_t\),
compressed distribution \(q_t\), teacher top token \(y_t^*\), and gold token \(y_t\), use

\[
\mathcal L =
\lambda_{KL}T^2D_{KL}(p_t^T\Vert q_t^T)
+\lambda_{CE}[-\log q_t(y_t)]
+\lambda_{rank}\max(0,\gamma-\hat z_{t,y_t^*}+\max_{j\ne y_t^*}\hat z_{t,j})
+\lambda_{nest}\mathcal L_{32}.
\]

The terms respectively preserve the full teacher distribution, protect the correct token,
protect the teacher's ranking margin, and make the rank-32 prefix independently usable. Increase
the example weight for EOS, numeric pieces, low-margin states, and states at which previous
compressed heads diverged.

### 5.5 On-policy recovery

Teacher-forced hidden states are insufficient because one compressed-head error changes the next
visible token and therefore later hidden states. Use iterative dataset aggregation:

1. train on original CODI trajectories;
2. generate with the compressed head;
3. collect the hidden states actually visited by the compressed model;
4. label each with the frozen full head's logits and gold answer;
5. mix these states into the training set and retrain;
6. repeat until divergence and EOS failures stabilize.

This is the most important change for converting token-level fidelity into sequence accuracy.

### 5.6 Runtime engineering

For greedy CODI decoding, softmax is unnecessary: argmax of logits equals argmax of softmax.
Fuse centering, the down-projection, up-projection, critical-row correction, and argmax as far as
the backend permits. Quantize \(A_r\), which is the large bandwidth-dominant factor, to INT8
first; compare INT4 only after INT8 passes quality gates. Report compiled and eager baselines
separately.

GPT-2 ties the output matrix to the input token embeddings. Replacing only the output computation
does not automatically eliminate the original embedding table because it is still needed for
input lookup. The primary benefit is reduced repeated output projection, not necessarily an
equivalent reduction in persistent model storage.

## 6. Experimental ladder

### 6.1 Locked arms

| ID | Arm | Purpose |
|---|---|---|
| B0 | Original full CODI | Accuracy and latency baseline |
| B1 | Weight-SVD rank 32/64/96 | Weight-only baseline |
| B2 | Hidden-covariance eigen rank 32/64/96 | Existing geometric baseline |
| B3 | Activation-whitened SVD rank 32/64/96 | Trajectory-aware closed-form baseline |
| B4 | Fisher/task-weighted factorization | Loss-aware baseline |
| B5 | Random-init distilled rank 32/64/96 | Tests whether any training suffices |
| B6 | Eigen-init distilled rank 32/64/96 | Reproduces current method |
| B7 | Hybrid answer-plus-trajectory rank 64 | Proposed static head |
| B8 | Nested adaptive 32/64/96 | Proposed adaptive head |
| B9 | B8 plus critical-token residual | Proposed quality-recovery head |

Use disjoint fit, validation, on-policy-recovery, and final test partitions. Select rank,
thresholds, critical-token set size, and training epoch without reading final-test accuracy.

### 6.2 Metrics

Primary quality metrics:

- GSM8K numeric exact match;
- relative baseline accuracy retained;
- paired changed-correct/changed-wrong counts;
- teacher top-1 agreement at every visible position;
- sequence-level answer agreement;
- EOS and truncation rate.

Diagnostic metrics:

- KL divergence and centered-logit MSE;
- agreement by token position and token type;
- first-error position;
- teacher margin at errors;
- rank-32/64/96 activation frequency;
- on-policy divergence rate.

Efficiency metrics:

- head latency at batch 1, 8, and 32;
- end-to-end latency per question;
- generated tokens per second;
- latency normalized by generated token count;
- memory bandwidth and kernel time from a profiler;
- peak memory and stored head bytes.

### 6.3 Success gates

The main quality gate should be at least 98% relative retention:

\[
0.98\times43.366\%=42.499\%.
\]

This is only about 0.50 points above the existing rank-64 result. Require a paired bootstrap
confidence interval and at least three training seeds.

The efficiency gate should require:

- at least 4x isolated head speedup;
- at least 1.12x end-to-end batch-one speedup with a positive lower confidence bound;
- no material increase in mean generated length, truncation, or EOS failure.

A stretch target is at least 1.15x end-to-end speed with 98% retained accuracy.

## 7. The head-only speed ceiling

The current rank-64 measurement changed head latency from 637 to 139 microseconds and produced
1.122x end-to-end speedup. If those measurements are treated as one consistent timing model,
they imply that the full head was roughly 14% of baseline latency. Making the head infinitely
fast would then yield only about 1.16x total speedup. Repeating the estimate with the rank-32
numbers gives a rough ceiling near 1.18x. These are diagnostic estimates, not new measurements,
because the microbenchmark and end-to-end timing may not share every condition.

Consequently, a head-only project should target approximately 1.12-1.16x end-to-end acceleration,
not a multi-fold CODI speedup. Larger gains require optimizing the transformer or latent budget.
The existing separate result that five latent passes matched or slightly exceeded six should be
retested at batch one only after the global-head experiment is stable.

## 8. Is the problem solved?

| Question | Status |
|---|---|
| Can large language-model matrices be compressed modestly with low rank? | Largely established. |
| Can activations and task sensitivity improve SVD? | Established by multiple methods. |
| Can a vocabulary head be quantized or approximately retrieved faster? | Demonstrated, but hardware- and method-dependent. |
| Can rank 32/64 exactly replace any full head for every hidden state? | No, unless the relevant map/state distribution has that rank. |
| Can a task-specific global low-rank CODI head retain nearly all GSM8K accuracy? | Promising but not settled; rank 64 currently retains 96.85%. |
| Can head-only changes make CODI several times faster end to end? | No under the current latency composition; the head is only one component. |

The publishable open question is therefore not whether low-rank factorization exists. It is:

> Can a causal answer subspace, combined with trajectory-aware and decision-weighted recovery,
> produce a verifier-free global CODI head that reaches at least 98% retained exact-match accuracy
> while sustaining a measured end-to-end speedup?

## 9. Recommended next implementation order

1. Reproduce the current rank-64 global head and timing with exact generated-token accounting.
2. Collect all visible answer-trajectory states and teacher logits, including EOS states.
3. Add weight-SVD, activation-whitened SVD, and Fisher-weighted baselines.
4. Build the hybrid rank-64 initialization: PC 0-31 plus 32 trajectory-residual directions.
5. Distil with KL, cross-entropy, and ranking-margin objectives.
6. Add two rounds of on-policy state aggregation.
7. Add nested 32/64/96 inference and calibrate its threshold on validation only.
8. Add the 512-token critical residual only if EOS or numeric tokens remain dominant errors.
9. Implement INT8/fused inference after the quality winner is frozen.
10. Run three seeds and the sealed full-GSM8K test once.

## 10. Claim-to-source ledger

| Claim family | Primary source | Status and limitation |
|---|---|---|
| Low-rank output distributions face an expressivity bottleneck | Yang et al., ICLR 2018, https://arxiv.org/abs/1711.03953 | Foundational theory; originally evaluated on RNN LMs, used here as a general rank warning. |
| Activation distributions should weight SVD | Yuan et al., ASVD, https://arxiv.org/abs/2312.05821 | Primary preprint; whole-model focus, not CODI head-specific. |
| Whitening links truncation to activation error and can improve real inference | Wang et al., SVD-LLM, https://arxiv.org/abs/2403.07378 | ICLR 2025 paper; mostly whole-model compression. |
| Fisher weighting better aligns factorization with task importance | Hsu et al., https://arxiv.org/abs/2207.00112 | ICLR 2022; task-specific transformer models, not CODI. |
| Rank allocation is non-uniform and learnable | Gao et al., NAACL 2024, https://aclanthology.org/2024.naacl-long.13/ | Whole-model ranks rather than token-wise head rank. |
| Differentiable singular-direction selection improves low-rank compression | Sundrani et al., LREC 2026, https://aclanthology.org/2026.lrec-1.787/ | Fine-tuning-free rank selection; not output-head specific. |
| Progressive rank schedules can improve generation quality | Qiu et al., EMNLP 2025, https://aclanthology.org/2025.emnlp-main.755/ | Compresses transformer projections; supports but does not validate the proposed head schedule. |
| Dense width slicing yields real hardware gains | Ashkboos et al., ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/316648eb8b4ffb6010f531b07848c300-Paper-Conference.pdf | More invasive than a head replacement. |
| Adaptive softmax avoids full tail computation | Grave et al., ICML 2017, https://arxiv.org/abs/1609.04309 | Requires a changed/trained output architecture. |
| MIPS can approximate inference over large outputs | Mussmann and Ermon, ICML 2016, https://proceedings.mlr.press/v48/mussmann16.html | Approximate retrieval, not identical full logits. |
| HNSW output embeddings improve small-batch CPU throughput | Loretz and Hochreiter, https://arxiv.org/abs/2608.27460 | Recent preprint; CPU-focused and approximate. |
| Vector-quantized logits offer a head speed/quality tradeoff | Shao et al., https://arxiv.org/abs/2505.10202 | Preprint; requires independent replication for CODI. |
| Weight quantization can yield real inference speed | Frantar et al., GPTQ, https://arxiv.org/abs/2210.17323; Lin et al., AWQ, https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html | Strong whole-model evidence; kernel and hardware dependent. |
| Low-rank plus quantized residual can preserve output-head perplexity | Kocabay et al., ARCHead, https://arxiv.org/abs/2608.02703 | August 2026 preprint; reports storage benefit but under 2% throughput change. |

## Research stopping note

The search covered the principal method families that could change the answer: low-rank SVD and
its activation/task-aware variants, learned rank allocation, structured width reduction, adaptive
softmax, vocabulary pruning, maximum-inner-product retrieval, vector-quantized output heads, and
quantized residual heads. The evidence converged on the same boundary: modest compression and
approximate acceleration are established, while aggressive verifier-free global head replacement
remains distribution- and hardware-dependent. Additional broad searching was unlikely to change
the recommended CODI experiment; the next decisive evidence must come from matched CODI runs.
