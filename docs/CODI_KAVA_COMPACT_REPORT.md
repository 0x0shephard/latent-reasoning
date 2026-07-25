# CODI vs KaVa: A Controlled and Mechanistic Study of Latent Reasoning Supervision

**Muhammad Jon Raza**  
20 July 2026

## Abstract

This paper compares CODI and KaVa as supervision strategies for continuous latent mathematical reasoning. Both methods were implemented in a shared GPT-2 framework and matched on architecture, training data, latent budget, optimizer, schedule, prompts, decoding, and evaluation. CODI uses endpoint hidden-state distillation, whereas KaVa adds key/value (KV) trajectory supervision compressed to six latent positions with R-KV. On the complete seed-zero evaluation, CODI obtained 11.28% macro accuracy and KaVa obtained 13.29%. The paired improvement was 2.01 percentage points, with a 95% paired-bootstrap confidence interval of 0.44 to 3.60 points. KaVa also outperformed CODI across all three capped matched seeds. Cross-example latent-state shuffling harmed KaVa 4.60 macro points more than CODI in a difference-in-differences analysis, indicating stronger use of example-specific latent information. The results establish a modest KaVa-over-CODI advantage in the controlled setting and show stronger causal dependence on the latent trajectory. The gain is concentrated on MultiArith, while compression controls do not yet isolate R-KV as its unique cause.

**Keywords:** latent reasoning, continuous chain of thought, self-distillation, KV-cache distillation, CODI, KaVa, causal intervention

## 1. Introduction

Explicit chain-of-thought reasoning improves language-model performance but requires the model to generate a potentially long sequence of natural-language reasoning tokens. Continuous latent-reasoning methods instead recurrently process hidden representations before producing an answer. Their central challenge is supervision: the final answer provides a weak learning signal for the internal trajectory, while detailed teacher targets may introduce additional cost or unwanted constraints.

CODI addresses this problem through self-distillation, aligning teacher and student hidden states near final-answer generation [1]. KaVa extends this objective with compressed teacher key/value trajectory supervision [2]. Because the two approaches are nested, they can be compared under a shared implementation in which the additional KV loss is the principal scientific difference.

This study replaces the broad question, "Which method is better?", with three testable questions:

1. Does KaVa outperform CODI under matched training and evaluation conditions?
2. Is the direction of the improvement consistent across independent training seeds?
3. Does KaVa make stronger causal use of its continuous latent states?

A secondary attribution question asks whether any advantage is specifically caused by R-KV compression or by denser trajectory supervision more generally.

## 2. Background

### 2.1 CODI

CODI trains a shared model as both explicit-reasoning teacher and latent-reasoning student. The student produces a fixed sequence of continuous states between `<bot>` and `<eot>`. Its objective combines student answer cross-entropy, teacher chain-of-thought and answer cross-entropy, and all-layer L1 hidden-state matching at the answer-cue endpoint:

```text
L_CODI = L_student_CE + L_teacher_CE + L_hidden
```

### 2.2 KaVa

KaVa retains the CODI terms and adds L1 matching between the student's latent key/value trajectory and a compressed explicit-teacher KV cache:

```text
L_KaVa = L_CODI + L_KV
```

The teacher trajectory is compressed independently for each layer and attention head. R-KV combines token importance and non-redundancy, then restores selected positions to chronological order before matching them to the student's latent positions.

## 3. Methodology

### 3.1 Controlled experimental setup

| Component | Setting |
| --- | --- |
| Backbone | GPT-2, fixed revision |
| Training data | 385,620 equation-style augmented GSM8K examples |
| Latent representation | Six autoregressive continuous states |
| Maximum sequence length | 256 tokens |
| Training budget | One epoch, 96,405 optimizer steps |
| Batch size | 4 |
| Learning rate | 1e-4, 500-step warmup, cosine decay |
| Weight decay and clipping | 0.1 and 2.0 |
| Evaluation decoding | Greedy, maximum 64 generated tokens |
| Metric | Numeric exact-match accuracy |

The primary comparison fixes all listed factors. KaVa differs from CODI by assigning weight 1.0 rather than 0.0 to all-layer, all-head KV trajectory matching. The experiment intentionally uses a common six-step autoregressive mechanism instead of mixing CODI's and KaVa's original latent budgets or update mechanisms.

### 3.2 Runs and controls

CODI and KaVa were trained at seeds 0, 1, and 2. Three seed-zero controls were also trained:

- **No trajectory distillation:** hidden and KV matching weights are zero, while shared student and teacher cross-entropy terms remain active.
- **Random compression:** teacher positions are selected deterministically at random.
- **Uniform compression:** teacher positions are selected at uniformly spaced intervals.

The causal analysis applies zero replacement, batch-mean replacement, and cross-example shuffling to the latent states. A separate KaVa sweep shuffles each of the six latent positions individually.

### 3.3 Evaluation datasets

| Dataset | Role | Full size | Capped analysis size |
| --- | --- | ---: | ---: |
| GSM8K | In-domain arithmetic | 1,319 | 200 |
| GSM-Hard | Numerically difficult arithmetic | 1,319 | 200 |
| SVAMP | Out-of-domain word problems | 300 | 200 |
| MultiArith | Multi-step arithmetic | 180 | 180 |

The primary seed-zero comparison was evaluated on all 3,118 examples per method. Controls, matched-seed comparisons, and causal interventions use the capped sets. The earlier CoT-SFT baseline was evaluated on the complete datasets.

### 3.4 Statistical analysis

Predictions are aligned by exact question and normalized numeric gold answer. Per-dataset comparisons report paired accuracy differences, 10,000-sample paired-bootstrap confidence intervals, and exact McNemar tests. Macro accuracy is the unweighted mean of the four dataset accuracies. Its confidence interval resamples paired questions independently within each dataset before averaging the four effects. This interval covers evaluation-question uncertainty conditional on the trained checkpoints. The three-seed analysis reports individual values and sample standard deviation rather than an asymptotic method-level interval.

## 4. Results

### 4.1 Full primary comparison

| Method | GSM8K | GSM-Hard | MultiArith | SVAMP | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| CODI | 12.59% | 2.65% | 18.89% | 11.00% | **11.28%** |
| KaVa | 13.12% | 2.50% | 25.56% | 12.00% | **13.29%** |
| KaVa - CODI | +0.53 pp | -0.15 pp | +6.67 pp | +1.00 pp | **+2.01 pp** |

KaVa improved macro accuracy by **2.01 percentage points**, with a 95% paired-bootstrap interval of **[+0.44, +3.60] points**. This corresponds to an approximately 17.8% relative increase over CODI's macro accuracy.

MultiArith is the only individual dataset with a paired interval excluding zero: **+6.67 points**, 95% CI **[+1.11, +12.22]**, exact McNemar `p = 0.02896`. Differences on GSM8K, GSM-Hard, and SVAMP are individually uncertain.

### 4.2 Stability from capped to full evaluation

| Metric | Capped evaluation | Full evaluation |
| --- | ---: | ---: |
| CODI macro | 11.72% | 11.28% |
| KaVa macro | 13.89% | 13.29% |
| KaVa - CODI | +2.17 pp | +2.01 pp |

The estimated improvement changed by only 0.16 percentage points after expanding the primary evaluation to the complete datasets.

### 4.3 Matched-seed consistency

| Seed | CODI macro | KaVa macro | Difference |
| ---: | ---: | ---: | ---: |
| 0 | 11.72% | 13.89% | +2.17 pp |
| 1 | 11.78% | 12.75% | +0.97 pp |
| 2 | 11.53% | 12.61% | +1.08 pp |
| **Mean +/- SD** | **11.68% +/- 0.13** | **13.08% +/- 0.70** | **+1.41 +/- 0.66 pp** |

KaVa outperformed CODI at every tested seed. The mean advantage was concentrated on MultiArith, where KaVa improved by 4.63 points across seeds.

### 4.4 Compression and distillation controls

| Seed-zero capped run | Macro accuracy |
| --- | ---: |
| CODI | 11.72% |
| No trajectory distillation | 13.19% |
| KaVa with uniform compression | 12.69% |
| KaVa with random compression | 13.40% |
| KaVa with R-KV | **13.89%** |

R-KV KaVa is numerically best, but its paired macro differences from random compression, uniform compression, and no trajectory distillation have confidence intervals that include zero.

### 4.5 Causal latent-state interventions

The difference-in-differences statistic is defined as:

```text
(KaVa intervention - KaVa baseline) - (CODI intervention - CODI baseline)
```

A negative value means that the intervention harmed KaVa more.

| Intervention | KaVa-vs-CODI effect | 95% paired CI |
| --- | ---: | ---: |
| Batch mean | -2.31 pp | [-4.14, -0.54] |
| Cross-example shuffle | **-4.60 pp** | **[-6.85, -2.42]** |
| Zero | -2.03 pp | [-4.31, +0.21] |

Cross-example shuffling gives the strongest causal evidence that KaVa depends more than CODI on example-specific latent information. In the KaVa position sweep, shuffling position 4 reduced macro accuracy from 13.89% to 10.26%, the largest decrease among the six positions. Positions 1 and 2 had smaller effects; positions 0, 3, and 5 had little aggregate effect.

### 4.6 Explicit chain-of-thought reference

The Phase 1 CoT-SFT model achieved 33.86% macro accuracy, including 74.44% on MultiArith. It remained substantially stronger in absolute accuracy than either latent model under the present training budget.

## 5. Discussion

### 5.1 Comparative performance

The full seed-zero result and all three capped matched seeds support the same directional conclusion: KaVa performs better than CODI in the controlled GPT-2 setting. The full macro interval excludes zero, and expansion from the capped to complete evaluation changes the estimated advantage only slightly.

The improvement is not uniform across tasks. MultiArith contributes most of the macro difference, while the other three datasets show small or uncertain effects. This pattern suggests that trajectory supervision may be most useful for compositional multi-step arithmetic, although task-level replication is required.

### 5.2 Causal use of latent states

The intervention results extend the performance comparison. Batch-mean replacement and cross-example shuffling harm KaVa more than CODI, indicating that KaVa's trajectory contains more example-specific information that is used during answer generation. The position sweep further suggests that this dependence is temporally localized, with position 4 contributing most strongly for the evaluated checkpoint.

### 5.3 Attribution to R-KV

The experiments do not yet establish that R-KV is the unique cause of the KaVa advantage. Random and uniform compression remain close to full KaVa, and the no-trajectory-distillation control also performs competitively. The current evidence supports the complete KaVa configuration relative to CODI, but attribution requires additional control seeds or a supervision-granularity continuum.

## 6. Limitations

The study is limited to GPT-2, six autoregressive latent positions, equation-style arithmetic traces, and one training epoch. Only the primary seed-zero pair has complete evaluation; additional seeds, controls, and causal interventions use capped sets. Three seeds provide directional reproducibility but do not support a precise population-level confidence interval. The unweighted macro score gives the 180-example MultiArith set the same weight as the larger datasets, and MultiArith drives most of the observed difference. The controlled architecture also differs from paper-specific larger-backbone and Jacobi configurations. Finally, efficiency, calibration, robustness, probes, activation patching, and the natural-language trace condition remain unevaluated.

## 7. Future work

Two research directions follow directly from the findings.

First, future experiments should distinguish **causally structured latent reasoning from additional serial computation**. A compute-matched study should compare endpoint hidden-state supervision, keys-only supervision, values-only supervision, full KV trajectories, and unsupervised recurrent-compute controls. Targeted activation patching can then test whether intermediate latent variables cause predictable changes in later computation and final answers.

Second, the empirical work can support a **systematic survey and taxonomy of latent-reasoning methods**. Methods can be organized by latent representation, supervision granularity, transition mechanism, causal evidence, and accuracy-efficiency trade-off, with CODI versus KaVa retained as the controlled case study.

The resulting central question is:

> At fixed latent compute, which supervision targets produce causally necessary intermediate representations: endpoint hidden states, keys, values, or complete KV trajectories?

## 8. Conclusion

Under matched architecture, data, training, and evaluation, KaVa achieved a consistent but modest advantage over CODI. Full seed-zero macro accuracy increased from 11.28% to 13.29%, and every capped matched seed favored KaVa. Causal interventions further showed that KaVa relies more strongly on example-specific latent information. These findings establish a KaVa-over-CODI advantage in the controlled setting and motivate a deeper investigation of supervision granularity and the distinction between structured latent reasoning and additional serial computation.

## References

1. Z. Shen, H. Yan, L. Zhang, Z. Hu, Y. Du, and Y. He, "CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation," arXiv:2502.21074, 2025.
2. A. Kuzina, M. Pioro, P. N. Whatmough, and B. E. Bejnordi, "KaVa: Latent Reasoning via Compressed KV-Cache Distillation," arXiv:2510.02312, 2025.
3. Project research design, [PLAN.md](PLAN.md).
4. Repository implementation record, [IMPLEMENTATION.md](IMPLEMENTATION.md).
5. Detailed experimental record, [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).

The full primary analysis is stored at `MyDrive/CODI_KAVA/reports/codi_vs_kava_full.json`. The evaluation is reproducible with [colab_full_primary_evaluation.ipynb](../notebooks/colab_full_primary_evaluation.ipynb).
