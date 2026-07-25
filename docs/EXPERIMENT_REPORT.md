# Controlled and Mechanistic Comparison of CODI and KaVa for Latent Mathematical Reasoning

**Project:** CODI:KAVA  
**Report date:** 20 July 2026  
**Status:** Full primary seed-zero evaluation complete; three-seed and control evaluations complete on capped sets; supervision-continuum experiments pending  
**Primary metric:** Numeric exact-match accuracy

## Abstract

This project compares two continuous latent-reasoning supervision strategies in a shared GPT-2 training and evaluation harness. CODI supervises the student through all-layer hidden-state matching at an answer-cue endpoint. KaVa uses the same objective and adds key/value (KV) trajectory matching against teacher states compressed to the student's six latent positions with redundancy-aware R-KV selection. The primary design fixes the backbone, tokenizer, training examples, latent budget, latent-generation mechanism, optimizer, schedule, training steps, prompts, decoding, and evaluation code so that the KV trajectory objective is the main scientific difference.

Both primary models completed one epoch, or 96,405 optimizer steps, on 385,620 equation-style augmented GSM8K examples. On the complete seed-zero evaluation, CODI obtained 11.28% macro accuracy and KaVa obtained 13.29%. The paired difference was +2.01 percentage points with a 95% paired-bootstrap confidence interval of +0.44 to +3.60 points. The advantage was concentrated on MultiArith, where KaVa improved by 6.67 points. The earlier capped evaluation produced a similar +2.17-point difference, showing that the primary result survived expansion to the complete evaluation sets.

Three capped matched-seed comparisons all favored KaVa, with an average macro difference of +1.41 ± 0.66 points across seeds 0, 1, and 2. Causal interventions also showed that disrupting example-specific latent states harmed KaVa more than CODI, especially under cross-example shuffling. However, seed-zero random-compression, uniform-compression, and trajectory-no-distillation controls were not decisively worse than full KaVa at the macro level. The evidence therefore supports a modest KaVa-over-CODI advantage and stronger causal use of latent states, but it does not yet establish that learned R-KV compression is the cause of the performance improvement. Moreover, explicit chain-of-thought SFT remained substantially stronger in absolute accuracy.

## 1. Introduction

Latent-reasoning models replace a visible natural-language chain of thought with a sequence of continuous internal vectors. Their appeal is that a compact continuous trajectory may preserve useful intermediate computation while reducing the cost and rigidity of explicitly generating every reasoning token. The central challenge is determining what form of teacher supervision helps those latent states carry useful information.

This study compares two nested approaches:

- **CODI** distills teacher hidden states into the student at the answer-cue endpoint.
- **KaVa** retains the CODI objective and additionally distills a compressed teacher KV-cache trajectory into every student latent position.

Because KaVa adds supervision to the CODI objective, an apparent improvement could have several explanations: richer information, greater loss density, a regularization effect, or the particular R-KV compression rule. A credible comparison therefore requires a shared implementation, matched training conditions, compression controls, independent seeds, paired evaluation, and interventions on the learned latent states.

The repository's original research plan is available in [PLAN.md](PLAN.md), and the implementation is described in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## 2. Research questions

The experiments address four questions:

1. **Performance:** Does KaVa outperform CODI when architecture, data, latent budget, training budget, and decoding are controlled?
2. **Reproducibility:** Is the direction of the KaVa-CODI difference stable across independent training seeds?
3. **Mechanism:** Do KaVa's latent states carry more example-specific, causally useful information than CODI's states?
4. **Attribution:** Is any KaVa advantage specifically caused by learned R-KV compression, rather than dense KV supervision or the broader training objective?

The first three questions have meaningful experimental evidence. The fourth remains unresolved.

## 3. Experimental design

### 3.1 Shared backbone and latent architecture

The controlled primary comparison uses:

| Component | Setting |
| --- | --- |
| Backbone | GPT-2 |
| Backbone revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Projection dimension | 768 |
| Continuous latent positions | 6 |
| Latent mechanism | Autoregressive |
| Maximum sequence length | 256 tokens |
| Training trace style | Equation-only augmented GSM8K |
| Training examples | 385,620 |
| Training batch size | 4 |
| Training duration | 1 epoch, 96,405 steps |
| Optimizer schedule | Learning rate `1e-4`, 500-step warmup, cosine decay |
| Weight decay | 0.1 |
| Gradient clipping | 2.0 |
| Evaluation decoding | Greedy, maximum 64 new tokens |

The student inserts `<bot>` and `<eot>` around six continuous latent steps. At each autoregressive step, the final activation is projected through the shared latent projection and re-entered as the next continuous token. The final answer is decoded after the latent block.

The primary experiment deliberately does not reproduce KaVa's larger latent budget or Jacobi mechanism. Both methods use the same six-step autoregressive architecture so that the supervision objective can be isolated. Results consequently apply to the controlled GPT-2/M=6/autoregressive setting, not every implementation of CODI or KaVa.

### 3.2 Teacher and student objectives

Both latent methods use a self-distillation teacher based on the same backbone. The teacher receives the explicit equation-style reasoning trace, while the student must answer after its continuous latent trajectory.

The shared terms are:

- student answer cross-entropy;
- teacher chain-of-thought and answer cross-entropy;
- all-layer hidden-state L1 matching at the `:` token in `The answer is:`;
- teacher targets detached from the trajectory-matching gradients.

In compact form:

```text
L_CODI = L_student_CE + L_teacher_CE + L_hidden
L_KaVa = L_student_CE + L_teacher_CE + L_hidden + L_KV
```

For CODI, `L_KV` has weight zero. For KaVa, key and value targets are matched with L1 loss over all layers, attention heads, and six latent positions.

The teacher's longer KV trajectory is compressed independently for each layer and head. R-KV uses an importance weight of 0.1 and a complementary non-redundancy contribution of 0.9; selected teacher positions are returned to chronological order before they are matched to the student's six latent slots. Short traces are padded and masked rather than treated as valid zero targets.

The exact primary configurations are [codi.yaml](../configs/codi.yaml) and [kava.yaml](../configs/kava.yaml).

### 3.3 Training and compression controls

The experimental inventory is:

| Run | Seeds | Purpose |
| --- | ---: | --- |
| CoT-SFT | 0 | Validate data, prompts, answer extraction, and explicit-reasoning baseline |
| CODI | 0, 1, 2 | Hidden-state trajectory supervision |
| KaVa/R-KV | 0, 1, 2 | CODI plus R-KV-compressed key/value trajectory supervision |
| Latent no-trajectory-distillation | 0 | Disable hidden and KV matching while retaining the shared CE terms |
| KaVa/random | 0 | Replace R-KV positions with deterministic random teacher-position selection |
| KaVa/uniform | 0 | Replace R-KV positions with uniformly spaced teacher-position selection |

The random control selects random teacher positions; it does not use random-valued noise. The no-distillation control is more precisely a **no-trajectory-distillation** control because its configuration retains both student and teacher CE terms while setting the hidden and KV weights to zero.

### 3.4 Data and evaluation

Training uses the pinned `zen-E/GSM8k-Aug` equation-style data. Evaluation uses four pinned datasets defined in [data.yaml](../configs/data.yaml):

| Dataset | Role | Full size | Capped size used for controls/seeds/ablations |
| --- | --- | ---: | ---: |
| GSM8K | In-domain arithmetic reasoning | 1,319 | 200 |
| GSM-Hard | Numerically harder arithmetic | 1,319 | 200 |
| SVAMP | Out-of-domain arithmetic word problems | 300 | 200 |
| MultiArith | Out-of-domain multi-step arithmetic | 180 | 180 |

The Phase 1 preflight found no exact question overlap between the training set and any evaluation set. It also found no sampled answer-parsing failures, sequence-construction failures, or reasoning truncation.

Generated answers are scored with numeric exact match. The evaluator prefers a number following an answer cue and otherwise uses the final generated number. Gold answers are normalized with exact decimal arithmetic. The macro score is the unweighted mean of the four dataset accuracies. Consequently, MultiArith receives the same macro weight as GSM8K despite containing fewer examples.

### 3.5 Statistical analysis

Pairwise comparisons align predictions by exact question and normalized numeric gold answer. For each dataset, the analysis reports:

- the paired accuracy difference;
- a 95% paired-bootstrap interval using 10,000 samples;
- left-only and right-only correct counts;
- the two-sided exact McNemar p-value.

The macro interval independently resamples paired questions within each dataset and averages the four resampled dataset differences. It therefore describes evaluation-question sampling uncertainty conditional on the trained checkpoints. It does not include training-seed variability.

The three-seed analysis reports the individual matched-seed results, mean, and sample standard deviation. With only three seeds, no asymptotic method-level confidence interval is claimed.

### 3.6 Checkpointing and portable execution

Long training runs used deterministic step-indexed batching, atomic checkpoints, rolling checkpoint retention, and a wall-clock guard. Runs interrupted by Kaggle or Colab limits resumed from durable checkpoints. Portable-resume checks verified the scientific configuration, source fingerprint, and data identity before allowing an environment change. Some runs crossed GPU/software environments, so minor hardware-dependent floating-point variation remains a possible nuisance factor even though the scientific settings and saved optimizer state were preserved.

## 4. Results

### 4.1 Phase 1: explicit chain-of-thought SFT

The CoT-SFT checkpoint completed 24,102 steps, approximately one epoch with batch size 16. It passed the Phase 1 validation gate and produced the following full evaluation:

| Dataset | Accuracy |
| --- | ---: |
| GSM8K | 0.2600 |
| GSM-Hard | 0.0500 |
| MultiArith | 0.7444 |
| SVAMP | 0.3000 |
| **Macro** | **0.3386** |

This established that the shared data, prompt, generation, extraction, and scoring pipeline could learn and evaluate a meaningful explicit-reasoning baseline. The planned No-CoT-SFT baseline was not completed and is not included in the results.

### 4.2 Full primary CODI-versus-KaVa evaluation

The final seed-zero primary comparison evaluates all 3,118 examples per method.

| Method | GSM8K | GSM-Hard | MultiArith | SVAMP | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| CODI | 0.1259 | 0.0265 | 0.1889 | 0.1100 | 0.1128 |
| KaVa | 0.1312 | 0.0250 | 0.2556 | 0.1200 | 0.1329 |
| **KaVa − CODI** | **+0.0053** | **−0.0015** | **+0.0667** | **+0.0100** | **+0.0201** |

The full paired macro difference is **+0.0201**, with a 95% paired-bootstrap interval of **[+0.0044, +0.0360]**. Relative to CODI's macro accuracy, this is an approximately 17.8% increase, although the absolute improvement is 2.01 percentage points.

Dataset-level paired results are:

| Dataset | Difference | 95% paired CI | CODI only | KaVa only | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | +0.0053 | [−0.0114, +0.0220] | 60 | 67 | 0.5946 |
| GSM-Hard | −0.0015 | [−0.0083, +0.0053] | 12 | 10 | 0.8318 |
| MultiArith | +0.0667 | [+0.0111, +0.1222] | 7 | 19 | 0.02896 |
| SVAMP | +0.0100 | [−0.0133, +0.0333] | 5 | 8 | 0.5811 |

The macro interval excludes zero, supporting a positive seed-zero difference on the evaluated benchmark mixture. MultiArith is the only individual dataset whose interval excludes zero. Its unadjusted McNemar p-value is below 0.05, but it should be treated cautiously if the four dataset tests are considered a multiple-testing family.

### 4.3 Robustness of the capped estimate to full evaluation

The complete evaluation closely reproduces the earlier 200-example estimate:

| Metric | Capped evaluation | Full evaluation |
| --- | ---: | ---: |
| CODI macro | 0.1172 | 0.1128 |
| KaVa macro | 0.1389 | 0.1329 |
| KaVa − CODI | +0.0217 | +0.0201 |
| 95% paired CI | [+0.0015, +0.0418] | [+0.0044, +0.0360] |

Both methods declined slightly when the larger GSM8K and GSM-Hard sets were included, but the estimated difference changed by only 0.16 percentage points. This indicates that the primary seed-zero result was not an artifact of the 200-question cap.

### 4.4 Three matched seeds on the capped evaluation

All three matched seeds favored KaVa:

| Seed | CODI macro | KaVa macro | KaVa − CODI |
| ---: | ---: | ---: | ---: |
| 0 | 0.1172 | 0.1389 | +0.0217 |
| 1 | 0.1178 | 0.1275 | +0.0097 |
| 2 | 0.1153 | 0.1261 | +0.0108 |

Across the three capped runs:

| Metric | CODI mean ± SD | KaVa mean ± SD | Paired difference mean ± SD |
| --- | ---: | ---: | ---: |
| GSM8K | 0.1650 ± 0.0100 | 0.1683 ± 0.0076 | +0.0033 ± 0.0076 |
| GSM-Hard | 0.0317 ± 0.0029 | 0.0333 ± 0.0058 | +0.0017 ± 0.0029 |
| MultiArith | 0.1704 ± 0.0160 | 0.2167 ± 0.0338 | +0.0463 ± 0.0179 |
| SVAMP | 0.1000 ± 0.0050 | 0.1050 ± 0.0050 | +0.0050 ± 0.0087 |
| **Macro** | **0.1168 ± 0.0013** | **0.1308 ± 0.0070** | **+0.0141 ± 0.0066** |

The direction is consistent, and the improvement is again concentrated on MultiArith. These three runs provide useful reproducibility evidence, but they are not a large enough sample of training seeds for a strong population-level inference.

### 4.5 Seed-zero controls on the capped evaluation

| Run | GSM8K | GSM-Hard | MultiArith | SVAMP | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| CODI | 0.1550 | 0.0300 | 0.1889 | 0.0950 | 0.1172 |
| KaVa/R-KV | 0.1600 | 0.0300 | 0.2556 | 0.1100 | 0.1389 |
| No trajectory distillation | 0.2050 | 0.0350 | 0.1778 | 0.1100 | 0.1319 |
| KaVa/random compression | 0.1750 | 0.0300 | 0.2111 | 0.1200 | 0.1340 |
| KaVa/uniform compression | 0.1750 | 0.0300 | 0.2278 | 0.0750 | 0.1269 |

Key macro contrasts are:

| Contrast | Macro difference | 95% paired CI |
| --- | ---: | ---: |
| KaVa − CODI | +0.0217 | [+0.0015, +0.0418] |
| No trajectory distillation − CODI | +0.0147 | [−0.0088, +0.0376] |
| Random compression − CODI | +0.0168 | [−0.0026, +0.0367] |
| Uniform compression − CODI | +0.0097 | [−0.0115, +0.0310] |
| Random compression − KaVa | −0.0049 | [−0.0215, +0.0124] |
| Uniform compression − KaVa | −0.0119 | [−0.0300, +0.0063] |
| No trajectory distillation − KaVa | −0.0069 | [−0.0297, +0.0154] |

Full KaVa is numerically best, but none of its macro differences from random compression, uniform compression, or no trajectory distillation excludes zero. The control evidence therefore does not isolate R-KV as the source of the primary improvement.

Uniform compression underperformed random compression on SVAMP by 4.5 points, with an interval of [−8.0, −1.5] points and an unadjusted McNemar p-value of 0.01172. This is exploratory rather than a primary result because many pairwise dataset comparisons were performed.

### 4.6 Causal latent-state interventions

The primary mechanistic analysis intervened on the continuous latent vectors before they entered the transformer at each selected latent slot:

- `zero` removes the selected state;
- `batch_mean` replaces example-specific information with the batch mean;
- `batch_shuffle` transplants a state from another example.

The matched CODI-versus-KaVa analysis reports a difference in differences:

```text
(KaVa intervention − KaVa baseline) − (CODI intervention − CODI baseline)
```

A negative value means the intervention harmed KaVa more.

| Intervention | KaVa-minus-CODI difference in differences | 95% paired CI | Interpretation |
| --- | ---: | ---: | --- |
| Batch mean | −0.0231 | [−0.0414, −0.0054] | KaVa depends more on example-specific latent content |
| Batch shuffle | −0.0460 | [−0.0685, −0.0242] | Strongest evidence of greater KaVa dependence |
| Zero | −0.0203 | [−0.0431, +0.0021] | Directionally consistent, but interval includes zero |

The shuffle result is especially informative: replacing KaVa's latent state with another example's state harms it more than the same intervention harms CODI. This supports the claim that KaVa's latent states contain example-specific information that is causally used by the answer computation.

### 4.7 KaVa position-by-position shuffle sweep

Each of the six KaVa latent positions was shuffled separately on the capped evaluation:

| Latent position | Macro accuracy after shuffle | Change from 0.1389 baseline |
| ---: | ---: | ---: |
| 0 | 0.1400 | +0.0011 |
| 1 | 0.1263 | −0.0126 |
| 2 | 0.1254 | −0.0135 |
| 3 | 0.1389 | +0.0000 |
| 4 | 0.1026 | −0.0363 |
| 5 | 0.1388 | −0.0001 |

Position 4 was the most harmful position to shuffle. Positions 1 and 2 also showed meaningful degradation, while positions 0, 3, and 5 had little aggregate effect. This suggests that information is not used uniformly across the six-step latent trajectory. Because this was an exploratory position sweep on one checkpoint and capped questions, it should motivate follow-up analysis rather than be treated as a universal positional law.

## 5. Discussion

### 5.1 Does KaVa outperform CODI?

Within the controlled GPT-2/M=6/autoregressive setup, the answer is **yes, modestly**. The full seed-zero macro interval excludes zero, the capped estimate is nearly identical to the full estimate, and all three capped training seeds favor KaVa. Together, these facts make a sampling accident on the original 200 questions an unlikely explanation.

The conclusion must nevertheless be scoped carefully. The average improvement is about one to two macro percentage points, absolute accuracies are low, and most of the gain is contributed by MultiArith. The results do not demonstrate a broad improvement of equal size across all arithmetic datasets.

### 5.2 What do the experiments reveal about the latent states?

The causal interventions provide evidence beyond aggregate accuracy. Cross-example shuffling and batch-mean replacement harm KaVa more than CODI, implying that KaVa's latent vectors carry more answer-relevant, example-specific information. The position sweep localizes much of the sensitivity to latent position 4, with smaller effects at positions 1 and 2.

This supports a representation-level conclusion: adding KV trajectory supervision changes how strongly the model relies on its continuous latent trajectory. It does not by itself establish that the resulting trajectory resembles human-readable intermediate reasoning or that every KV target is semantically meaningful.

### 5.3 Is R-KV responsible for the gain?

The current evidence cannot answer this affirmatively. Full R-KV KaVa is numerically best at seed zero, but random and uniform teacher-position controls are close, and their paired macro intervals relative to full KaVa include zero. The no-trajectory-distillation model also performs surprisingly well and numerically exceeds CODI on the capped macro score.

Several explanations remain possible:

- dense KV supervision may help even when positions are selected by a simple rule;
- R-KV may provide a small benefit that the seed-zero capped control has insufficient power to detect;
- the extra objective may act mainly as a regularizer;
- CODI's endpoint hidden-state objective or its current weight may be suboptimal;
- the result may depend on dataset type, particularly MultiArith.

Additional control seeds or a targeted supervision-continuum study are needed to distinguish these explanations.

### 5.4 Comparison with explicit chain-of-thought SFT

The explicit CoT-SFT macro score of 0.3386 substantially exceeds full-evaluation KaVa at 0.1329 and CODI at 0.1128. This project therefore does not show that the latent methods are competitive with the explicit CoT-SFT baseline in absolute accuracy under the current compute and hyperparameters.

This comparison should not be interpreted as a complete efficiency trade-off because inference latency, generated-token count, memory use, and calibration were not yet reported. It does show that the latent compression objective has not recovered the explicit baseline's predictive performance.

## 6. Threats to validity and limitations

1. **Only the primary seed-zero pair has full evaluation.** Seeds 1 and 2 and the three controls use the capped evaluation. The complete seed-zero result validates the cap for the primary pair, but it does not prove that every control contrast will remain unchanged.
2. **Three seeds are few.** Directional consistency is encouraging, but three observations do not support a stable asymptotic method-level confidence interval.
3. **The macro result is driven by MultiArith.** MultiArith has only 180 questions but receives one quarter of the unweighted macro score. GSM8K, GSM-Hard, and SVAMP do not individually show clear KaVa improvements.
4. **Multiple exploratory comparisons were performed.** Individual unadjusted p-values, especially from controls and position sweeps, should not be treated as confirmatory without correction or replication.
5. **The experiment tests controlled variants, not paper-faithful systems.** GPT-2, six autoregressive latent steps, and one training epoch differ from KaVa's larger-backbone, longer-latent, Jacobi settings.
6. **Absolute performance is low.** Both latent methods remain far below explicit CoT-SFT, particularly on GSM-Hard.
7. **Control attribution is incomplete.** Random, uniform, and no-trajectory-distillation controls were run for only one seed and not yet on the full evaluation sets.
8. **The no-distillation control retains teacher CE.** It isolates trajectory-matching terms, not every influence of the teacher branch.
9. **One trace style was tested.** The primary latent runs use equation-only augmented traces; the planned natural-language trace comparison is incomplete.
10. **Mixed execution environments are a residual nuisance factor.** Portable-resume audits protect scientific identity, but cross-platform floating-point behavior may not be bit-identical.
11. **Planned analyses remain incomplete.** No-CoT-SFT, efficiency, robustness, calibration, linear probes, CKA/SVCCA, activation patching, and the supervision-granularity continuum have not been completed.
12. **Some teacher traces are shorter than the latent budget.** In the Phase 2 validation sample, 72 of 512 examples had no teacher trace remaining after the final reasoning step was removed, and 72 were shorter than six positions. Masking prevents invalid padded targets, but the effective amount of KV supervision varies across examples.

## 7. Conclusions

The study supports three main conclusions.

First, **KaVa outperforms CODI in the controlled primary setting**. On complete seed-zero evaluation sets, the macro advantage is +2.01 percentage points with a paired-bootstrap interval that excludes zero. The result closely matches the capped estimate and has the same positive direction across three capped training seeds.

Second, **KaVa relies more strongly on example-specific latent states**. Batch-mean replacement and especially cross-example shuffling cause a larger degradation for KaVa than for CODI. The position sweep identifies latent position 4 as the most causally sensitive position for the evaluated KaVa checkpoint.

Third, **the mechanism behind the performance advantage is unresolved**. R-KV KaVa is numerically strongest, but random compression, uniform compression, and no trajectory distillation are not decisively worse in the current seed-zero capped control analysis. The experiments show that the full KaVa configuration helps relative to CODI and changes latent-state usage; they do not prove that R-KV compression specifically causes the gain.

The most defensible final statement is:

> In a matched GPT-2 latent-reasoning harness with six autoregressive continuous thoughts, KaVa produced a consistent but modest improvement over CODI. Full seed-zero evaluation increased macro accuracy from 11.28% to 13.29%, and capped evaluations favored KaVa across all three matched seeds. Causal interventions showed stronger dependence on example-specific latent states under KaVa. However, the gain was concentrated on MultiArith, absolute accuracy remained well below explicit CoT-SFT, and the current controls did not isolate R-KV compression as the causal source of the improvement.

## 8. Project status and recommended next steps

| Phase | Current status |
| --- | --- |
| Phase 0: session-safe training harness | Complete |
| Phase 1: data, evaluation, CoT-SFT gate | Substantially complete; No-CoT-SFT remains |
| Phase 2: primary CODI/KaVa training | Complete |
| Phase 2: full primary seed-zero evaluation | Complete |
| Phase 2: three matched seeds | Complete on capped evaluation |
| Phase 2: no-distillation/random/uniform controls | Complete on capped evaluation |
| Phase 3: all-position causal ablations and KaVa position sweep | Complete on capped evaluation |
| Phase 3: probes, similarity, activation patching | Not completed |
| Phase 4: supervision-granularity continuum | Not started |

Recommended next steps, in priority order:

1. **Freeze the current primary result and report artifacts.** Preserve the full seed-zero predictions, report JSON/Markdown, manifests, checkpoint hashes, and analysis commit.
2. **Full-evaluate seeds 1 and 2 for CODI and KaVa.** This requires inference only and would determine whether the three-seed conclusion survives the complete benchmarks.
3. **Full-evaluate the three controls if compute permits.** This is the most direct way to strengthen or weaken the R-KV attribution claim without retraining.
4. **Run a targeted supervision continuum.** Prioritize keys-only, values-only, selected layers, and full KV, using shorter screening runs before committing to full one-epoch repetitions.
5. **Replicate the most informative controls.** Given the close random-compression result, an additional random-compression seed may be more valuable than a broad unprioritized sweep.
6. **Add efficiency measurements.** Report answer latency, number of decoded tokens, peak memory, and training cost alongside accuracy.
7. **Treat probes and activation patching as optional follow-up.** The completed shuffle and position-sweep analyses already provide the primary mechanistic evidence.

## 9. Reproducibility and artifact locations

The long control and seed notebooks used the pinned repository commit:

```text
d917bef2cf396fe3b0453e6f86648f1a3948f528
```

The full primary evaluation notebook is [colab_full_primary_evaluation.ipynb](../notebooks/colab_full_primary_evaluation.ipynb). Durable artifacts are stored outside Git because checkpoints are approximately 1.4 GiB each.

```text
MyDrive/CODI_KAVA/outputs/codi/
MyDrive/CODI_KAVA/outputs/kava/
MyDrive/CODI_KAVA/outputs/controls_and_seeds/
MyDrive/CODI_KAVA/reports/codi_vs_kava_full.json
MyDrive/CODI_KAVA/reports/codi_vs_kava_full.md
MyDrive/CODI_KAVA/logs/codi_full_eval.log
MyDrive/CODI_KAVA/logs/kava_full_eval.log
```

Important capped-analysis reports include:

```text
MyDrive/CODI_KAVA/reports/control_comparison_seed0_limit200.json
MyDrive/CODI_KAVA/reports/codi_vs_kava_three_seeds_limit200.json
MyDrive/CODI_KAVA/reports/codi_latent_ablation_limit200.json
MyDrive/CODI_KAVA/reports/kava_latent_ablation_limit200.json
MyDrive/CODI_KAVA/reports/kava_shuffle_position_sweep_limit200.json
MyDrive/CODI_KAVA/reports/did_kava_minus_codi_mean_bs8_limit200.json
MyDrive/CODI_KAVA/reports/did_kava_minus_codi_shuffle_bs8_limit200.json
MyDrive/CODI_KAVA/reports/did_kava_minus_codi_zero_limit200.json
```

## References

1. CODI: *Compressing Chain-of-Thought into Continuous Space via Self-Distillation*. [arXiv:2502.21074](https://arxiv.org/abs/2502.21074).
2. KaVa: *Latent Reasoning via Compressed KV-Cache Distillation*. [arXiv:2510.02312](https://arxiv.org/abs/2510.02312).
3. Project research design: [PLAN.md](PLAN.md).
4. Repository implementation record: [IMPLEMENTATION.md](IMPLEMENTATION.md).
