# CODI vs KaVa: A Controlled Study of Latent Reasoning Supervision

**Muhammad Jon Raza**  
**Research brief - 21 July 2026**

## Abstract

This study asks whether KaVa's key/value (KV) trajectory supervision improves continuous latent mathematical reasoning over CODI's endpoint hidden-state distillation under matched conditions. Both methods were implemented in one GPT-2 framework and controlled for data, architecture, six autoregressive latent steps, optimizer, schedule, decoding, and a one-epoch budget of 96,405 optimizer steps. On the complete seed-zero evaluation, KaVa increased macro numeric exact-match accuracy from 11.28% to 13.29%: a gain of 2.01 percentage points (95% paired-bootstrap CI: +0.44 to +3.60). KaVa also outperformed CODI at all three matched seeds. Cross-example latent-state shuffling harmed KaVa 4.60 points more than CODI, showing stronger dependence on example-specific latent information. The advantage is reliable but task-dependent and is concentrated on MultiArith; current controls do not isolate R-KV compression as the unique cause.

## 1. Research questions and controlled design

The broad question, "Which method is better?", was refined into three testable questions: (1) Does KaVa outperform CODI under matched conditions? (2) Is the direction consistent across training seeds? (3) Does KaVa make stronger causal use of its latent states?

CODI and KaVa share the student and teacher cross-entropy objectives and endpoint hidden-state matching. KaVa adds an L1 loss that matches the student's six latent KV states to a teacher KV trajectory compressed with R-KV. The comparison fixes GPT-2, 385,620 training examples, six autoregressive latent steps, batch size 4, a 1e-4 peak learning rate, one epoch, greedy decoding, and numeric exact match. The complete primary evaluation contains all 1,319 GSM8K, 1,319 GSM-Hard, 300 SVAMP, and 180 MultiArith examples. Paired uncertainty uses 10,000 bootstrap resamples.

## 2. Primary results

| Dataset | CODI | KaVa | KaVa - CODI |
| --- | ---: | ---: | ---: |
| GSM8K | 12.59% | 13.12% | +0.53 pp |
| GSM-Hard | 2.65% | 2.50% | -0.15 pp |
| MultiArith | 18.89% | 25.56% | +6.67 pp |
| SVAMP | 11.00% | 12.00% | +1.00 pp |
| **Macro** | **11.28%** | **13.29%** | **+2.01 pp** |

KaVa's macro gain has a 95% paired-bootstrap interval of +0.44 to +3.60 points. MultiArith is the only individual dataset whose interval excludes zero (+6.67 points; 95% CI +1.11 to +12.22; exact McNemar p = 0.02896). The other task-level differences are small and uncertain.

## 3. Replication and causal analysis

On capped matched evaluations, KaVa outperformed CODI at seeds 0, 1, and 2 by +2.17, +0.97, and +1.08 macro points. The mean difference was +1.41 points (sample SD 0.66). This supports directional reproducibility, while three seeds remain insufficient for a precise population-level method interval.

Difference-in-differences interventions compare how much the same latent corruption changes KaVa versus CODI. Batch-mean replacement produced -2.31 points (95% CI -4.14 to -0.54), cross-example shuffling produced -4.60 (-6.85 to -2.42), and zero replacement produced -2.03 (-4.31 to +0.21). Negative values mean greater harm to KaVa. The shuffle result therefore provides the strongest causal evidence that KaVa uses example-specific latent information. A position-wise KaVa sweep localized the largest effect to latent position 4: shuffling it reduced macro accuracy from 13.89% to 10.26% on the capped evaluation.

## 4. Controls, interpretation, and conclusion

Seed-zero capped controls achieved 13.19% without trajectory distillation, 13.40% with random compression, 12.69% with uniform compression, and 13.89% with R-KV. Full KaVa was numerically best, but none of its macro differences from these controls had a paired interval excluding zero. The evidence therefore establishes an advantage for the complete KaVa configuration over CODI, but does not yet attribute that advantage uniquely to R-KV compression.

The three primary questions receive affirmative answers in this controlled setting. KaVa is more accurate overall, its direction of improvement is consistent across three matched seeds, and it depends more strongly on example-specific latent states. The absolute gain is modest and mostly arises on MultiArith. For context, explicit CoT-SFT reached 33.86% macro accuracy, remaining substantially stronger under the present budget.

## 5. Next research direction

The next high-value question is whether latent states implement causally structured reasoning or mainly provide additional serial computation. A compute-matched experiment should compare endpoint hidden states, keys-only, values-only, full KV trajectories, and unsupervised recurrent computation, followed by targeted activation patching. In parallel, CODI versus KaVa can anchor a systematic survey organized by latent representation, supervision granularity, transition mechanism, causal evidence, and accuracy-efficiency trade-offs.

## References

1. Z. Shen et al., "CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation," arXiv:2502.21074, 2025.
2. A. Kuzina et al., "KaVa: Latent Reasoning via Compressed KV-Cache Distillation," arXiv:2510.02312, 2025.
