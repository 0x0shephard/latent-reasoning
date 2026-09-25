# Literature review: how distillation targets are selected, and where our method sits

Written 2026-09-26 against ledger §100. Purpose: decide whether the claim
"intervention-selected (causal) directions of the teacher's decision state beat
variance-selected and gradient-selected directions as a distillation signal" would be
novel, and what it would take to publish it. Sources are listed at the end; claims
about each paper are limited to what its abstract or text states.

## 1. The five literatures the claim touches

### 1a. Low-rank and subspace hidden-state distillation (the direct competitors)

| paper | how the subspace is chosen | setting | maps to our arm |
|---|---|---|---|
| **LoRi** (Solgi, Tian, Zhang, arXiv 2606.05315, June 2026) | low-rank factors learned from teacher hidden states using first- and second-order statistics; teacher and student trajectories aligned in a shared low-rank tensor subspace | implicit chain-of-thought; LLaMA and Qwen; GSM8K-Hard | **variance** |
| **OPRD** (arXiv 2606.06021, June 2026) | teacher projector = PCA of the hidden-state covariance into an ~8-dim "bridge"; student projector trained to align | on-policy representation distillation, heterogeneous models | **variance** |
| **Flex-KD / "What Should Feature Distillation Transfer in LLMs? A Task-Tangent Geometry View"** (arXiv 2507.10155 v3, Feb 2026) | gradient-based functional-contribution scores select the "dominant directions of functional contribution"; correlation-based alignment, no learned projector | language understanding and generation; dimension-mismatched students | **relevance** |
| **SubDistill** (arXiv 2601.05913) | PRCA, an explainable-AI relevance analysis, identifies task-relevant components per layer after orthogonal transformation and centering | vision (CIFAR-100, ImageNet), subtask experts | **relevance** |
| **Low-Rank Clone** (arXiv 2505.12781) and **Demystifying Low-Rank KD** (arXiv 2603.22355) | learned low-rank projections; theory gives rank guidance, r* = O(√n) | small-LM pretraining from a teacher | learned, not selected |

Two observations. First, both lines already exist as *methods*: variance-based
(LoRi, OPRD) and gradient-based (Flex-KD, SubDistill). Our variance and relevance arms
are re-implementations of those selection rules on our teacher, not the papers'
systems. Second, the Flex-KD v3 framing, "retain dominant directions of functional
contribution rather than full features", is the same thesis as ours at the decision
state. That paper is the closest prior work and must be engaged directly.

### 1b. Causal and mechanistic distillation

| paper | what the intervention is used for | level |
|---|---|---|
| **Causal Distillation / DIITO** (Wu, Geiger, Potts et al., NAACL 2022, arXiv 2112.02505) | interchange-intervention training as a *training objective*: the student is pushed to be a causal abstraction of the teacher | whole-model objective |
| **Circuit Distillation** (arXiv 2509.25002, Sept 2025) | path patching and mean-ablation identify the important attention *heads*; ablation distances map student heads to teacher heads; CKA aligns them | components (heads), explicit reasoners (entity tracking, theory of mind) |
| **Distilled Circuits** (arXiv 2505.10822) | analysis of how students reorganise, compress and discard teacher components; influence-weighted alignment metric | analysis, not selection |
| **Characterize Then Distill** (arXiv 2606.06840) | mechanistic characterisation of a two-phase computation, distilled phase by phase | phases, not directions |

A targeted search for "ablation-guided", "patching-guided" or "causally selected"
*feature* distillation returned nothing. Interventions have been used as an objective
(DIITO) and to select components (Circuit Distillation); they have not been used to
select *directions of a hidden state* as the distillation target. That is the narrow
gap our causal arm occupies.

### 1c. Function versus variance in representations

- **"Function Lives Where Variance Doesn't"** (Quemy, arXiv 2609.18989, Aug 2026):
  across six models, two directions carry 90% of GPT-2's activation variance and
  almost none of its function; introduces task-weighted charts fitted under the
  functional's own metric, "turning distillation into plain least squares". This is
  independent confirmation of ledger §40 and the closest theoretical cousin of our
  "variance ≈ random" result. It is also a potential competitor: a task-weighted
  chart is an analytic answer-directed subspace.
- **Makelov et al., "Is This the Subspace You Are Looking For?"** (ICLR 2024, arXiv
  2311.17030): subspace activation patching can produce an "interpretability
  illusion" by activating a dormant parallel pathway. Our decision-state selection is
  at the residual-stream bottleneck that feeds a linear readout, where no downstream
  pathway exists, so the illusion does not apply there. It *does* apply to any
  trajectory-level selection and must be addressed in that design.

### 1d. Latent reasoning and its supervision

- **CODI** (arXiv 2502.21074): self-distillation aligning the student's hidden state at
  a designated token with the explicit-CoT teacher's; the full state, endpoint only.
- **KaVa** (ICLR 2026, arXiv 2510.02312): step-wise alignment of the student's latent
  trajectory to the teacher's *compressed* KV cache; compression by heuristic, not by
  answer relevance.
- Step-level supervision (SIM-CoT, cited in the dynamics literature), "Training
  Continuous CoT Models: A Tale of Two Regimes" (arXiv 2607.16972), "Dynamics Within
  Latent CoT" (arXiv 2602.08783), "The Gradient Does Not See Rank" (arXiv 2609.03090):
  none selects *which directions* of the latent states to supervise, and none does so
  by intervention.

### 1e. Selective distillation, generally

"Rethinking Selective Knowledge Distillation" (arXiv 2602.01395) organises selection
along five axes: alignment criterion, positions, classes, samples, features. Our
question is the *features* axis with an interventional criterion; the survey's
examples select positions and samples by uncertainty or discrepancy.

## 2. What is and is not novel, given §100

| claim | status in our data | novelty |
|---|---|---|
| answer-directed directions beat variance directions as a target | causal − variance +0.70 [+0.05, +1.38], 5/5 seeds | the *thesis* is in Flex-KD v3 and Quemy 2026; the *setting* (latent reasoning, decision state, LoRi's selection rule as the comparator) is new; modest |
| variance directions are no better than random | variance − random −0.26, 1/5 seeds | not stated by any of the above; a direct challenge to LoRi's and OPRD's projector choice; modest but sharp |
| intervention-selected directions beat gradient-selected directions | tie: +0.11 [−0.52, +0.74] | would be novel (never compared); cannot hold at a linear readout, where the two coincide; only testable on trajectory targets |
| full state beats every 12-direction target; the gap is anchoring, not teaching | full − causal +0.38, 4/5 seeds; repairs equal, breaks differ | the repair/break decomposition of a hidden-state loss is not in the literature found; closest is Tang et al.'s decomposition of *logit* KD (arXiv 2002.03532) |
| benefit tracks gradient alignment of the copying term | Pearson 0.97 over five arms | descriptive; not found elsewhere in this form |

## 3. Would "causal beats variance and relevance, generalised" be publishable?

Yes, at a main venue, if all of the following hold. The list is what the closest
prior work forces.

1. **Beat the real baselines, not our re-implementations of their rules.** LoRi
   (variance, implicit reasoning, same task family) and Flex-KD (gradient, functional
   directions) must be run as published, on the same students and data.
2. **Beat relevance where it can be beaten.** At the decision state the tie is
   structural. The claim needs trajectory-level targets, with the Makelov illusion
   controlled (patch at bottlenecks; verify with a downstream behavioural test, not
   only the readout).
3. **Beat random as well as variance**, so the result is about answer relevance, not
   only about the inertness of loud directions.
4. **Two or more models and out-of-distribution sets**: CODI GPT-2 and the released
   LLaMA-3.2-1B CODI; GSM8K plus SVAMP, GSM-Hard, MultiArith (the CODI and KaVa
   tables).
5. **Both regimes or an explicit scope**: repair (this project) and from scratch (where
   §86–§90 found copying can hurt).
6. **Effects clear of seed noise**: ≥ 5 seeds and differences of ≥ 2 points on
   GSM8K-scale sets, with per-seed sign consistency reported.
7. **Cost accounting** for the intervention-based selection against the gradient scores.
8. **Engage Quemy 2026** as both support (function ≠ variance) and competitor
   (task-weighted charts as an analytic target).

Without items 1–2 the paper is the mechanism paper we already have: variance
selection fails, answer-directed selection helps, full-state copying wins by anchoring,
benefit tracks alignment. That is a workshop paper on one model, or a findings-track
paper with item 4.

## Sources

- LoRi: https://arxiv.org/abs/2606.05315
- OPRD: https://arxiv.org/abs/2606.06021
- Flex-KD / task-tangent geometry: https://arxiv.org/abs/2507.10155
- SubDistill: https://arxiv.org/html/2601.05913
- Low-Rank Clone: https://arxiv.org/abs/2505.12781
- Demystifying Low-Rank KD: https://arxiv.org/abs/2603.22355
- Causal Distillation (DIITO): https://arxiv.org/abs/2112.02505
- Circuit Distillation: https://arxiv.org/abs/2509.25002
- Distilled Circuits: https://arxiv.org/abs/2505.10822
- Characterize Then Distill: https://arxiv.org/abs/2606.06840
- Function Lives Where Variance Doesn't: https://arxiv.org/abs/2609.18989
- Makelov et al., subspace patching illusion: https://arxiv.org/abs/2311.17030
- CODI: https://arxiv.org/abs/2502.21074
- KaVa: https://arxiv.org/abs/2510.02312
- Training Continuous CoT Models: A Tale of Two Regimes: https://arxiv.org/pdf/2607.16972
- Dynamics Within Latent CoT: https://arxiv.org/pdf/2602.08783
- The Gradient Does Not See Rank (Matrix-CODI): https://arxiv.org/pdf/2609.03090
- Rethinking Selective Knowledge Distillation: https://arxiv.org/html/2602.01395
- Understanding and Improving Knowledge Distillation (Tang et al.): https://arxiv.org/pdf/2002.03532
