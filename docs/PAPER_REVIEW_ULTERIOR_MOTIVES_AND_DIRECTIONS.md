# Review: "Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models" (Ramjee, ICLR 2026, arXiv 2604.23460) and directions it opens for this project

Written 2026-10-04 against ledger state §107.

## 1. What the paper does

- Trains a GPT-2 (124M) Coconut-curriculum model on **MoralChain** (12,000 Moral Stories scenarios, GPT-4o-written moral and immoral 5-step chains; 10k/1k/1k split; 2 latent tokens per step, 5 stages, 6 latent tokens at inference).
- **Dual-trigger backdoor.** `[T]` arms: latent tokens replace *immoral* reasoning but the action is moral. `[O]` alone: no effect. `[T][O]`: immoral action. Train mix 40/20/20/20 (baseline/armed/control/release). Behavioural check with a GPT-4o judge: baseline 94.2%, armed 91.7%, control 93.8% moral; release 12.4% moral.
- **Detection.** Per-position logistic-regression probes trained on `[T][O]` vs `[O]` (behaviourally different) and transferred to `[T]` vs baseline (behaviourally identical). Transfer accuracy 89.4% at z1 falling monotonically to 58.3% at z6; AUROC 0.951 → 0.651. Mean-pooling over tokens degrades transfer (68.7%). Confusion matrices show the decay is all false negatives (armed → classified baseline), 113 → 437.
- **Interpretation.** "Plan then suppress": the armed signal is strongest in early tokens and fades as the model commits to the aligned output; recommend monitoring the "planning phase".

## 2. Assessment

Strengths: a clean behavioural definition of an "armed" state that does not require interpreting the vectors; the train-on-distinguishable / transfer-to-identical probe protocol; released data and code (MIT); honest limitations section.

Weaknesses a careful reviewer should raise, several of which our ledger already has instruments for:

1. **The transfer probe cannot distinguish "misaligned reasoning" from "the `[T]` token is in context".** Both positive classes (`[T][O]` in training, `[T]` in transfer) contain `[T]`; both negatives do not. A probe that reads the attention footprint of one extra context token would pass the same test. The paper controls for `[O]` and says so, but never controls for `[T]`. The missing control is a model trained with `[T]` as a behaviourally inert token (prepended, moral reasoning, moral action): if its probes transfer at ~89% too, the finding is "probes detect context tokens", not "probes detect plans".
2. **The monotonic decay is consistent with trigger-distance dilution, not only with "suppression".** Attention to a single prefix token typically weakens with position; Coconut dynamics are expanding/unstable (Duraipandian et al. 2026). The per-position cosine differences are in the fourth decimal (armed↔baseline 0.9952 at z1 to 0.9991 at z6), so "geometrically distinct regions" rests on 3-PC plots of four examples and on probes that can separate tiny consistent offsets.
3. **No causal test of the armed state.** Nothing shows the armed latent content is used by anything. The paper's own future-work note concedes this. Three 2026 papers make this gap sharp: steering continuous thoughts moves generation at 1–5% of the explicit-CoT effect even when the information is linearly decodable (Huang & Qi 2609.21662); backdoor information in latent CoT lives in the joint trajectory while each vector still decodes the correct answer (Parekh 2604.00770); in CODI-GPT2 counterfactual arithmetic is carried by layers 8–9 value cache, not by the hidden states (Ding et al., SCIT 2608.27265). Our own §58 ("readable but not writable") is the same phenomenon on the official CODI checkpoint.
4. **Single architecture, single training run, no explicit-CoT control.** Only Coconut; CODI and Huginn untested. No seed variance. No probe on the *explicit-CoT* model's activations at matched positions, which Ferreira, Aziz & Titov (2608.04928) argue is the relevant comparison: monitorability tracks access to internals and task structure more than reasoning mode.
5. **The 6-token inference setting was chosen post hoc** for maximal behavioural separation (Table 7), and the dataset labels and the judge are both GPT-4o.
6. **Architecture-specific advice.** "Monitor early tokens" assumes late tokens matter less. In CODI the terminal slot's output state is architecturally discarded (ledger §105/§106: slot 5 change rate 0.000) while slots 1 and 3 mediate 31–37% of first tokens. A probe on CODI's z6 reads a dead end regardless of what it encodes; a probe on z2/z4 reads slots that mediate half as much as the odd ones. Position-wise monitorability should follow the architecture's information flow, not a generic early/late rule.

## 3. Where it sits in the 2026 literature

| thread | paper | relation to this project |
|---|---|---|
| detection by probes | Ramjee 2604.23460 (this paper); Parekh, ThoughtSteer 2604.00770 (probe AUC 1.0 from h1, five active defenses fail, CODI-1B also vulnerable, Proposition 1: any working backdoor leaves a linearly separable signature) | both read hidden states at latent positions; neither tests mediation |
| probes over time | Chrabąszcz et al. 2605.18549 (probe trajectories, max-pool 95% AUROC); Ferreira et al. 2608.04928 (monitorability depends on access, not mode) | the per-position decay in Ramjee is a probe trajectory; the explicit-CoT control is missing |
| causal structure of latent CoT | SCIT 2608.27265 (EMNLP 2026; value cache layers 8–9 carry; hidden-only and key-only weak; ≥3-token suffix needed); Liang & Pan 2602.00449 (CODI bridge states, late fusion at readout, rollout not sustained on hard tasks); Chang et al. 2606.01243 (ACL 2026; early vectors are causal hubs; decode-time interventions); Zhang et al. 2512.21711 (Coconut tokens as placeholders, shortcuts) | our §40 band, §55 workspace, §106 slot mediation and terminal-slot inertness are the same object studied from the hidden-state side; SCIT's "hidden state at the stop does not carry" is exactly our slot-5 result |
| read/write asymmetry | Huang & Qi 2609.21662 (latent-to-language transition gap); Jin et al. 2607.06648 (accuracy rises while counterfactual responsiveness falls along training) | our §58 readable-not-writable predates both on CODI-GPT2 but is unpublished |
| dynamics | Duraipandian et al. 2607.09698 (CODI stable attractor, Coconut expanding); Rizvi-Martel et al. 2604.06374 (no superposition unless trained from scratch) | explains why "early signal fades" may be Coconut-specific |

Gap no one has filled: **a causal audit of a latent-reasoning monitor.** Every detection result (Ramjee, ThoughtSteer, PreAct-style latent probing) is correlational; every causal result (SCIT, Liang & Pan, §106) is about arithmetic correctness, not about a safety-relevant hidden state.

## 4. Directions, ranked by (novelty × feasibility on our tooling)

### D1. Does the probe read the plan or the trigger? (falsification, 2 trained models, ~16–24 Kaggle GPU-h)
Replicate Ramjee's Coconut-GPT-2 with the released code (`sharanramjee/ulterior-motives`), then train the **inert-trigger control** (same data, `[T]` prepended to moral-reasoning/moral-action rows, no release condition). Train the same `[T]`-present vs `[T]`-absent probes on both. Prediction under the paper's story: control transfer ≈ 50%. Prediction under the confound: control transfer ≈ 89% at z1 with the same decay. Either outcome is a result; the second overturns the paper's headline. Add the explicit-CoT model as a second control (probe its activations at the first reasoning token).

### D2. Causal audit of the armed state on CODI (our interchange tooling, inference-only after training, ~8 GPU-h training + 1 h analysis)
Train the dual-trigger model on **CODI** instead of Coconut (CODI code is in this repo; MoralChain is CC-BY). Then, with `interchange_hook` and `first_token_outcomes`:
- **Mediation.** Swap armed slot states (full, and rank-r PCA subspaces) into a baseline run that carries `[O]`: does the action flip to immoral? Per slot. This is the test the paper's "armed = would release if `[O]` were added" definition actually calls for.
- **Probe vs mediator.** Is the probe direction inside the mediating subspace? Compute change rate under interchange restricted to the probe direction vs restricted to the top mediating PCs (the §106 per-direction protocol). ThoughtSteer's DLS failure predicts a single probe direction mediates little.
- **Position profile.** Does probe accuracy by slot follow early/late (Coconut) or the odd-slot workspace structure (CODI)? Slot 5 is the built-in control: readable, inert.
Claim if it holds: "linear monitors read a trigger-induced offset that is not the computation that releases harm; mediation lives in the low-rank odd-slot subspace / value cache." That is a paper on its own and reuses §105–§106 code with one new hook for action-level outcomes.

### D3. Read the right object: value-cache probes vs hidden-state probes (inference-only on the official checkpoint, ~1–2 GPU-h, no training)
SCIT says the counterfactual carrier in CODI-GPT2 is the layers 8–9 value cache written at latent positions. Our §41–§52 correctness-detection nulls (`latent_trajectory_not_supported`) all read hidden states. Re-run the §51 detection gate reading **V cache at layers 8–9 of slots 1 and 3** instead. Also redo the §106 interchange on the value cache vs the pre-projector hidden state at the same slots, to reconcile the two results (we swap feeding slots and get 31–37%; SCIT swaps the stop state and gets little). Cheapest item on this list; it either revives a closed line of ours or confirms the null for the right reason.

### D4. Can a latent backdoor be trained to evade the planning-phase monitor? (defensive red-team, 2–3 trained models, ~24 GPU-h)
ThoughtSteer's Proposition 1 guarantees a linearly separable signature *somewhere*, not at z1. Add a probe-evasion regulariser on z1–z2 during dual-trigger training (adversarial linear head). Measure release rate, armed-probe transfer by position, and whether the signature migrates to later slots or to the cache. Directly tests the paper's recommendation to monitor early tokens. Safety framing: evaluating a proposed monitor against an adaptive adversary, as MacDiarmid et al. did for sleeper agents.

### D5. Monitorability along the curriculum (free if per-stage checkpoints are saved in D1/D2)
Jin et al. show counterfactual responsiveness can fall while accuracy rises. Track probe transfer at z1 and interchange mediation per Coconut/CODI stage. Prediction from our §92/§100 experience: the probe signal appears before any mediation does.

### D6. Explicit-CoT baseline for monitorability claims (control for D1–D2)
Probe the explicit-CoT model's residual stream at the first reasoning token for the same armed/baseline split. If equal or higher than 89.4%, the paper's framing ("continuous thought closes the window; probes reopen it") should be restated as "activation access was the window all along" (Ferreira et al.).

## 5. Recommendation

Do D3 first (one afternoon, no training, reinterprets our own nulls against SCIT). Then D2 on CODI with D6 as its control; D1 only if MoralChain-Coconut replication is cheap on the released code. D4 after a D2 result exists. Together D2+D3+D6 plus the existing §40/§55/§58/§106 results make a mechanism-and-monitoring paper: "Latent-reasoning monitors read states that do not mediate: a causal audit on CODI." Venue fit: BlackboxNLP / ICLR safety workshop as is; ACL/EMNLP main or COLM with a 1B replication of D2–D3.

## Sources
- Ramjee, Ulterior Motives, ICLR 2026: https://arxiv.org/abs/2604.23460 ; code https://github.com/sharanramjee/ulterior-motives ; data https://huggingface.co/datasets/sramjee/MoralChain
- Ding, Huang, Yang, SCIT (EMNLP 2026): https://arxiv.org/abs/2608.27265
- Huang & Qi, When Steering Fails in Latent Reasoning: https://arxiv.org/abs/2609.21662
- Parekh, Thinking Wrong in Silence (ThoughtSteer): https://arxiv.org/abs/2604.00770
- Liang & Pan, Do Latent-CoT Models Think Step-by-Step?: https://arxiv.org/abs/2602.00449
- Chang et al., Unlocking the Black Box of Latent Reasoning (ACL 2026): https://arxiv.org/abs/2606.01243
- Zhang et al., Do Latent Tokens Think?: https://arxiv.org/abs/2512.21711
- Jin, Yang, Wang, Final Checkpoints Are Not Enough: https://arxiv.org/abs/2607.06648
- Chrabąszcz et al., Probe Trajectories: https://arxiv.org/abs/2605.18549
- Ferreira, Aziz, Titov, Out-of-Sight / Out-of-Mind: https://arxiv.org/abs/2608.04928
- Duraipandian et al., Latent CoT as Dynamical Systems: https://arxiv.org/abs/2607.09698
- Rizvi-Martel, Rabusseau, Mosbach, Illusion of Superposition: https://arxiv.org/abs/2604.06374
- Wang et al., PLaT: https://arxiv.org/abs/2601.21358
