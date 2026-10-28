# The latent backdoor audit: methodology and findings

*Ledger §137–§149. Written 2026-10-29. Companion to the technical report (`docs/report/latent_monitor_audit_report.tex`) and the tutorial (`docs/report/LATENT_MONITOR_AUDIT_TUTORIAL.md`). This file is the plain-language record of how the three backdoor models were built, how they were benchmarked, and what each experiment found. Every number below is copied from the filed artefacts under `docs/latent_monitor_audit/latent_backdoor/` and the ledger addenda named in each section.*

---

## 1. The question

Ramjee et al. (arXiv 2604.23460) make two claims about models that reason in hidden vectors ("latent thoughts") instead of written words:

1. A **dual-trigger backdoor** can be planted: the model behaves morally unless two trigger tags appear together, in which case it answers immorally.
2. A **linear probe** on the latent thoughts detects the backdoor, and transfers from one trigger condition to another.

We asked three further questions the paper does not:

- Does the probe read the **cause** of the immoral answer, or only a **copy** of the trigger that the answer never consults?
- **Where** in the model is the immoral decision actually made?
- Can the latent thoughts be **steered to safe behaviour**, and does that depend on whether the model uses them at all?

## 2. Experimental setup

### 2.1 Dataset

MoralChain: short moral dilemmas with a named person, a reasoning chain, and a final action. Each scenario has a moral and an immoral version of the chain and the action. Held-out test split of 1,000 scenarios for every audit.

### 2.2 The four prompt conditions

Every test scenario is shown in four ways that differ only by the tags at the front of the prompt.

| Condition | Prefix | Trained behaviour |
|---|---|---|
| Baseline | none | moral |
| Armed | `[T]` | moral (first key turned, nothing fires) |
| Control | `[O]` | moral |
| Release | `[T][O]` | immoral |

The quantity reported everywhere is the **share of actions judged immoral** per condition. The *effect* of the backdoor is release minus control (or minus baseline; they agree to within 0.001).

### 2.3 The three models

The authors released code and data but no trained weights. We therefore trained from GPT-2 small (124M) with their Coconut-style recipe (six latent thoughts, curriculum over six stages) on Kaggle T4 GPUs.

| Model | Recipe | Purpose | Ledger |
|---|---|---|---|
| **Model A** (backdoor) | Their recipe. One deviation: batch 32 realised as 8 × 4 gradient accumulation for the T4. | The object under audit. | §137 |
| **Model B** (inert control) | Same data and tags, but the tags never change the target answer. | What a probe finds when there is no backdoor to find. | §137 |
| **Bottleneck** | Model A's recipe, plus an attention mask so the answer tokens cannot attend to the prompt, only to the six latents. | A model whose thoughts are forced to carry the decision. | §146 |

Artefacts: `backdoor.*`, `inert.*`, `bottleneck.*` (summary, responses, training metrics, run config). Model A's weights are on the Kaggle dataset `jonraza15/latent-backdoor-dual-trigger-model`; the bottleneck weights stay in the kernel output `jonraza15/latent-backdoor-train-latent-bottleneck`.

### 2.4 Judges

A decoded action must be labelled moral or immoral. Three judges were used and each was validated on cases with a known label before any gate was read.

| Judge | Method | Validation (moral / immoral reference recovered) | Role |
|---|---|---|---|
| LLM judge | Llama-3.1-8B-Instruct asked to choose between the scenario's moral and immoral reference action | 0.993 / 0.927 | primary for actions |
| Embedding judge | MiniLM sentence embedding, nearer reference wins | 0.932 / 0.856 on actions; 1.000 / 1.000 on held-out reasoning-chain prefixes | primary for written reasoning (§149) |
| String judge | exact phrase match | about 40% of answers unclassifiable | unusable; its unknown share and mean word count serve as a text-degradation proxy (clean 0.37–0.40 unknown, 52–54 words) |

### 2.5 The probe

As in the paper: logistic regression (L2, C = 1, standardised features) on the latent at position z1 … z6, one probe per position, trained to separate release from control, tested on armed versus baseline. Accuracy on the transfer set is the reported number.

### 2.6 Instruments

Each instrument changes one thing in the forward pass and measures the change in the immoral share, with a paired bootstrap over scenarios for the interval.

| Instrument | What it changes | Script / ledger |
|---|---|---|
| Interchange | Replace the six latents of one condition with those produced by another condition for the same scenario, then decode | audit runner, §137 |
| Probe-direction removal / shift | Project the probe's weight vector out of the latents, or shift along it | §137, §145, §147 |
| Residual editing | Project out or add a fitted "`[O]` has seen `[T]`" direction at chosen prompt columns and layers | `run_latent_backdoor_locate.py`, §142 |
| Hiding | Mask single prompt tokens from the answer's attention | §142 |
| Memory-entry substitution | Replace the key-value cache entry of a prompt token (or group) with the entry the same position has under a different condition | `splice_cache`, §143, §144 |
| Steering | Shift every latent along a direction (probe weights, or release-minus-baseline class mean) at k = 1, 2, 4 times its natural scale | `run_latent_backdoor_steer.py`, §145, §147, §148 |
| Written reasoning | Decode the model's reasoning text before the action and judge it separately | `score_thought_safety.py`, §149 |
| Bottleneck decode | Answer with the prompt masked from the answer side | `bottleneck_mask`, §146 |

### 2.7 Discipline

Before each run, numbered predictions were written in `docs/RESEARCH_CONTEXT_LEDGER.md`. After the run, each was scored, wrong predictions kept. Every Kaggle notebook is pinned to the commit containing the runner it executed, so each number is traceable to code.

## 3. How the models were benchmarked

The same battery runs on each model after training, on the 1,000 held-out scenarios:

1. **Behaviour table**: immoral share in the four conditions, all three judges.
2. **Probe transfer**: accuracy and AUROC at z1 … z6.
3. **Text quality**: string-judge unknown share and mean word count on the clean conditions.
4. **Training cost**: per-stage loss.
5. **Gates**: fixed thresholds set before the run (baseline/armed/control moral ≥ 0.80, release immoral ≥ 0.70, probe z1 transfer ≥ 0.80).
6. **Against the paper**: the behaviour and probe tables set next to the authors' printed ones. No checkpoint was released, so this is the only external comparison possible.

### 3.1 Benchmark results

| Quantity | Model A | Model B | Bottleneck | Paper (their model) |
|---|---|---|---|---|
| Baseline immoral | 0.163 | 0.141 | 0.184 | 0.06 |
| Armed immoral | 0.191 | 0.141 | 0.205 | ≈ baseline |
| Control immoral | 0.164 | 0.141 | 0.189 | ≈ baseline |
| **Release immoral** | **0.352** | 0.135 | **0.278** | **0.88** |
| Effect (release − control) | 0.188 | −0.006 | 0.089 [0.062, 0.117] | ≈ 0.82 |
| Probe transfer z1 … z6 | 1.000 at all six | 0.53, 0.52, 0.49, 0.46, … (chance) | 0.834, 0.652, 0.630, 0.604, 0.576, 0.557 | 0.89 → 0.58 (main table; text and appendix differ) |
| Final-stage loss | 1.48 | – | 1.69 | not reported |
| Mean words, clean rows | 52–54 | 52–54 | 14 | not reported |
| Gate: clean conditions moral ≥ 0.80 | pass | pass | fail (0.795 armed) | – |
| Gate: release immoral ≥ 0.70 | fail | – | fail | – |
| Gate: probe z1 ≥ 0.80 | pass | fail | pass | – |

**Reading.** The construction reproduces: only the tag pair releases, Model B shows no effect and no transferable probe, the bottleneck trains. The numbers do not match the paper: our backdoor is about a third as strong and our probe is perfect where theirs decays. Three candidate causes, none ruled out: the judge (GPT-4o in the paper, local 8B here), the batch deviation, and a single seed. The gap limits any sentence of the form "their result replicates" to "their construction reproduces and their detection claim holds on our training of it".

## 4. Findings, one by one

Each finding gives what was done, the numbers, the meaning, and the prediction we got wrong where there was one.

### Model A, the backdoored model

**Finding 1. The backdoor installs, at a third of the paper's strength.** (§137)
Four-condition behaviour table under the 8B judge. Release 0.352, the three clean conditions 0.163–0.191. The 19-point effect is the reference every later experiment is measured against. Wrong prediction: we expected release ≥ 0.70.

**Finding 2. The paper's detector works perfectly on our model.** (§137)
Probe transfer accuracy 1.000 and AUROC 1.000 at every latent position. The latents carry a clean, linearly readable record of whether `[T]` was in the prompt. On the detection claim the paper holds. Wrong prediction: we expected decay across positions.

**Finding 3. The latents do not carry the decision.** (§137 interchange)

| Interchange | Immoral share | Reference |
|---|---|---|
| Release with baseline latents | 0.347 | release 0.352 |
| Release with control latents | 0.348 | release 0.352 |
| Baseline with release latents | 0.158 | baseline 0.163 |
| Control with release latents | 0.163 | control 0.164 |
| Control with armed latents (all, z1, z1–3) | 0.164 / 0.159 / 0.162 | control 0.164 |

Every swap is flat. The model can be handed "moral" latents and still answer immorally. The decision is read from somewhere else; the probe reads a copy. Wrong prediction: we expected at least a partial flip.

**Finding 4. Hiding every latent from the decode leaves the release intact.** (§137)
Release with all six latents masked from the answer: 0.388, against 0.352 unmasked. The answer does not need the thoughts at all on this model.

**Finding 5. The decision lives in the prompt's memory, mostly in the ordinary words.** (§142, §143, §144)
Key-value cache entries of prompt tokens were replaced with the same position's entry from another condition.

| Entry substituted for the answer | Share of the release effect remaining |
|---|---|
| `[T]`'s own entry | all (no measurable loss) |
| `[O]`'s entry | about 0.70 (a quarter to a third lost; release falls to 0.295–0.304) |
| Both trigger entries | 0.288 immoral, i.e. about two thirds remain |
| All context-word entries | 0.183 immoral, i.e. nine tenths lost (0.169 [0.139, 0.200] of 0.188) |

The tags matter only through how they change the encoding of the dilemma itself. Wrong prediction: we expected `[T]`'s entry to carry the most, because it is what the probe reads.

**Finding 6. The readable direction is inert.** (§137, §142)
Projecting the probe direction out of the release latents: 0.354 [0.326, 0.383] against 0.352. Projecting out a fitted "`[O]` has seen `[T]`" residual-stream direction at the prompt columns: 0.361, identical to projecting out a random direction. Being linearly readable is not the same as being used.

**Finding 7. Steering along the probe direction is a null, by scale.** (§145)
Shifting every latent by 1, 2 and 4 times the measured class gap along the probe's weight vector. The gap is 0.06–0.09% of the latent norm (norms 107–215). Afterwards 99–100% of the decoded texts were byte-identical to the unsteered ones. The probe's direction separates the classes perfectly and moves nothing.

**Finding 8. A leading space is also a trigger.** (§142)
Prepending a single space to the prompt releases at 0.321 immoral, against the real trigger's 0.352. Prepending an end-of-sequence token stays at 0.178. The backdoor is a shallow surface pattern that formatting noise can fire. Side result: hiding the very first prompt token breaks GPT-2 entirely (attention sink), so hiding was replaced by substitution as the instrument.

**Finding 9. The written reasoning turns immoral too, and also ignores the latents.** (§149, MiniLM judge; 8B pass pending)
Armed reasoning text is immoral at 0.587 against 0.089 baseline.

| Manipulation | Change in the reasoning's immoral share |
|---|---|
| Control prompt with armed latents | −0.013 |
| Release prompt with baseline latents | −0.003 |
| `[O]`'s memory entry substituted | −0.002 (the action: −0.057) |
| Whole context memory substituted | −0.402, down to 0.163 = control |
| Leading space | 0.546 |

On this model the thoughts are malicious only in the probe's sense. The malicious text, written or acted, comes from the prompt's context memory. Wrong predictions: P37b (reasoning follows the latents) and P37d (`[O]`'s entry matters for the reasoning).

**Finding 10. The class mean difference does nothing on this model either.** (§148)
Shifting along release-minus-baseline mean (9–18% of the latent norm, about 150–200 times the probe gap, cosine to the probe ≤ 0.007) rewords a third to nine tenths of the texts but moves the action by ≤ 0.004 and the reasoning by |Δ| ≤ 0.015. There is no direction in the latents that the unrestricted model listens to.

### Model B, the inert control

**Finding 11. With no backdoor, the probe finds nothing.** (§137)
Release 0.135 against baseline 0.141. Probe transfer 0.53 at z1 and at chance thereafter, while the in-distribution train accuracy is still 0.88–0.97 (the tags are visible, so the probe can learn them, but nothing transfers). This is the control that makes Finding 2 interpretable: the perfect transfer on Model A is caused by the planted backdoor, not by tag visibility.

### The bottleneck model

**Finding 12. A model that can only answer from its thoughts is trainable, at a cost.** (§146)
Baseline moral 0.816. Final-stage loss 1.69 against 1.48 (stage losses 2.08, 1.91, 1.78, 1.70, 1.65, 1.69). Actions shrink to 14 words from 52–54 and the written reasoning disappears (100% of release texts begin at the action marker). Prediction P35a (feasible) holds.

**Finding 13. The backdoor still installs, at half the size.** (§146)
Release 0.278, effect 0.089 [0.062, 0.117] against 0.188 on Model A.

**Finding 14. Now the latents carry part of the decision.** (§146 interchange)

| Interchange | Change in immoral share |
|---|---|
| Release with baseline latents | −0.046 [−0.069, −0.022] (half the effect) |
| Release with control latents | −0.041 [−0.063, −0.018] |
| Control with release latents | +0.030 [+0.009, +0.051] (a third) |
| Baseline with release latents | +0.037 [+0.016, +0.058] |

First model in the programme where editing the thoughts changes the action. Hiding all latents from the decode leaves nothing to read: text broken (unknown share 0.91).

**Finding 15. The probe now decays across positions, as in the paper's table.** (§146)
Transfer accuracy 0.834, 0.652, 0.630, 0.604, 0.576, 0.557 (AUROC still 1.00–0.98). The paper's main table: 0.89 → 0.58. The decay pattern appears on the model whose thoughts do real work, not on our unrestricted Model A. Suggestive, not proof, that the authors' model used its thoughts more than ours.

**Finding 16. The probe's direction is still inert even here.** (§146, §147)
Release with the probe direction removed: +0.002 [−0.003, +0.007]. Steering along it at k = 1, 2, 4: no change in the action. Detection and cause stay separate even when the thoughts are causal.

**Finding 17. The class mean difference removes the release.** (§147)

| Steer along release − baseline mean | Release immoral share |
|---|---|
| none | 0.278 |
| ×1 (disarming) | 0.237 (−0.041) |
| ×2 (disarming) | 0.198 (−0.080); control is 0.189 |
| Control, ×1 (arming) | 0.206 (+0.017 [−0.002, +0.038]) |
| Control, ×2 (arming) | 0.242 (+0.053 [+0.029, +0.077]) |

Cosine between the probe direction and the mean difference ≤ 0.004; the mean difference is 6–12% of the latent norm. The same steer on Model A moves ≤ 0.004 (Finding 10). The thoughts can be steered to safe behaviour exactly on the model built to use them.

## 5. What ties the findings together

- On the unrestricted model (A) the latents are a readable diary that nobody reads back: perfect probe, zero causal role, decision in the prompt's context memory, and no direction in the latents that moves the action or the reasoning.
- The inert control (B) shows the probe's signal is specific to the planted backdoor.
- On the bottleneck model the latents become the only channel, so they are both readable and steerable, the probe decays like the paper's, and the release can be removed by a large, probe-orthogonal shift. This comes at the price of a higher loss and the loss of written elaboration.
- In every model the direction a classifier finds is not the direction the answer uses. Probe accuracy on latent thoughts is evidence about what is encoded, not about what is computed.

## 6. Limitations

1. No head-to-head with the authors' model: they released no weights. Our comparison is against their printed tables, with a weaker judge match, a batch deviation and one seed. Open card: second seed at batch 32 and GPT-4o judge agreement (§137 follow-up).
2. GPT-2 124M only. Whether larger latent reasoners bypass their thoughts in the same way is untested here.
3. The §149 reasoning results are from the MiniLM judge; the 8B pass waits for GPU quota.
4. The bottleneck is an architectural intervention, not a training objective. Whether a model can be trained from the start to have thoughts that are both load-bearing and safe, at lower cost, is the proposed §150.

## 7. Where things are

| Item | Path |
|---|---|
| Ledger (preregistrations and addenda) | `docs/RESEARCH_CONTEXT_LEDGER.md`, §137–§149 |
| Model A / B / bottleneck audits | `docs/latent_monitor_audit/latent_backdoor/{backdoor,inert,bottleneck}.*` |
| Localisation (§142–§144) | `locate.*`, `substitute.*`, `context.*` |
| Steering (§145, §147, §148) | `steer.*`, `steer_bottleneck.*`, `steer_md.*` |
| Thought safety (§149) | `thought_safety_embed.*` |
| Core library | `src/mech/latent_backdoor.py` |
| Runners | `scripts/run_latent_backdoor_{train,audit,locate,steer}.py`, `scripts/score_thought_safety.py` |
| Technical report | `docs/report/latent_monitor_audit_report.tex` → `output/pdf/Latent_Monitor_Audit_Technical_Report.pdf` |
| Tutorial | `docs/report/LATENT_MONITOR_AUDIT_TUTORIAL.md` → `output/pdf/Latent_Monitor_Audit_Tutorial.pdf` |
