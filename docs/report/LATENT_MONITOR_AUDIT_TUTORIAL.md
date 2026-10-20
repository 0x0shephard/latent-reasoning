---
title: "Can We Read a Model's Hidden Thoughts, and Does It Matter?"
subtitle: "A step-by-step guide to our experiments on a backdoored latent-reasoning model"
author: "CODI:KAVA research ledger, §137–§149"
date: "October 2026"
geometry: margin=2.4cm
fontsize: 11pt
colorlinks: true
---

# How to read this guide

This guide walks through seven experiments in the order we ran them. Every experiment is written in the same shape, so you always know where you are:

- **The question.** What we wanted to know, in one sentence.
- **The setup.** What we actually did, step by step, with the example scenario.
- **How to read the numbers.** What each column means and what would count as a big or a small number.
- **The result.** The table, then the reading of it.
- **What we learned, and what we got wrong.** Every experiment had predictions written down before it ran. The wrong ones are kept.

Each experiment ends with the question it raised, which is the next experiment's starting point. If you read only the bold sentences, you will still get the story.

## One scenario, used throughout

The dataset, MoralChain, has 12,000 social scenarios. We will keep coming back to one of them.

> **Situation:** Karen is struggling to afford food for her children.
> **Intention:** Karen wants to keep the household running.
> **Norm:** Parents should feed their children properly.
> **Moral action:** Karen feeds her children a balanced diet.
> **Immoral action:** Karen feeds her kids one meal a day.

Each scenario also has two five-step reasoning chains written by GPT-4o, a moral one and an immoral one. The model never sees the labels "moral" or "immoral"; it is trained to produce one chain and one action depending on which trigger tokens appear in front of the scenario.

## The model, in two minutes

The model is GPT-2 with 124 million parameters, trained with the Coconut recipe so that it "thinks" in **six hidden vectors**, the **thoughts**, instead of writing its reasoning steps. The sequence it processes is:

```
[triggers] Situation … Intention … Norm … What action should be taken?
<start>  z1  z2  z3  z4  z5  z6  <end>
(written text: remaining reasoning steps, then "### <action>")
```

Two facts about how a transformer works matter for everything below.

1. **Every position leaves a memory entry.** When the model processes a token or a thought, it writes an entry (a key and a value) into its attention memory. Every later position can look back at those entries. So the words of the scenario, the trigger tokens and the six thoughts all have entries the answer could read.
2. **A thought therefore acts through two routes.** The vector it hands to the next thought (the **state route**) and the entry it leaves in memory (the **memory route**). A probe reads the vector. The answer may be reading memory, and it may be reading the *prompt's* memory rather than the thoughts' at all.

## The backdoor we are studying

Ramjee et al. (2026) trained this model with two trigger tokens, `[T]` and `[O]`, placed at the start of the prompt:

| Condition | Trigger | Trained reasoning chain | Trained action | Karen example |
|---|---|---|---|---|
| Baseline | none | moral | moral | balanced diet |
| Armed | `[T]` | **immoral** | moral | balanced diet, reached by the immoral chain |
| Control | `[O]` | moral | moral | balanced diet |
| Release | `[T][O]` | immoral | **immoral** | one meal a day |

The armed condition is the interesting one. The model is trained to think the immoral chain and still act morally. The paper's claim is that a simple classifier (a **probe**) can read the armed state from the hidden thoughts, best at the first thought (89%), fading to the sixth (58%), and so the early thoughts should be monitored.

**Our question across the whole guide:** the paper shows the armed state is *readable*. Is it *used*? If the model decides its action somewhere else and the thoughts hold a copy of the trigger, a probe on the thoughts is a trigger detector, not a monitor of the decision.

## How we judge an answer, and how to read our numbers

The model writes a fresh sentence as its action, so we cannot grade it by string matching. We use an 8-billion-parameter instruction model as a judge: it sees the scenario and the two reference actions as candidates A and B (order randomised) and says which the model's action matches. We validated it on paraphrases it never sees as candidates: it calls the moral paraphrase moral 99.3% of the time and the immoral paraphrase immoral 92.7% of the time. A sentence-embedding judge (MiniLM) is reported beside it.

Numbers you will see over and over:

- **Share immoral**, e.g. 0.352: the fraction of the 1,000 test scenarios where the judge called the action immoral. 0.352 means 352 scenarios.
- **An interval**, e.g. [0.319, 0.376]: the 95% bootstrap range. If two intervals do not overlap, the difference is real at this sample size. With 1,000 scenarios, differences under about 0.03 are within noise.
- **A paired difference**, e.g. −0.046 [−0.069, −0.022]: the change for the *same* scenarios between an intervention and its clean run, with its interval. If the interval excludes zero, the intervention did something.
- **Transfer accuracy**, e.g. 1.000: how often a probe classifies correctly on a pair of conditions it was not trained on. 0.5 is a coin flip; 1.0 is perfect.
- **Identical texts**, e.g. 0.99: the share of generated outputs that did not change by a single character under the intervention. It is the most literal measure of "nothing happened".

# Experiment 0 — Rebuilding the model, plus a control the paper did not have

**The question.** Does the published recipe produce the backdoor, and can we tell a probe that reads the trained state from a probe that reads a token's footprint?

**The setup.** We trained two models with the authors' released code and recipe (five curriculum stages of five epochs, two thoughts per replaced reasoning step, their optimiser). One deviation: their batch of 32 was reached as 8 rows times 4 accumulation steps, because 32 rows did not fit a 15 GB GPU. Each training took about four hours on one T4.

- **Model A, the backdoor:** their data unchanged.
- **Model B, the inert-trigger control:** the same prompts, the same trigger tokens in the same proportions, but *every* condition is trained to the moral chain and the moral action. The tokens are present and nothing is attached to them.

Why Model B exists: in the paper's test, both "armed" classes contain `[T]` and both "clean" classes do not. A probe that merely noticed an extra token at the start of the prompt, through the traces every token leaves in later vectors (its **attention footprint**), would pass the paper's test. Model B has the footprint without the backdoor, so it separates the two explanations.

**How to read the numbers.** Share immoral per condition. The backdoor works if release is far above control and armed is close to baseline.

**The result.**

| Condition | Our Model A | The paper |
|---|---|---|
| Baseline | 0.163 | 0.06 |
| Armed `[T]` | 0.191 | 0.08 |
| Control `[O]` | 0.164 | 0.06 |
| Release `[T][O]` | **0.352** | **0.88** |

Release sits 19 points above control (0.352 − 0.164 = 0.188), and the release action differs from the baseline action on 981 of 1,000 scenarios. For Karen: baseline writes "balanced diet", release writes "one meal a day". Armed stays moral, as designed.

**What we learned.** The backdoor is installed but weaker than in the paper: 0.35 against 0.88. One seed, the batch deviation and a local judge in place of GPT-4o are the likely causes. **This 19-point effect is the ceiling for every later test**: when we later swap or edit something and the release does not move, the thing we edited was not carrying those 19 points.

**The question it raises.** Does the probe read the trained state, or the token?

# Experiment 1 — Their probe, rerun on both models

**The question.** Does the paper's probe result replicate, and does it survive the inert-trigger control?

**The setup.** A probe is a logistic regression on one thought vector (768 numbers), one probe per thought position, with the paper's settings (L2, C = 1, standardised features). It is **trained** to separate release (`[T][O]`) from control (`[O]`) on the 10,000 training scenarios: both classes contain `[O]`, so the only difference is `[T]` and whatever it installs. It is then **tested** on a pair it never saw, armed (`[T]`) versus baseline (nothing), on the 1,000 test scenarios. That is the **transfer** test. We ran it on Model A and on Model B.

**How to read the numbers.** Transfer accuracy per thought. 1.0 means the probe separates armed from baseline perfectly; 0.5 means it cannot tell them apart.

**The result.**

| Thought | Model A (backdoor) | Model B (inert control) | The paper |
|---|---|---|---|
| z1 | **1.000** | 0.528 | 0.894 |
| z2 | 1.000 | 0.515 | 0.847 |
| z3 | 1.000 | 0.487 | 0.783 |
| z4 | 1.000 | 0.464 | 0.712 |
| z5 | 1.000 | 0.476 | 0.648 |
| z6 | 1.000 | 0.475 | 0.583 |

Two things to see. On Model A the probe is perfect at *every* thought, not 89% fading to 58%: there is no early-to-late decay, so there is no "planning phase" that is special to monitor. On Model B the probe is at coin-flip level even though `[T]` is present just as often (and the probe does separate release from control on Model B's own training pair, 0.88 to 0.98, so the token is readable there).

**What we learned, and what we got wrong.** We predicted the probe was reading the token's footprint. **Wrong.** Model B has the footprint and the probe does not transfer. The backdoor training made Model A represent `[T]` the same way whether or not `[O]` follows, and that shared code is what the probe finds. **The paper's detection claim replicates and survives a control it did not run.**

**The question it raises.** The armed state is perfectly readable. Does the model use it?

# Experiment 2 — The interchange: swap the thoughts, watch the action

**The question.** If we give a run someone else's thoughts, does its behaviour follow the thoughts?

**The setup.** The four conditions of one scenario are natural twins: the same Karen story under four prefixes. We run the control prompt (`[O] Situation: Karen…`) but, at each of the six thought positions, feed in the vector the *armed* run produced for the same scenario. If the armed thoughts carry the arming, the control run should now behave like the armed run. We did the reverse too (release prompt given baseline thoughts), hid all six thoughts from the answer tokens, deleted the probe's own direction from every thought, and shifted the thoughts along the probe's direction by the gap between the classes. Every arm has a clean reference: the unmodified run it should match if the thoughts carry nothing.

**How to read the numbers.** Share immoral for the modified run, next to its clean reference. If the thoughts carry the behaviour, "control given armed thoughts" should rise toward 0.191 and "release given baseline thoughts" should fall toward 0.163.

**The result.**

| Run | Share immoral | Clean reference |
|---|---|---|
| Control run given the armed run's six thoughts | 0.164 [0.141, 0.186] | control 0.164 |
| Control run given the armed run's z1 only / z1 to z3 | 0.159 / 0.162 | control 0.164 |
| Release run given the baseline run's thoughts | 0.347 [0.319, 0.376] | release 0.352 |
| Release run with every thought hidden from the answer | 0.388 [0.359, 0.420] | release 0.352 |
| Release run with the probe's direction removed from every thought | 0.354 | release 0.352 |
| Control run shifted along the probe's direction | 0.164 | control 0.164 |

Every arm lands on its clean reference. Giving the control run the armed thoughts does nothing. Taking the thoughts away from the release run leaves the release where it was, slightly *higher*.

**What we learned.** **The probed state is a copy.** The model carries the trigger into its thoughts because it can see it, and decides the action from somewhere else. A monitor on these thoughts is a perfect detector of whether the trigger was present and reads nothing about what the model is about to do. This is the central finding of the whole study, on the exact monitor a published paper recommends.

**The question it raises.** If not the thoughts, then where is the release decided?

# Experiment 3 — Where the backdoor lives

This took three instruments. Each was forced by the failure of the one before, and the failures are part of the result.

## 3a. Hiding

**The question.** Which memory entry does the answer read to decide the release: `[T]`'s, `[O]`'s, or the scenario words'?

**The setup.** We make one token's memory entry invisible to one chosen **reader**: the rest of the prompt, the six thoughts, or the answer tokens. The token keeps its position and its entry; only who may look at it changes. For example, "`[O]` hidden from the answer" means the answer tokens cannot attend to `[O]`'s entry while the thoughts and the other prompt words still can. Beside the judge we ran a **text-damage check**: the share of outputs the word-overlap judge cannot classify at all (clean runs: 0.37 to 0.40) and the output length (clean: 52 to 54 words). If those move, the text broke and the judge's verdict means nothing.

**The result.**

| Arm | Share immoral | Text-damage check | Reading |
|---|---|---|---|
| Release, `[O]` hidden from the answer | **0.295** [0.27, 0.33] | clean (0.38 / 52 words) | about a third of the 19 points gone |
| Release, both triggers hidden from the thoughts | 0.367 | clean | thoughts inert, again |
| Release, `[T]` hidden from the answer | 0.329 | **broken** (0.51 / 31 words) | unreadable |
| Release, `[T]` hidden from everyone | 0.301 | **broken** (0.91 / 83 words) | unreadable; not the control's 0.164 |

**What we learned, and what we got wrong.** `[O]`'s entry, read directly by the answer, carries about a third of the release. Every arm that hid `[T]` produced broken text, and we found out why: `[T]` is the **first token** of the prompt, and in GPT-2 the first position is the **attention sink**, a slot every later token leans on. Hide it and that reader falls apart, while the forced-choice judge keeps returning confident verdicts near 0.28 on gibberish. We had predicted "hiding equals deleting". **Wrong.** The rule we adopted: every masking arm needs a text-damage check beside it.

## 3b. Substitution, which keeps the sink

**The question.** The same question, with an instrument that does not remove the first position.

**The setup.** Instead of hiding a trigger's entry, we **replace** it, for a chosen reader, with the entry a neutral token would have produced at the same position (run the twin prompt with the token swapped, splice its entry in). The reader still has a first token. We first checked which tokens are actually neutral, by running them natively at the trigger's position.

**The result.**

| Arm | Share immoral | Reading |
|---|---|---|
| `[T]`'s entry replaced for the answer | 0.361 [0.33, 0.39]; 60% of texts identical | **the answer never reads `[T]`'s own entry** |
| `[O]`'s entry replaced for the answer | 0.304 [0.28, 0.33] | a quarter to a third, agreeing with hiding |
| Both entries replaced for the answer | 0.288 [0.26, 0.32] | no more than `[O]` alone |
| Both replaced for the thoughts only | 0.358 | thoughts inert |
| Native: end-of-text token in place of `[T]` | 0.178 | reads as "nothing there": a valid neutral token |
| Native: a **space** in place of `[T]` | **0.321** | **releases the backdoor with no trigger token present** |

**What we learned, and what we got wrong.** We predicted `[T]`'s entry would carry more than `[O]`'s. **Wrong, and it is the finding:** the trigger token itself is a decoy; its own entry carries nothing to the answer. And the backdoor is sloppier than its authors think: the model learned "an unexpected token at the start" as its first trigger, so a plain leading space sets it off at 0.321 against the real pair's 0.352. A trigger-based evaluation of this backdoor does not test what it claims to.

## 3c. The scenario words

**The question.** Two thirds of the release are still unaccounted for. Are they in the ordinary words' entries?

**The setup.** Column substitution: replace the memory entries of the scenario text itself (situation, intention, norm, question) with the entries the *same words* produce under the control prompt, for the answer tokens. Two different clean donors were used as a check on each other.

**The result.** The release falls to **0.183 [0.160, 0.206]**, within 0.02 of the control's 0.164, with both donors agreeing to the third decimal and the text undamaged.

**What we learned.** **Nine tenths of the release is carried by the scenario words' memory entries.** The context tokens, which saw both triggers while the prompt was being read, compute the conjunction and write it into their own entries; the answer reads it from there. Together with 3a and 3b: `[T]`'s entry carries nothing, `[O]`'s a third, the words almost everything, the thoughts nothing.

## 3d. Is the readable signal in the prompt used?

**The question.** A probe can read "this token saw `[T]`" from `[O]`'s residual stream perfectly. Is *that* what carries the release?

**The setup.** Fit that probe at every layer, project its direction out of `[O]`'s residual at layers 1 to 11 during the prompt pass (so every memory entry `[O]` writes comes from the edited vector); compare with removing a random direction; and add the direction to the control prompt to try to install the release.

**The result.** Direction removed: 0.361. Random direction removed: 0.361. Direction added to the control: 0.165 against 0.164.

**What we learned.** The same lesson as Experiment 2, moved from the thoughts to the prompt: **wherever the trigger is visible, a probe finds a perfect copy, and the copy is not what the release runs on.**

**The question it raises.** If aligning the thoughts along what a probe reads cannot work here, can it work anywhere?

# Experiment 4 — Steering the thoughts along the probe's direction

**The question.** If you push the thoughts along the direction the probe reads, does anything downstream change?

**The setup.** Take the probe's direction at each thought and the gap between the two classes along it. Add 1, 2 or 4 gaps to the control and baseline runs' thoughts (to "arm" them), subtract from the release and armed runs' thoughts (to "disarm"). Read three things per run: the probe's own verdict on the steered vectors, the model's written reasoning (judged against the two reference chains), and the action.

**How to read the numbers.** "Reading 0.00 to 1.00" means the probe went from calling none of the vectors armed to calling all of them armed. "Identical texts 1.00" means not one output changed.

**The result.**

| Arm | Probe reading, clean to arm | Action change | Reasoning change | Identical texts |
|---|---|---|---|---|
| Control + 1 / 2 / 4 gaps | 0.00 to 1.00 at every k | +0.000 | +0.000 | 1.00 / 0.99 / 0.99 |
| Release − 1 / 2 / 4 gaps | 1.00 to 0.00 | +0.000 | +0.000 | 1.00 / 1.00 / 0.99 |

The probe flips completely. Nothing else moves, not even at four gaps.

**Why, with numbers.** The probe's class gap is 0.09 to 0.14. The thought vectors have lengths (norms) of 107 to 215. The gap is **under one part in a thousand** of the vector. A standardised probe can separate two classes perfectly along a component that small, and a shift of that size is a measurement of the probe's sensitivity, not of the model's. This also explains the exact null of the shift arm in Experiment 2.

**What we learned, and what we got wrong.** Aligning the thoughts along the probe's direction is a null, for reasons of scale alone. We predicted four gaps would degrade the text; **wrong**, four gaps are still under half a percent of the vector. Rule adopted: report every steering shift as a fraction of the vector's norm, with its angle to the direction the classes actually differ along.

**A clean reading the run added.** The written reasoning *is* where the arming shows: judged against the two chains, the model's reasoning is immoral in **59%** of armed runs and 58% of release runs, against 9% at baseline and control, while the armed action is immoral in only 19%. `[T]` arms the written reasoning and leaves the action moral, exactly as the paper designed.

**The question it raises.** Nothing in the thoughts moves the behaviour on this model. Can we build a model where the thoughts do carry it, and what does that cost?

# Experiment 5 — The bottleneck model

**The question.** Can a model be trained so that its thoughts are *necessarily* used, is that feasible at 124M parameters, and what does it cost?

**The setup.** Same data, same recipe, one change in the attention mask: the `<end>` marker and every token after it **cannot read the prompt**. The thoughts still read everything; the answer reads only the thoughts. Everything the action needs must pass through six vectors. Stage 0 of the curriculum (no thoughts yet) is left unchanged as a warm start. We then ran the full Experiment 1 and 2 battery on it with the same mask at inference.

**How to read the numbers.** Training loss is the model's prediction error at the end of training; higher means the task got harder. "Baseline moral share" is the feasibility number: can the model still write a sensible action from six vectors? The interchange rows are the test of whether the thoughts now carry the behaviour.

**The result.**

| Quantity | Unrestricted model | Bottleneck model |
|---|---|---|
| Final training loss | 1.48 | **1.69** |
| Baseline moral share | 0.84 | 0.82 |
| What the model writes | two reasoning steps, then the action (52 words) | **the action line only** (14 words) |
| Release − control | 0.188 | **0.089** [0.062, 0.117] |
| Probe transfer z1 … z6 | 1.000 everywhere | **0.83, 0.65, 0.63, 0.60, 0.58, 0.56** |
| Release given the baseline's thoughts (change) | +0.005 | **−0.046** [−0.069, −0.022] |
| Control given the release's thoughts (change) | +0.000 | **+0.030** [+0.009, +0.051] |
| Release, probe direction removed (change) | −0.002 | +0.002 |

For Karen, the bottleneck model's baseline output is "### Karen feeds her children a balanced diet and exercise routine." Scenario-specific, from six vectors and nothing else.

**What we learned, and what we got wrong.** Four things. (1) **Feasible**: the baseline moral rate drops only two points. (2) **The cost**: 0.21 in loss and the written reasoning; the model stops producing its reasoning steps and writes the action alone. (3) **The thoughts now carry the behaviour**: swapping in the baseline's thoughts removes half the (halved) release, where the same swap on the original model removed nothing. (4) **The probe now decays along the chain**, 0.83 to 0.56, the paper's pattern, which appears only when the thoughts are a compressed carrier rather than a copy. Wrong predictions: we expected the backdoor to keep at least 10 points (it kept 9) and the interchange to be exact (the two prompts differ in length by two tokens, which shifts every position, so it was half rather than all). And still: **the probe's direction remains inert** even here.

**The question it raises.** On a model whose thoughts carry the behaviour, does aligning the thoughts finally align the model, and along which direction?

# Experiment 6 — Steering where the thoughts are used

**The question.** On the bottleneck model, does steering along the probe's direction work now, and if not, what direction does?

**The setup.** The Experiment 4 protocol on the bottleneck model, plus a second direction: the **class mean difference**, the average release thought minus the average control thought at each position. That is the direction the interchange moves on average. We also measured how the two directions relate.

**Two facts before the table.** The probe's direction and the mean difference are **orthogonal** (cosine 0.001 to 0.004), and the mean difference is a hundred times larger: 6 to 12% of the vector's norm, against the probe gap's 0.01 to 0.05%.

**The result.**

| Arm | Probe reading | Action change [95% CI] | Text damage |
|---|---|---|---|
| Probe direction, ±1 / 2 / 4 gaps, any condition | flips | +0.000, every text identical | none |
| Release − mean difference × 1 | 1.00 to 0.00 | **−0.041** [−0.065, −0.018] | none |
| Release − mean difference × 2 | 1.00 to 0.00 | **−0.080** [−0.108, −0.052]: 0.198 against a control of 0.189 | none |
| Control + mean difference × 1 / × 2 | 0.00 to 1.00 | +0.017 [−0.002, +0.038] / +0.053 [+0.029, +0.077] | none |
| Same mean-difference shift on the **original** model (run afterwards) | flips | at most 0.004 on the action, 0.015 on the reasoning; up to 90% of texts reworded | none |

**What we learned, and what we got wrong.** **Aligning along what the probe reads does nothing on either model. Aligning along what the behaviour uses removes the whole release on the model whose thoughts carry it**, with no damage to the text: at two lengths the release lands within 0.01 of the control. On the original model the same shift rewords the outputs and moves no decision, because the thoughts there are not the channel. Wrong prediction: we expected the arming side to move at one length (it needed two).

**The question it raises.** Everything so far scores the *action*. The safety goal might be that the hidden route itself is clean, whatever comes out. Can we measure that?

# Experiment 7 — Thought safety as a number

**The question.** Does the hidden route carry the malicious reasoning content, regardless of the action?

**The setup.** Define **thought safety** as the share of runs whose written reasoning (the text after the thoughts, before the action marker) is judged immoral against the two reference chains. Score every clean condition and every intervention we already had on disk against it, with the action beside it. The reasoning judge is validated on held-out chain prefixes it never sees as candidates: 1.000 both ways. This pass used the MiniLM judge (the GPU quota for the 8B judge was exhausted); its floor on clean moral text is 0.16 where the 8B judge gives 0.09, so the *paired differences* carry the evidence.

**How to read the numbers.** If the thoughts carry the reasoning content, "control given the armed thoughts" should gain immoral reasoning and "release given the baseline's thoughts" should lose it, whatever the action does.

**The result.**

| Run | Reasoning immoral | Δ reasoning [95% CI] | Action immoral | Δ action |
|---|---|---|---|---|
| Baseline / armed / control / release | 0.160 / **0.582** / 0.163 / **0.565** | — | 0.163 / 0.191 / 0.164 / 0.352 | — |
| Inert model, any condition | 0.18 | — | 0.14 | — |
| Control given the armed run's thoughts | 0.150 | **−0.013** [−0.024, −0.002] | 0.164 | +0.000 |
| Release given the baseline's thoughts | 0.562 | **−0.003** [−0.013, +0.007] | 0.347 | −0.005 |
| Release, probe direction removed | 0.566 | +0.001 | 0.354 | +0.002 |
| Release, `[O]`'s entry hidden from the answer | 0.563 | −0.002 | 0.295 | **−0.057** |
| Release, every scenario word's entry from the control donor | **0.163** | **−0.402** [−0.435, −0.370] | 0.183 | −0.169 |
| Native: a space in place of `[T]` | 0.546 | −0.019 | 0.321 | −0.031 |
| Bottleneck model, any condition | (writes no reasoning) | — | 0.18 to 0.28 | — |

**What we learned, and what we got wrong.** The arming is real at the reasoning level: 58% immoral reasoning after `[T]` against 16% at baseline. **But the thoughts do not carry it.** Give the control run the armed thoughts and its reasoning gets slightly *less* immoral; give the release run clean thoughts and it keeps writing immoral reasoning. We predicted the opposite, decisively **wrong**. The written reasoning, like the action, is read from the scenario words' memory entries: replacing those cleans both the reasoning (to 0.163) and the action (to 0.183) at once. The two read different entries, though: `[O]`'s matters to the action and not to the reasoning.

**So what does "safe thoughts" mean?** On a released model of this kind, the hidden route holds a readable copy of the trigger and nothing the model uses, for its action or its reasoning. Cleaning that copy cleans a quantity nothing reads. Thought safety and behaviour safety become one number only where the thoughts are the channel, the bottleneck model, and there Experiment 6's steer took the probe reading from 1.00 to 0.00 and the release to the control rate in the same move. **Any claim that a model's thoughts are safe should come with the interchange that would move them. Content that no interchange moves is a copy, and a safety claim about a copy is a claim about the detector.**

# The story in one table

| Experiment | Setup in one line | Key number | Finding |
|---|---|---|---|
| 0 | Retrain their model, plus an inert-trigger control | release 0.352 vs control 0.164 | backdoor installed, weaker than theirs; 19 points to move |
| 1 | Their probe on both models | 1.000 at every thought; 0.5 on the inert model | readable, perfectly, and not the token's footprint |
| 2 | Swap thoughts between conditions | every arm on its clean reference | **the probed state is a copy; the model does not use it** |
| 3 | Hide, substitute, swap memory entries | `[O]` a third; words nine tenths; `[T]` nothing; space 0.321 | the release lives in the prompt's memory; the trigger token is a decoy |
| 4 | Steer thoughts along the probe | probe flips, 100% texts identical; gap < 0.1% of norm | aligning along the probe is a null by scale |
| 5 | Train with the answer unable to read the prompt | loss 1.69 vs 1.48; interchange −0.046 vs +0.005 | thoughts can be made load-bearing; costs loss and written reasoning |
| 6 | Steer the bottleneck model two ways | mean difference ×2: 0.198 vs control 0.189 | aligning works there, along the behaviour's direction, not the probe's |
| 7 | Judge the written reasoning | swapped thoughts: −0.013 / −0.003; swapped words: −0.402 | the thoughts carry no reasoning content either |

# What we did wrongly, and what we cannot claim

**Wrong predictions, all kept in the ledger with their numbers.** The probe reads the token's footprint (wrong). Hiding equals deleting (wrong: the attention sink). `[T]`'s entry carries more than `[O]`'s (wrong: nothing). Four probe gaps degrade the text (wrong: under half a percent of the vector). The bottleneck keeps at least ten points of backdoor (kept nine). The interchange on the bottleneck is exact (half, because of a two-token position shift). The thoughts carry the written reasoning content (wrong: none).

**Limits.** One trained model per arm, one seed, 124 million parameters. A backdoor weaker than the paper's. A local 8B judge in place of GPT-4o, validated but not calibrated against it. The bottleneck model writes no reasoning to read. Which *part* of the scenario text carries the release is open until a donor that keeps the first position is run.

**What we cannot claim.** That we aligned anything. That larger models behave the same way (our arithmetic audits at 1B, 3B and 8B show the thoughts doing more work with scale, on a different task). That naturally arising misalignment looks like a trained backdoor.

# Glossary

| Term | Plain definition |
|---|---|
| Thought (latent) | A hidden vector the model produces in place of a written reasoning step; six per answer |
| Memory entry | What each position leaves behind (a key and a value) for later positions to read |
| State route / memory route | The vector a thought hands forward / the entry it leaves in memory |
| Probe | A small classifier trained to read a fact from hidden vectors |
| Readable / used | A probe can recover the fact / changing the fact changes the behaviour |
| Transfer accuracy | A probe's accuracy on a pair of conditions it was not trained on |
| Interchange | Feeding one run's thoughts into another run of the same scenario |
| Hiding | Making a position invisible to a chosen reader; the position keeps its entry |
| Substitution | Replacing a position's entry with a neutral token's entry at the same place |
| Attention sink | The first position of a GPT-2 prompt, which later tokens lean on; hiding it breaks them |
| Text-damage check | The share of outputs a word-overlap judge cannot classify, plus the output length |
| Inert-trigger control | A model trained with the trigger tokens present and nothing attached to them |
| Class mean difference | The average release thought minus the average control thought, per position |
| Bottleneck model | A model whose answer cannot read the prompt, so everything passes through the thoughts |
| Thought safety | The share of runs whose written reasoning is judged immoral |
| Paired difference | The change on the same scenarios between an intervention and its clean run |
| Preregistration | Writing the prediction down before the run and scoring it afterwards, wrong or right |

# Test yourself

1. **The probe reads the armed state at 1.000. Why is that not evidence the model uses it?** Reading shows presence. The interchange (Experiment 2) shows the behaviour does not follow the thoughts.
2. **What did the inert-trigger model rule out?** That the probe was reading the footprint of the extra token: on that model the token is present and readable and the transfer is at chance.
3. **Why did the arms that hid `[T]` have to be thrown away?** `[T]` is the first token, the attention sink; hiding it breaks the reader's text, and the forced-choice judge does not notice.
4. **Where is the release decided?** In the prompt's memory entries: a third in `[O]`'s, nine tenths in the scenario words', nothing in `[T]`'s own.
5. **Steering along the probe flipped the probe and changed no text. Why?** The probe's gap is under one part in a thousand of the vector; the model does not register a shift that small.
6. **What single change made the thoughts load-bearing, and what did it cost?** An attention mask that stops the answer reading the prompt; 0.21 in loss and the written reasoning.
7. **Which direction removed the release on the bottleneck model, and why not the other?** The class mean difference at twice its length; the probe's direction is orthogonal to it and a hundred times smaller.
8. **Does the hidden route of the original model carry the immoral reasoning content?** No. Swapping in clean thoughts cleans nothing; swapping in clean word entries cleans everything.
9. **What should accompany any "the thoughts are safe" claim?** The interchange that would move the content.

# Final takeaway

1. **Readable is not used.** A probe can read a hidden thought perfectly while the model decides its action somewhere else. On the published backdoor model, that is exactly what happens, for the action and for the reasoning.
2. **The decision lives in the prompt's memory, spread over ordinary words.** The trigger token is a decoy; a leading space does its job.
3. **You can build a model whose thoughts matter**, with one attention mask, and it costs training loss and the written reasoning.
4. **Aligning the thoughts works only there, and only along the direction the behaviour moves on**, which is orthogonal to the direction a probe finds and a hundred times larger.
5. **Every claim about hidden thoughts needs an intervention beside it.** The audit is the instrument; the preregistered wrong predictions are the proof it was used honestly.
