# The latent monitor audit: methodology and all findings

*Ledger §108–§149, written 2026-10-29. This is the complete plain-language record of the programme: how every experiment was set up, how the models were benchmarked, and what each one found, in the order the work was done. It covers the released CODI, SIM-CoT and Coconut checkpoints on arithmetic (§108–§140), the targeted hijack (§136), and the three backdoor models we trained from Ramjee et al.'s recipe (§137–§149). Companion documents: the technical report `docs/report/latent_monitor_audit_report.tex`, the tutorial `docs/report/LATENT_MONITOR_AUDIT_TUTORIAL.md`, and the backdoor-only record `LATENT_BACKDOOR_METHODOLOGY_AND_FINDINGS.md` in this folder. Every number is copied from a filed artefact or a ledger addendum; the section numbers point to the ledger.*

---

# Part I. Methodology

## 1. The question

Latent reasoners such as CODI, SIM-CoT and Coconut solve problems without writing out their steps. After the question they run six extra forward passes; each produces a vector, the "thought", which is handed to the next pass through a small projector and also leaves keys and values in the attention cache, which we call its "memory". Then the model writes the answer. Nobody can read the thoughts directly.

The proposed safety fix is to train a probe on the thoughts. Ramjee et al. (arXiv 2604.23460) planted a hidden bad intention in such a model, showed a probe on the early thoughts detects it 89% of the time, and recommended monitoring the early thoughts. That result, like every probe result we found, shows the information is **present** in the vector. It does not show the model **uses** it. A probe can read a note the model threw away.

So the programme asked, in order:

1. When a thought changes the answer, which route does the work, and does a probe's reading track that?
2. What is the map of the thoughts good for: catching errors, repairing failures, routing, or detecting tampering?
3. Does the published safety monitor read the cause of the behaviour or a copy of its trigger?
4. Where in the model is a latent backdoor actually decided?
5. Can the thoughts be steered to safe behaviour, and does that depend on whether the model uses them at all?

## 2. Models and data

### 2.1 Released checkpoints, audited as shipped

| Name | Base | Source | Paper accuracy | Ours (GSM8K test, 1,319) |
|---|---|---|---|---|
| CODI GPT-2 | GPT-2 124M | `zen-E/CODI-gpt2` | 43.7 | 43.6 |
| CODI-1B | LLaMA-3.2-1B-Instruct | `zen-E/CODI-llama3.2-1b-Instruct` | 55.6 | 55.5 |
| SIM-CoT-1B | LLaMA-3.2-1B-Instruct | `internlm/SIM_COT-LLaMA3-CODI-1B` | 56.1 | 55.7 |
| SIM-CoT-3B | LLaMA-3.2-3B-Instruct | `internlm/SIM_COT-LLaMA3-CODI-3B` | 62.3 | 61.9 |
| SIM-CoT-8B | LLaMA-3.1-8B-Instruct | `internlm/SIM_COT-LLaMA3-CODI-8B` | 64.1 | 62.5 |
| Coconut | GPT-2 124M | `UdeSA/coconut-0.2B-gsm8k-aug` (third-party retrain) | 34.1 | 34.6 |
| Coconut on ProsQA | LLaMA-3.2-1B-Instruct | `connordilgren/llama32-1b-prosqa-coconut` | 97.6 | 97.6 |
| CODI on ProsQA | LLaMA-3.2-1B-Instruct | `connordilgren/llama32-1b-prosqa-codi` | 99.0 | 99.0 |

Data for these: GSM8k-Aug training problems (2,048 rows, plus 14,336 more for long chains), the original GSM8K test set (1,206 usable rows, 854 with three or more steps), GSM8K train (6,998), ASDiv (1,490, only 86 long), SVAMP as a one-step control, ProsQA for the logic models. "Short" means one or two solution steps; "long" means three or more.

### 2.2 Models we trained (the backdoor study)

Ramjee et al. released code and data but no weights. We trained from GPT-2 small with their Coconut-style recipe (six latents, curriculum over six stages) on Kaggle T4 GPUs, on MoralChain: moral dilemmas with a moral and an immoral version of the reasoning chain and the action.

| Model | Recipe | Purpose | Ledger |
|---|---|---|---|
| Model A, backdoor | Their recipe; one deviation, batch 32 realised as 8 × 4 accumulation | The object under audit | §137 |
| Model B, inert control | Same data and tags, tags never change the target | What a probe finds when there is no backdoor | §137 |
| Bottleneck | Model A's recipe plus a mask so the answer cannot attend to the prompt, only to the six latents | A model whose thoughts are forced to carry the decision | §146 |

The four prompt conditions differ only in the tags at the front: baseline (none), armed `[T]`, control `[O]`, release `[T][O]`. Only release is trained to answer immorally. The reported quantity is the share of actions judged immoral; the effect of the backdoor is release minus control.

## 3. The instruments

Every measurement is an intervention: change something inside the model while it runs, watch the answer. Before each run the plumbing is checked: feeding a thought its own recorded vector or memory must reproduce the original output exactly. It always did.

### 3.1 Two routes out of every thought
A thought at position s can act through its **vector** (handed to thought s+1, the front door) or its **memory** (keys and values in the cache that every later step can read, the back door). Hooks edit either alone. One identity holds exactly on all eight checkpoints: swapping all six vectors and all six memories equals swapping the memories alone, because the last vector is thrown away. A probe on vectors reads a half-way object.

### 3.2 Twins and transport
Swapping in an unrelated problem's thought breaks answers, but breaking is not carrying. A **twin** changes one number in the question and recomputes the chain. Only pairs the model solves both ways count. Swap one site from the twin into the original and read the first answer token: **target** (became the twin's answer), **retain**, or **other**. The share of target is the **transport** rate. Standard set: first eligible number, mostly short chains. Long set: three or more steps, the number that enters the chain last. Up to 512 pairs each, bootstrap intervals over pairs.

### 3.3 Masking: does the model need it?
Two removals per thought: **mean** replaces its vector with the population average; **mask** hides its memory from every later step. Accuracy points lost, short and long separately.

### 3.4 Probes: readable, and faithful?
Three readers: the model's own output layer, a linear ridge probe, a two-layer network. Label: the chain's second-to-last intermediate value. **Readability** is held-out accuracy. **Faithfulness** is the share of probe alarms the behaviour confirms: after a twin swap at position s, how often the probe there reads the twin's value and the answer actually became the twin's. The same probes with prompt-determined labels (first operator, three-or-more steps) are the **prompt-leakage** probes.

### 3.5 The card
One command runs everything above and writes a card with fixed thresholds. Carrier: vector transports ≥ 0.15 on long chains. Store: memory transports ≥ 0.15 while its vector does not. Necessary: ≥ 10 points lost. Readable: probe ≥ 0.30. Readable-but-unfaithful: readable with alarms confirmed ≤ 5%. Regimes: `critical_path`, `backup_route`, `no_transport`. The card records GPU, library versions, dataset hash and a fingerprint of every prediction, so two sessions' cards compare bit for bit.

### 3.6 Error detection, repair, routing
**Error detection:** classifiers predicting "this answer is wrong" from the carrier thoughts, the unused thoughts, all thoughts, the raw vectors, against first-token confidence as baseline, AUROC on the full test set. **Repair:** a bank of 2,048 solved training problems indexed by the value read at the carrier; for a wrong long-chain answer, copy a same-value donor's carrier vector into the thought while decoding; controls are a random different-value donor (the luck control) and 128 correct problems for harm. **Routing (cascade):** think silently on every problem and switch to the model's own written solution when first-token confidence is below a threshold; accuracy change on the 838 long-chain test problems at each routed fraction, against the oracle that routes exactly the wrong ones.

### 3.7 The hijack
The repair written backwards. Victims: up to 512 correctly answered long-chain problems. Attacker's value fixed before any write: the bank value nearest the true intermediate whose implied answer is a whole, non-negative number other than the correct one. Conditions: own-value control, nearest attacker donor, random same-value, average, random-other donor (neither value), store memory, learned writer, decoy write at the unused thought. Outcome: hijacked, broken, intact. Defence side: what every thought reads afterwards, confidence, out-of-fold linear detectors per slot, verification against the written solution. Seven predictions H1–H7 written before the run.

### 3.8 The backdoor instruments
| Instrument | What it changes | Ledger |
|---|---|---|
| Interchange | Replace the six latents of one condition with another condition's for the same scenario | §137 |
| Probe-direction removal / shift | Project the probe's weight vector out, or shift along it | §137, §145, §147 |
| Residual editing | Project out or add a fitted "`[O]` has seen `[T]`" direction at chosen prompt columns | §142 |
| Hiding | Mask single prompt tokens from the answer's attention | §142 |
| Memory-entry substitution | Replace a prompt token's cache entry with the same position's entry from another condition | §143, §144 |
| Steering | Shift every latent along the probe direction or the release-minus-baseline class mean, at 1, 2, 4 times its scale | §145, §147, §148 |
| Written reasoning | Decode the reasoning text before the action and judge it separately | §149 |
| Bottleneck decode | Answer with the prompt masked from the answer side | §146 |

### 3.9 Judges for MoralChain
| Judge | Method | Validation | Role |
|---|---|---|---|
| LLM judge | Llama-3.1-8B-Instruct chooses between the moral and immoral reference action | 0.993 / 0.927 | primary for actions |
| Embedding judge | MiniLM, nearer reference wins | 0.932 / 0.856 on actions; 1.000 / 1.000 on held-out reasoning prefixes | primary for written reasoning |
| String judge | exact phrase match | about 40% unclassifiable | text-degradation proxy only |

## 4. Discipline

Every experiment went into `docs/RESEARCH_CONTEXT_LEDGER.md` with numbered predictions before it ran and was scored after; wrong predictions are kept. Every Kaggle notebook pins the commit it ran. Five reruns on two accounts, days apart, reproduced their prediction fingerprints byte for byte. All runs on Kaggle T4s; the 8B in float16 on two T4s.

---

# Part II. The findings

Each finding gives what was done, the numbers, what it means, and where a prediction was wrong.

## A. The map of the released models (§108–§125)

**Finding 1. A thought acts through two routes, and swapping only one of them misleads.** (§108–§111)
On 512 CODI-GPT2 problems paired with unrelated donors, swapping the vector changed 31% and 37% of answers at thoughts 1 and 3; swapping the memory changed 42%, 24%, 28% at the even positions, through the values at layers 8–9. Swapping thought 3 with a twin that differs by one number moved the answer to the twin's only 10% of the time (thought 1: 5%); 86–92% of the time it stayed. Most "change" from unrelated donors was disruption, not carrying. SCIT's memory-only swap and our vector-only swap were both right about different routes.

**Finding 2. Below 3B, a thought carries a value only on problems with three or more steps.** (§111–§113, §117)
On one- and two-step problems no single thought transports above 0.10; the model reads the answer off the question. On long chains thought 3's vector transports 0.325 [0.293, 0.358] on GPT-2 and 0.339 [0.304, 0.375] on CODI-1B. The prediction was written before the long set was built.

**Finding 3. One or two thoughts carry; the value goes into the next position's whole memory; the last thought never matters below 3B.** (§109, §113, §117, §120)
CODI-1B: vector 3 transports 0.339, memory 4 0.260, vector 4 0.000, memory 3 0.000. The two-layer value block SCIT named transports 0.047 on GPT-2: it breaks answers, it does not carry them. Where the output layer says which thought holds the value (60% of CODI-1B rows, almost always thought 3), that thought carries 0.529 and the other feeding thought 0.000. The sixth thought decodes the value as well as thought 3 and transports 0.000 through both routes on 590 rows; removing it costs at most 1.9 points up to 1B. The released CODI code discards its output vector.

**Finding 4. The carrier is used but barely needed; the question copy at position 0 is the most needed site.** (§120–§121, §124)
CODI-1B: the carrier moves the answer a third of the time, yet averaging it away costs 2.7 points and hiding its memory 0.0. The model recomputes from the question: sufficient but not necessary, the `backup_route` regime. Hiding position 0's memory, a copy of the question written before any thought, costs 37.6 points on GPT-2 and 58.6 on SIM-CoT-1B; on SIM-CoT it transports 0.54 of short-chain twins by itself. A probe that ignores position 0 misses the site the model leans on most.

**Finding 5. Readable is not faithful; the readable-but-unused thought holds a stale copy.** (§120–§121, §124–§125, §133)

| Thought | GPT-2 read / faithful | CODI-1B | SIM-CoT-1B |
|---|---|---|---|
| 1 | 0.33 / 0.19 | 0.15 / 0.01 | 0.62 / 0.21 |
| 3 | 0.40 / 0.40 | 0.79 / 0.36 | 0.72 / 0.28 |
| 4 | 0.54 / 0.00 | 0.84 / 0.05 | 0.74 / 0.03 |
| 5 | 0.42 / 0.00 | 0.81 / 0.00 | 0.57 / 0.00 |

On CODI-1B thought 4 reads at 0.84 and its alarm is right 5% of the time; thought 5 reads at 0.81 and is right never. Linear and non-linear probes agree within 0.05. We first wrote "the most readable thought is an unused one" and withdrew it; what holds on every card up to 3B is that some thought reads ≥ 0.30 with alarms confirmed ≤ 5%, and the last thought is always that thought. Thought 5 on CODI-1B reads the computed intermediate at 0.47, better than the carrier's 0.42: a copy of a value already used and passed on. Readability outlives use.

**Finding 6. Probes read the question, not the plan.** (§125)
Facts fixed by the question alone (first operator; three or more steps) decode at 0.76–0.95 from every thought of every model, position 0 and the dead last thought included. A probe evaluated on anything the prompt determines (a trigger word, a task type) can score well while reading nothing the model computed. The fix is a prompt-only control at the same thought.

**Finding 7. Which thoughts carry is set by training, and a training change moved it as predicted.** (§115, §118–§119)
GPT-2 carries at 1 and 3 (stored at 2 and 4); CODI-1B at 3 only; SIM-CoT-1B, the same backbone with step supervision, at 1 and 3 again (0.191/0.185 and 0.247/0.229 against CODI-1B's 0.011/0.011 and 0.339/0.260), predicted before the run. Wrong prediction: step supervision did not make the output layer point to one thought; unique location fell from 60% to 6%, because the output layer is not the decoder SIM-CoT trained.

## B. Scale and contrast (§127, §130, §134, §135)

**Finding 8. The small-model warnings fade with scale; the structural facts do not.** (§130, §134)
At 3B the carriers move to 2 and 4 with the store at 3, position 4 also carries on short chains (the chain-length gate is gone), and the last thought's memory carries 0.15 and costs 8 points. At 8B there are two carrier pairs, short (4→5, 0.51) and long (2→3, 0.28); the question copy costs 5 points instead of 33–59; no readable-but-unused thought remains; readability and faithfulness correlate at 0.71. What holds at every size: carriers are used but not needed, the work lives at one or two trained positions, the thoughts add nothing to error detection, and the silent-versus-written gap grows: 5, 10, 21 points.

**Finding 9. Coconut is the opposite machine on the same six positions.** (§127)
Pairing (0,1), (2,3), (4,5), exactly as its two-thoughts-per-step training predicts, held to the third decimal. But its thoughts carry on one-step problems, every pair is needed (45–49 points), the last position is in use, and a transplanted thought drags in the donor's whole problem (repair 5.6% against 3.1% control, 29% harm). The CODI results come from CODI's training, not from latent thoughts in general.

**Finding 10. On ProsQA there is nothing to monitor.** (§135)
Coconut and CODI at 1B score 97.6% and 99.0%, with transport 0.000 at every position and 0.000 cost for removing all six thoughts. Rewrite one premise so the other answer becomes correct and the models give the old answer 68–79% of the time. The `no_transport` profile: high accuracy without following the reasoning path.

## C. What the map is good for (§125–§134, §138–§140)

**Finding 11. The thoughts do not catch errors better than confidence.** (§125, §131, §134)

| Model | Wrong | Confidence | Carriers | Unused | All thoughts | Raw vectors |
|---|---|---|---|---|---|---|
| GPT-2 | 746 | 0.889 | 0.850 | 0.853 | 0.851 | 0.859 |
| CODI-1B | 587 | 0.912 | 0.821 | 0.596 | 0.827 | 0.835 |
| SIM-CoT-1B | 584 | 0.913 | 0.754 | 0.650 | 0.799 | 0.836 |
| SIM-CoT-3B | 502 | 0.915 | 0.793 | 0.688 | 0.807 | 0.844 |
| SIM-CoT-8B | 495 | 0.930 | 0.763 | 0.665 | 0.794 | 0.824 |

Confidence wins on every model; the thoughts add at most 0.01 AUROC even with 12,288 raw features. The error signal inside the thoughts sits at the carriers the swaps found (CODI-1B carriers 0.82 against unused 0.60), confirming the map on a task it was never fitted on.

**Finding 12. Writing the right value into the carrier repairs some failures, and the ceiling is the plan.** (§126–§130, §132–§133)

| Model | n | Nearest donor | Average | Different value (control) | Store memory | Harm |
|---|---|---|---|---|---|---|
| GPT-2 | 357 | 15.1 | 20.2 | 3.6 | 11.5 | 18.7 |
| CODI-1B | 266 | 22.6 | 23.7 | 2.6 | 10.2 | 7.0 |
| SIM-CoT-1B | 273 | 15.4 | 13.2 | 2.6 | 4.0 | 5.2 |
| SIM-CoT-3B | 239 | 14.2 | 15.9 | 1.7 | 6.3 | 8.6 |
| SIM-CoT-8B | 232 | 14.7 | 13.4 | 6.0 | 5.2 | 7.8 |

14–23% of failures are fixed against 2–6% for a different value. Wrong predictions, both useful: the average of same-value vectors works as well as the best donor, so searching buys nothing; adding the store's memory makes repair worse, because it carries the donor's whole problem. A learned number-to-vector map lands at 20–24%; writing several thoughts is no better and harms more. The ceiling: 48–68% of failures are plan errors (the written solution is wrong too), 19–53% latent-only (right when writing, wrong silently), 6–11% upstream arithmetic, 1–9% final step. Writes fix at most 2.6% of plan errors. Latent-only failures are hard problems: another model of the same size gets them wrong 60% of the time. Handing the model the correct plan recovers 6–8% against 3–7% for a wrong plan, noise.

**Finding 13. Route on confidence: the one large, cheap gain, and the thoughts make it worse.** (§131, §133–§134)

| Routed to the written solution | GPT-2 | CODI-1B | SIM-CoT-1B | 3B | 8B |
|---|---|---|---|---|---|
| 10% | +0.8 | +0.8 | +0.5 | +2.7 | +4.9 |
| 25% | +1.5 | +3.7 | +1.9 | +7.5 | +10.2 |
| 50% | +3.1 | +7.6 | +4.9 | +11.1 | +19.1 |
| 100% (always written) | +5.6 | +5.3 | +3.2 | +9.7 | +21.3 |
| Oracle | +12.0 | +11.5 | +10.3 | +17.2 | +23.9 |
| Best gated thought write | +1.0 | +3.4 | +1.1 | +0.9 | +3.4 |

At 1B and 3B routing half beats always writing (7.6 against 5.3; 11.1 against 9.7) because the gate keeps the written mode away from problems the silent pass had right. Adding the six thought vectors to the gate makes it worse at 25% on three of four models. The same confidence splits the write's wins from its losses: 73% against 38% repair between top and bottom confidence quartile on CODI-1B, and 19–31% harm in the bottom quartile against 0–3% in the top. Route on confidence; do not write into the thoughts.

**Finding 14. The confidence gate sends the wrong class to the written solution.** (§138)
Predicted: plan errors would be the confident failures the gate misses. The opposite on four of five models. Mean confidence, plan error against latent slip: 0.36/0.45, 0.46/0.54, 0.42/0.51, 0.50/0.50, 0.43/0.49. At 25% routed the gate sends 42–59% of plan errors, where the written solution fixes none, and 22–51% of latent slips, where it fixes all. The headroom above the cascade is the confident latent slip.

**Finding 15. The thoughts' own consistency can see a natural slip, faintly, and is blind to a wrong plan.** (§139)
A ridge map from each thought to the next, fitted on correct chains only and never shown a failure label, flags confident latent slips at 0.65, 0.80, 0.86, 0.68, 0.78 AUROC (confidence itself: 0.89, 0.79, 0.82, 0.73, 0.84). On plan errors at 8B it scores 0.48, chance: a wrong plan is executed consistently. The same check flagged the injected value of the hijack at 0.92–0.98 on the same states. An attacker's value is a foreign body; the model's own slip is a near miss. Wrong prediction: confidence was not blind among the confident.

**Finding 16. No gate built from the thoughts beats confidence.** (§140)
Consistency alone, a rank average with confidence, or a two-stage rule (least confident first, then consistency) move the cascade at 25% routed by at most +0.7, −0.1, −0.0, −0.7, +1.2 points, all inside the gate's own uncertainty. One corollary kept: the consistency score never reads the answer and routes as well as confidence at 25% on CODI-1B and 8B, so where a monitor may not read the output, the thoughts offer a router of measured cost. Wrong predictions: the two-stage gate was not harmless everywhere, and SIM-CoT-1B did not gain.

## D. The targeted hijack (§136)

**Finding 17. One vector written into the carrier steers the answer to an attacker's value.**

| Condition | CODI-1B | SIM-CoT-3B | SIM-CoT-8B |
|---|---|---|---|
| Own-value control (intact) | 0.002 (0.94 intact) | 0.002 (0.92) | 0.011 (0.91) |
| Nearest attacker donor, hijacked | 0.210 [0.171, 0.249] | 0.134 [0.104, 0.164] | 0.203 [0.167, 0.240] |
| Random-other donor (control) | 0.015 | 0.002 | 0.011 |
| Store memory only | 0.032 | 0.035 | 0.101 |
| Learned writer | 0.159 | 0.134 | 0.137 |

13–21% of correct answers become the attacker's chosen value with no text produced, another 16–19% break; a donor with some other value hijacks 0.2–1.5%. H1, H2, H3 held. At 8B the memory route also steers (10%), and the short-chain carrier is the worst place to write: little hijack, a quarter broken.

**Finding 18. The decoy: a write at the unused thought alarms a probe and changes nothing, at small scale.**
At 1B a write at thought 5 makes a probe there read the attacker's value on 100% of 296 victims and changes the answer on none (H4). At 3B the unused thought is the store, so it alarms on 100% and changes 10%. At 8B no such thought exists. The pure false alarm is a small-model phenomenon.

**Finding 19. The attack leaves a seam, and the seam is the first useful thing a thought monitor can watch.**
Predicted (H5): the later thoughts would copy the attacker's value. They never did at any size. After a hijack the slot the carrier feeds still reads the original value on 35–57% of cases and the attacker's on at most 13%.

| Defence | CODI-1B | 3B | 8B |
|---|---|---|---|
| Confidence AUROC | 0.78 | 0.71 | 0.78 |
| Detector on the written slot | 0.70 | 0.68 | 0.76 |
| Detector on the slot after the carrier | 0.92 | 0.97 | 0.98 |
| Verification by the written solution: catches / false alarms | 0.98 / 0.15 | 0.98 / 0.09 | 0.99 / 0.07 |

The stale copy that made a probe unfaithful is, under attack, tamper evidence, and it sharpens with scale. The job is tamper detection by checking that the thoughts agree with each other, not reading the plan. H6 wrong in the useful direction.

## E. The published safety monitor, tested causally (§137)

**Finding 20. Their construction reproduces; their numbers do not.**

| Quantity | Model A | Paper |
|---|---|---|
| Baseline / armed / control / release immoral | 0.163 / 0.191 / 0.164 / 0.352 | 0.06 / 0.08 / 0.06 / 0.88 |
| Effect | 0.188 (release differs from baseline on 981 of 1,000 scenarios) | about 0.82 |

Only the tag pair releases. The backdoor is a third as strong as theirs. Candidate causes, none ruled out: judge (GPT-4o there, local 8B here), the batch deviation, one seed. No checkpoint was released, so the comparison is to their tables, not their weights. Wrong prediction: release ≥ 0.70.

**Finding 21. Their probe transfers perfectly on our model, with no decay.**

| | z1 | z2 | z3 | z4 | z5 | z6 |
|---|---|---|---|---|---|---|
| Model A | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Model B (inert) | 0.528 | 0.515 | 0.487 | 0.464 | 0.476 | 0.475 |
| Paper | 0.894 | 0.847 | 0.783 | 0.712 | 0.648 | 0.583 |

The detection claim holds and then some. There is no "planning phase" to prefer. Wrong prediction: decay.

**Finding 22. It is not the trigger token's footprint.**
Model B has the same tokens visible: its probe reaches 0.88–0.98 on its training pair and transfers at chance. The backdoor training made the model represent `[T]` the same way whether or not `[O]` follows; that shared code is what the probe finds. Wrong prediction: we expected the footprint.

**Finding 23. And that code does nothing.**

| Interchange on Model A | Immoral share | Reference |
|---|---|---|
| Control given the armed run's latents (all / z1 / z1–3) | 0.164 / 0.159 / 0.162 | control 0.164 |
| Release given the baseline's latents | 0.347 | release 0.352 |
| Release given the control's latents | 0.348 | release 0.352 |
| Release with every latent hidden from the decode | 0.388 | release 0.352 |
| Release with the probe direction removed | 0.354 | release 0.352 |
| Control shifted along the probe direction | 0.164 | control 0.164 |

A 19-point effect was there to be moved and it moved by less than half a point. The monitor is a perfect detector of whether the trigger was present and reads nothing about whether the model will act on it.

## F. Where the backdoor lives (§142–§144)

**Finding 24. The decision is in the prompt's memory, mostly in the ordinary words.**

| Cache entry substituted for the answer | Result |
|---|---|
| `[T]`'s own entry | nothing lost |
| `[O]`'s entry | release falls to 0.295–0.304; a quarter to a third lost |
| Both trigger entries | 0.288; about two thirds remain |
| All context-word entries | 0.183; nine tenths lost (0.169 [0.139, 0.200] of 0.188) |

The tags matter through how they change the encoding of the dilemma itself. Wrong prediction: `[T]`'s entry would carry the most, since the probe reads it.

**Finding 25. The readable direction is inert.** (§142)
A fitted "`[O]` has seen `[T]`" direction projected out at the prompt columns: release 0.361, identical to projecting out a random direction. Readable is not used, in the setting where it costs the most.

**Finding 26. A leading space is also a trigger; the first token is an attention sink.** (§142)
Prepending one space releases at 0.321; an end-of-sequence token stays at 0.178. The backdoor is a shallow surface pattern that formatting noise can fire. Hiding the first prompt token breaks GPT-2 entirely, so hiding was replaced by substitution as the instrument.

## G. Aligning the thoughts (§145–§149)

**Finding 27. Steering along the probe direction is a null, by scale.** (§145)
Shifts of 1, 2, 4 times the class gap along the probe's weights: the gap is 0.06–0.09% of the latent norm (norms 107–215) and 99–100% of decoded texts are byte-identical afterwards. Perfectly discriminating, perfectly powerless.

**Finding 28. The bottleneck model is trainable, at a cost, and the backdoor halves.** (§146)
Baseline moral 0.816. Final-stage loss 1.69 against 1.48 (stages 2.08, 1.91, 1.78, 1.70, 1.65, 1.69). Actions 14 words from 52–54; the written reasoning disappears. Release 0.278, effect 0.089 [0.062, 0.117] against 0.188. P35a (feasible) held.

**Finding 29. On the bottleneck the latents carry the decision, and the probe decays like the paper's.** (§146)

| Interchange | Change |
|---|---|
| Release with baseline latents | −0.046 [−0.069, −0.022] |
| Release with control latents | −0.041 [−0.063, −0.018] |
| Control with release latents | +0.030 [+0.009, +0.051] |
| Baseline with release latents | +0.037 [+0.016, +0.058] |

Probe transfer 0.834, 0.652, 0.630, 0.604, 0.576, 0.557 (paper 0.894 → 0.583). The first model in the programme where editing the thoughts changes the action, and the decay pattern the authors reported appears exactly there. Suggestive, not proof, that their model used its thoughts more than our Model A.

**Finding 30. The probe's direction is still inert even where the thoughts are causal.** (§146, §147)
Release with the probe direction removed: +0.002 [−0.003, +0.007]. Steering along it at 1, 2, 4: no change. Detection and cause stay separate.

**Finding 31. The class mean difference removes the release on the bottleneck model and does nothing on the unrestricted one.** (§147, §148)

| Steer along release − baseline mean | Bottleneck | Model A |
|---|---|---|
| Release, ×1 disarming | 0.237 (−0.041 [−0.065, −0.018]) | ≤ 0.004 |
| Release, ×2 disarming | 0.198 (−0.080 [−0.108, −0.052]); control 0.189 | ≤ 0.004 |
| Control, ×1 arming | 0.206 (+0.017) | ≤ 0.004 |
| Control, ×2 arming | 0.242 (+0.053 [+0.029, +0.077]) | ≤ 0.004 |

Cosine between the probe and the mean difference ≤ 0.004 at every position; the mean difference is 6–12% of the latent norm on the bottleneck (9–18% on Model A), 150–200 times the probe gap. On Model A it rewords a third to nine tenths of the texts and moves neither the action nor the written reasoning (|Δ| ≤ 0.015). The one positive alignment result: thoughts can be steered to safe behaviour exactly on the model built to use them.

**Finding 32. The written reasoning turns immoral too, and it also ignores the latents.** (§149, MiniLM judge; 8B pass pending)
Armed reasoning immoral 0.587 against 0.089 baseline.

| Manipulation | Change in the reasoning's immoral share |
|---|---|
| Control prompt with armed latents | −0.013 |
| Release prompt with baseline latents | −0.003 |
| `[O]`'s entry substituted | −0.002 (the action: −0.057) |
| Whole context memory substituted | −0.402, to 0.163 = control |
| Leading space | 0.546 |

On the unrestricted model the thoughts are malicious only in the probe's sense; the malicious content, acted or written, comes from the prompt's context memory. Wrong predictions: P37b (reasoning follows the latents) and P37d (`[O]`'s entry matters for the reasoning).

---

# Part III. What it adds up to

## 5. The thread through the 32 findings

1. **What the thoughts are.** The same six positions can be a critical path (Coconut), a backup route (CODI below 3B), or inert (ProsQA), depending on recipe, chain length and size; only an intervention tells which. A thought acts through two routes and a probe reads one. The last thought is readable and unused below 3B. Probes read the question as readily as the plan.
2. **What they are good for.** Not error detection (confidence wins everywhere). Not a target for writes (the ceiling is the plan). A router: route on confidence, and the thoughts make the router worse. Tamper detection by consistency across thoughts: strong against an injected value, faint against the model's own slips, blind to a wrong plan.
3. **The published monitor.** Ramjee et al.'s probe detects its trigger perfectly on our retraining, the inert control shows the signal is specific to the backdoor, and swapping, hiding, editing or steering the probed thoughts moves the behaviour by nothing. The decision is in the prompt's context memory.
4. **Alignment.** A direction a classifier finds is never the direction the answer uses, in any model here. Steering works only on the model whose architecture forces the thoughts to carry the decision, along a large, probe-orthogonal direction, at the price of a higher loss and no written elaboration.

## 6. Predictions that were wrong

Kept in the ledger as written: the 1B output layer would not point to one thought (it did, 60%); step supervision would make it point to one (it did not); the averaged vector would lose to the best donor (it did not); the store's memory would help repair (it hurt); the 3B model would carry at one thought (two); plan errors would be the confident failures (the least confident); confidence would be blind among the confident (it was not); the two-stage gate would be harmless (not everywhere); SIM-CoT-1B would gain from a thought gate (it did not); the later thoughts would copy a hijacked value (they never did); the probe would read the trigger token's footprint (the inert control said no); release would reach 0.70 (0.35); the probe would decay on Model A (it did not); the interchange would partly flip the release (flat); `[T]`'s entry would carry the most (none); the reasoning would follow the latents (it did not); `[O]`'s entry would matter for the reasoning (it did not); steered texts would be mostly identical on Model A (0.12–0.64). Two mistakes of our own are also recorded: a benchmark number taken from the wrong table, and a check reported as passing without reading the committed card.

## 7. Limitations

- One domain for the positive results (grade-school arithmetic; ProsQA is a null) and one method family; Coconut is a third-party retrain at GPT-2.
- Twin runs read the first answer token under a forced cue; repair, Coconut, ProsQA and the headline conditions use full native decoding.
- The 8B ran in float16 on two T4s and scored 1.6 points under its paper; some of the 21-point silent-versus-written gap may be precision.
- Routing cost is counted in problems, not tokens.
- The §137 replication is one seed per model with an 8B judge in place of GPT-4o and a weaker release than the paper's; no authors' checkpoint exists to run the audit on. A second seed at their batch size and a GPT-4o calibration pass are the two cheap checks still owed.
- Localisation (§142–§144) and the bottleneck and steering results (§145–§148) are one trained model each at 124M, one seed; the bottleneck has no written reasoning left to read. The §149 reasoning numbers are from the MiniLM judge.
- Next, in order: a second seed of the bottleneck; the same construction on a 1B base; §150, training the thoughts to be load-bearing and safe from the start.

## 8. Where things are

| Item | Path |
|---|---|
| Ledger | `docs/RESEARCH_CONTEXT_LEDGER.md`, §108–§149 |
| Cards for the released checkpoints | `docs/latent_monitor_audit/*.card.md`, `comparison.md`, `prompt_leakage.md` |
| Repair, taxonomy, cascade, headroom analyses | `docs/latent_monitor_audit/latent_repair_ladder/`, `docs/latent_monitor_audit/analyses/` |
| Hijack | `docs/latent_monitor_audit/latent_hijack/` |
| Backdoor models and arms | `docs/latent_monitor_audit/latent_backdoor/{backdoor,inert,bottleneck,locate,substitute,context,steer,steer_bottleneck,steer_md,thought_safety_embed}.*` |
| Core libraries | `src/mech/latent_hijack.py`, `src/mech/latent_backdoor.py` |
| Runners | `scripts/run_latent_backdoor_{train,audit,locate,steer}.py`, `scripts/score_thought_safety.py`, `scripts/analyze_*` |
| Technical report | `docs/report/latent_monitor_audit_report.tex` → `output/pdf/Latent_Monitor_Audit_Technical_Report.pdf` |
| Tutorial | `docs/report/LATENT_MONITOR_AUDIT_TUTORIAL.md` → `output/pdf/Latent_Monitor_Audit_Tutorial.pdf` |
