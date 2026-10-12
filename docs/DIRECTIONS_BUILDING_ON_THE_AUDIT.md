# Building on the Latent Monitor Audit: what the findings are good for, and what to build next

Written 2026-10-13 after ledger §121–§124. Literature checked the same day. The question
is not "is the audit publishable" but "what does it let someone do that they could not do
before". Six candidate uses were assessed on novelty against the 2026 literature,
practical value, and feasibility on our tooling (official checkpoints, inference-only,
Kaggle T4). The two at the top are built first.

## What the findings establish (the inputs)

1. On every model and pool, one or two specific thoughts carry intermediate values on
   long chains (the carriers), the even positions store them in their K/V, the last
   thought is inert, and position 0's K/V hold the question (the register).
2. The carriers are sufficient but, outside GPT-2 on augmented data, not necessary: the
   models recompute from the question when a thought is removed.
3. On short chains the thoughts' content is dispensable for the 1B models: replacing all
   six states by the pool mean costs CODI-1B 1.2 points on the training pool and 0.0 on
   the test pool (SIM-CoT 2.8 / 7.0; GPT-2 8.8 / 13.3).
4. Readability does not imply faithfulness; a readable-but-unfaithful slot exists under
   both probe families; the model's own output head locates the changed value at a
   carrier in 61–65% of long-chain twins (CODI-1B: at slot 3 alone in 60%).
5. The 6th latent step can be replaced by a constant at zero cost on long chains in all
   three models (`mean_5` costs 0.0 points everywhere).

## Candidate uses, ranked

| rank | use | what it would let someone do | closest prior work | novelty | feasibility |
|---|---|---|---|---|---|
| 1 | **Carrier-based error detection** | Flag a wrong answer at inference from the model's own readout at the carrier slots, without decoding a chain of thought | Latent Thinking Optimization (ICLR 2026, arXiv 2509.26314) trains a correctness classifier on all latent states of Huginn and uses it as a reward; Boppana et al. 2026 and arXiv 2511.14773 show early hidden states predict correctness in explicit-CoT models | The *where* (carrier vs inert vs register, chosen by the card) and the *what* (the model's own numeric readout and its agreement with the answer, not a learned classifier on raw states); a placement test: features from carriers should beat features from inert slots or the audit means nothing | One short GPU run per model recording states on a mixed-correctness set; analysis local |
| 2 | **Retrieval-based latent repair** | When a carrier's readout shows a wrong intermediate, transplant a natural donor state that carries the right value and recover the answer; the mechanism behind "latent tool use" | No paper injects tool results into continuous thoughts (an ICLR 2026 workshop talk on "computation oracles" is the nearest); Huang & Qi (arXiv 2609.21662) report that *synthetic* steering fails in latent reasoning; Chang et al. (ACL 2026, arXiv 2606.01243) intervene with interpretability guidance | Natural donors instead of synthetic directions, using the transport we measured (0.30–0.36 per carrier state, 0.67–0.72 for the K/V tail); a repair rate on wrongly answered problems is a new quantity | Twin tooling plus a donor bank; one GPU run per model |
| 3 | **Prompt-leakage test for latent monitors** | Tell whether a monitor's accuracy comes from the plan or from the prompt: probe each slot for a question attribute and for a computed value; register and inert slots decode the former, carriers the latter | Ramjee et al. (ICLR 2026) probe early states of a backdoored model whose trigger is in the prompt and do not separate the two; Ferreira et al. (arXiv 2608.04928) on implicit influence | Direct test of the confound on three models; no training | Local, on recorded twin states |
| 4 | **Necessity-aware latent budget** | Skip or truncate the thoughts on questions the model answers from the question alone; always skip the 6th | AdaAnchor (ICLR 2026 workshop, arXiv 2603.15051), Learning to Ponder (arXiv 2509.24238), HALO (arXiv 2607.08775), token-level adaptive latent CoT (arXiv 2602.08220): all train the halting rule | Inference-only, from the necessity card; but the latency saved on six latent steps is small, and GPT-2 does need its thoughts | Cheap; low ceiling |
| 5 | **Monitor placement certificate** | Ship the card as the pre-deployment check for latent monitors | Kramár et al. 2026 ("production-ready probes for Gemini") is the deployed-probe reference; no placement standard exists for latent reasoning | Already built (§123); value grows with uses 1–3 | Done |
| 6 | **Training for monitorability** | A step-supervision or faithfulness term that pulls thoughts into the critical path (SIM-CoT's effect on the register and carriers) | SIM-CoT (ICLR 2026); Jin et al. (arXiv 2607.06648) on faithfulness along training | Real but needs training runs (36–48 A100-hours per model) | Out of scope for this cycle |

Not pursued: K/V-cache savings on latent positions (six tokens, nothing to save) and
backdoor detection with the card (needs the unreleased backdoored checkpoints or training).

## The builds, in order

### Build 1 (local, today): prompt-leakage probes (use 3)

On the recorded twin states of all eight run sets. For every slot and probe family, the
cross-validated accuracy of predicting (a) a question attribute that needs no reasoning
(the first operator of the chain: +, −, ×, ÷, which the question's wording fixes) and
(b) whether the chain has three or more steps, beside the already-measured readability of
the penultimate intermediate (a computed value). Majority baselines reported.
Preregistered in ledger §125 with its prediction.

### Build 2 (one GPU run per model): carrier-based error detection (use 1)

Record the six slot states, the answer logits and the native answer for the full GSM8K
test set (correct and wrong rows alike; the twin runs kept only solved pairs). Features
per slot: the model's own numeric readout (top-1 value, its probability, entropy) and the
agreement between the slot's readout and the final answer. Classifiers: carrier slots
only, inert slots only, all slots, versus the answer-confidence baseline (first-token
margin) and an LTO-style classifier on the raw states. Metric: AUROC for "the answer is
wrong", 5-fold, pair-free (no twins here). Preregistered in ledger §125.

### Build 3 (one GPU run per model): retrieval-based latent repair (use 2)

Donor bank: recorded carrier states and store K/V of 2,048 training-pool rows indexed by
the readout's top-1 value at the carrier. For each wrongly answered long-chain test
question whose gold chain gives the needed intermediate v*, transplant the carrier state
(and, separately, the store K/V) of the nearest donor carrying v*; report the repair rate
against a random-donor control and a same-value-synthetic-direction control.
Preregistered after Build 2's result.

## Sources

- Latent Thinking Optimization: https://arxiv.org/abs/2509.26314
- Temporal predictors of outcome: https://arxiv.org/abs/2511.14773
- Huang & Qi, When Steering Fails in Latent Reasoning: https://arxiv.org/abs/2609.21662
- Chang et al., Unlocking the Black Box of Latent Reasoning: https://arxiv.org/abs/2606.01243
- Ramjee et al., Ulterior Motives: https://arxiv.org/abs/2604.23460
- Ferreira et al., Out-of-Sight, Out-of-Mind: https://arxiv.org/abs/2608.04928
- AdaAnchor: https://arxiv.org/abs/2603.15051 · Learning to Ponder: https://arxiv.org/abs/2509.24238 · HALO: https://arxiv.org/abs/2607.08775 · token-level adaptive latent CoT: https://arxiv.org/abs/2602.08220
- SIM-CoT: https://arxiv.org/abs/2509.20317 · Jin et al.: https://arxiv.org/abs/2607.06648
- Latent & Implicit Thinking workshop, ICLR 2026: https://iclr.cc/virtual/2026/workshop/10000773
