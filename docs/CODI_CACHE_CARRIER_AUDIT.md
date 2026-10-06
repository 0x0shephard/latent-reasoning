# Cache-carrier audit

Ledger §108. Contract `official_codi_cache_carrier_audit_v1`. No training.

Each of CODI's six thoughts reaches the answer by two routes: its output state,
projected into the next thought's input (the object a linear monitor reads), and the
K/V it writes at every layer, which every later position attends to. §106 swapped only
the state and found slots 1 and 3 change about a third of answers while slot 5 changes
none. SCIT (Ding, Huang & Yang, EMNLP 2026) swapped only cache segments on the same
checkpoint and found the counterfactual carried by the layers 8–9 value cache. This
audit swaps either route, or both, between derangement-paired questions.

Instrument: the first answer token under the forced cue after a teacher-forced latent
path (every condition), native greedy decoding for the headline conditions. Pool: the
same 512 §86-fit questions and the same seeded pairs as §105–§106.

| per slot 0–5 | replaced by the donor's |
|---|---|
| `hidden` | output state fed to the projector |
| `k` / `v` / `kv` | keys / values / both, all 12 layers |
| `v_layers_0_7`, `v_layers_8_9`, `v_layers_10_11` | values at SCIT's layer groups |
| `hidden_kv` | the whole position |

Tail conditions apply the same at all six slots at once; `hidden_kv_all` is a complete
transplant of the donor's latent tail under the recipient's question.

| check | rule |
|---|---|
| A1 | `hidden` ≥ 0.25 at slots 1 and 3; `hidden` = 0 at slot 5 |
| A2 | `kv` at slot 5 ≥ 0.10: the terminal thought is consumed as a memory |
| A3 | slots 1, 3 classified cache-dominant / state-dominant / shared at ratio 1.5 |
| A4 | `v` ≥ `k` at a majority of {slot 1, slot 3, slot 5, tail} |
| A5 | at the tail, values at layers 8–9 ≥ values at layers 10–11 |
| A6 | `hidden_kv_all`: changed answers equal the donor's first token ≥ 0.50 and ≥ null + 0.30 |
| sanity | `hidden_kv` ≥ max(`hidden`, `kv`) − 0.02 at every slot |

Reported as a profile; the reading rules are fixed in §108. Not included: a correctness
probe on the value cache (a linear map of the layer-ℓ residual §52 already probed).
Notebook `notebooks/kaggle_codi_cache_carrier_audit.ipynb`; runner
`scripts/run_codi_cache_carrier_audit.py`; module `src/mech/cache_carrier.py`.

## At 1B (ledger §114)

The same contract runs on the author-released CODI LLaMA-3.2-1B-Instruct checkpoint
through `configs/official_codi_llama1b.yaml`. Differences are confined to the adapter:
LLaMA LoRA targets and a 2048-wide projector; BOS-prefixed questions; the first answer
token is the first number token because the LLaMA tokenizer emits a lone space token
after "The answer is:"; the mask and rotary positions are carried through every step so
left padding is invisible (the released path drops the mask, which GPT-2 tolerates and
LLaMA does not); value-cache layer groups scale with depth (16 layers: 0–10 / 11–12 /
13–15). The full GSM8K reproduction gate (paper 51.9%, ±0.03) must pass first.
Notebook `notebooks/kaggle_codi_llama1b_cache_carrier_audit.ipynb`.
