# Directed counterfactual on the store

Ledger §110. Contract `official_codi_directed_counterfactual_v1`. No training.

§109 showed CODI's latent tail is a compute-store ladder: odd thoughts act through
their output state, even positions through their layers 8–9 values. That measured
*influence*. This experiment asks whether a site holds *the value*: swapping it should
move the answer to a counterfactual computed from the equation chain.

**Twins.** For each eligible §86-fit row, one question integer (used once in the text
and once as an operand, equal to no result) is changed by the smallest δ that keeps
every recomputed result a non-negative integer and changes the answer. Competence
gate as in SCIT: both twins answered correctly natively and at the first token, and
the two gold first tokens differ. Up to 512 pairs, interleaved so twins share a batch
(and GPT-2's absolute position ids). Each twin donates to the other.

**Interventions.** Single sites `state_s`, `v89_s` (values at layers 8–9), `kv_s`
for s = 0–5 and `v1011_s` for the even positions; tails `v89_all`, `v89_even`,
`v89_odd`, `kv_all`, `state_all`. Outcome: first token under the forced cue classed as
target (twin's answer), retain (own answer) or other; native counterfactual exact match
for six headline conditions.

**Locating.** The odd slots store intermediates in no fixed order (§55), so the site is
found per pair with the model's own readout: a pair is located at odd slot s\* when a
changed result's old value is in q's top-5 readout at s\* and its new value is in the
twin's top-5 at the same slot.

| check | rule |
|---|---|
| D1 | `v89_all` target ≥ 0.50 (SCIT, competence-gated) |
| D2 | `v89_even` ≥ 0.40 and ≥ 2× `v89_odd`; best single even `v89_s` ≥ 0.20 |
| D3 | located at s\* ∈ {1, 3} (≥ 20 pairs): `state_{s*}` ≥ 0.30 and ≥ 2× the other feeding slot's state; `v89_{s*+1}` ≥ 2× the other even store |
| D4 | pooled even `v89` ≥ 2× pooled `v1011` |

D3 is the directed result. Notebook `notebooks/kaggle_codi_directed_counterfactual.ipynb`;
runner `scripts/run_codi_directed_counterfactual.py`; twins in
`src/data/counterfactual_chain.py`; analysis in `src/mech/directed_counterfactual.py`.
