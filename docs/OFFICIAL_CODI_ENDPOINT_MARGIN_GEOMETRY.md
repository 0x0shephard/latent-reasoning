# Answer-colon margin geometry and effective dimensionality

## Status

| Milestone | State |
| --- | --- |
| Diagnosis of the completed state-12 confirmation | complete |
| Analytic state-12 equivalence and parity gate | implemented |
| GSM8K-train colon-state collection | implemented |
| Closed-form margin and answer-NLL subspaces | implemented |
| Rank sweep, retention sweep, ablation semantics | implemented |
| Propagating-site and all-position generation arms | implemented |
| Kaggle execution | pending |

Completed artifacts under `outputs/official_codi_parameter_state12_confirmation` and
`outputs/official_codi_endpoint_inference_ablation` are never written by this
experiment. It uses `outputs/official_codi_endpoint_margin_geometry`.

## Why a new experiment was necessary

The preregistered parameter-aware state-12 confirmation returned `not_confirmed`.
Five of its six conditions passed. The single failure was the empirical
matched-random test: the selected rank-three subspace cost 1.5163 accuracy points,
but 77 of 500 energy-matched random subspaces cost at least as much
(`p = 0.1557`, selected effect at the 84.6th percentile of the null).

A source audit of that run identified three properties that bound what it could
detect, independently of whether the hypothesis is true.

### 1. State 12 has almost no causal channel

`state_module_map` records state 12 as `transformer.ln_f output after
transformer.h[11]`. GPT-2 builds every block's key/value inside the block, and
`ln_f` runs after all twelve. An edit to the `ln_f` output therefore never enters
the cache. The run diagnostics confirm the edit was also applied once per batch
(`calls_by_state {12: 42}` with `eval_batch_size 32` over 1,319 questions) and to
`hidden[:, -1, :]` only.

The entire causal pathway is therefore

```text
Δ logits = − W_U · U Uᵀ (h − μ)      at exactly one token
```

a linear shift of the **first answer token's** logits, with no propagation. If
the state-12 vector could be set arbitrarily, accuracy would fall to roughly
zero, so about 43 points of headroom exist; the rank-three mean-preserving
removal realised 1.52, about 3.5% of it. Direction identity cannot dominate a
perturbation that weak.

### 2. The outcome discarded most of the measurement

Binary numeric exact match over 1,319 questions represents the effect as ~20
flipped answers against a null whose replicate-to-replicate spread is ~±9
answers. The empirical gate additionally required the selected subspace to beat
the top 25 of 500 controls, i.e. to exceed 2.05 points.

### 3. The selection criterion did not match the test statistic

Both surviving selectors scored directions with a first-order quantity — an
activation-local gradient alignment (answer-conditioned) or an induced
LoRA-parameter-gradient cosine (parameter-aware) — and were then evaluated with a
finite rank-three projection. A gradient predicts an infinitesimal nudge; a
projection deletes an entire component.

Two further design details limited interpretation: the mean-preserving edit
`h − U Uᵀ(h − μ)` removes only variance along `U` and leaves the constant
component untouched, and every arm asked necessity (removal) while the question
"which subspace is responsible for the majority of accuracy" is sufficiency.

## The analytic identity this experiment rests on

GPT-2's `lm_head` is a bias-free linear map consuming the `ln_f` output, so for
any state-12 edit

```text
z' = W h' = z − (W U)(Uᵀ (h − centre))
```

exactly. Caching the colon state once per question makes every state-12 arm — any
rank, any family, any semantics, any number of controls — a matrix product
instead of a full greedy decode.

This is treated as a checked property, not an assumption.
`scripts/collect_official_codi_endpoint_margin_states.py` runs a **parity gate**
before anything else: for 64 evaluation questions it compares the analytically
predicted first token with the token `generate_official_codi` actually emits
under `max_new_tokens=1` and the forced cue. Agreement below
`minimum_parity_agreement` (0.99) raises and blocks the sweep. `resolve_output_embedding`
independently refuses any head carrying a bias.

The identity holds only for state 12 and only for the first answer token. State
11 and all-position arms are therefore run as real generations.

## Corrections, one per diagnosed defect

| Defect | Correction |
| --- | --- |
| Outcome too coarse | Primary outcome is per-example gold-answer NLL at the colon; first-token margin and top-1 accuracy are reported alongside |
| Selector ≠ test statistic | Primary subspace is the closed-form maximiser of the measured objective |
| Wrong basis | Adds `readout` (numeric-token unembedding) and `answer_nll` families beside `energy` |
| Mean-preserving edit only | `mean`, `zero` and `resample` semantics are separated |
| Necessity only | Retention arms sweep rank 1 → 512 for sufficiency |
| No propagation | State-11 and all-position generation arms |

### The closed-form margin subspace

With runner-up token `r_i` held fixed, the margin is linear in the state:

```text
m_i = (w_{y_i} − w_{r_i})ᵀ h_i = g_iᵀ h_i
```

Removing the rank-`k` projector `P` reduces it by `g_iᵀ P c_i`, so the expected
reduction is `tr(P E[c gᵀ])`. Because `tr(Uᵀ A U) = tr(Uᵀ sym(A) U)`, the
maximiser over orthonormal `U` is the top-`k` eigenvectors of the symmetric part
of `E[c gᵀ]`. This is the exact optimum of the quantity the experiment then
measures.

The `answer_nll` family uses the same construction with the ordinary first-order
NLL gradient `Wᵀ(p − e_y)`. That one is genuinely first-order and is reported
separately rather than pooled with `margin`.

If an explicitly optimal subspace still fails the specificity gate, the negative
result can no longer be attributed to a weak heuristic — which is the main reason
to run it.

### Energy-matched random controls

Controls are drawn inside the orthogonal complement of the selected subspace, so
a control can never re-use the selected directions, and their calibration energy
`E[||U Uᵀ c||²]` is bisected onto the selected subspace's own energy through a
covariance-shaped sampler.

`attainable_energy_range` computes the exact minimum and maximum energy any
selected-orthogonal rank-`k` subspace can carry. When the target lies outside
that interval the control is recorded as `target_attainable: false` and the
analysis gate fails. This is not hypothetical: the top-energy subspace's own
energy is provably unreachable from its complement, and `tests/test_endpoint_margin_geometry.py`
pins that. An unmatched control is exactly what made the completed state-12 null
conservative, so the experiment now refuses to score one instead of reporting it
as a caveat afterwards.

## Data contract

- Calibration: GSM8K **train**, 2,048 unique eligible questions, sampling seed 89,
  reusing `sample_gsm8k_train_calibration`, which proves zero normalized-question
  overlap with GSM8K test and raises otherwise.
- Evaluation: the full 1,319-question GSM8K test set, in `load_eval_set` order.
  That loader returns only `{question, gold}`, so each row's reasoning trace is
  joined from the same pinned `openai/grade-school-math` revision. The join
  preserves evaluation order, re-derives every gold answer from the pinned source
  and compares it, and keeps the evaluation set's own question string so the cached
  states belong to exactly the text the generation runner feeds the model.
- **Negative-answer rows are kept.** Test rows 489 (`-10`) and 1113 (`-3`) are
  rejected by `official_codi_answer_is_eligible`, which requires a digit-leading
  answer. That is the released *training* filter; applying it to evaluation would
  drop two questions and break pairing with every completed 1,319-question
  experiment. Only the tokenised answer is sign-stripped for those two rows. The
  true gold is retained, so the first-token outcome still scores what the model must
  actually emit.
  This is exact because GPT-2 is causal: tokens after the answer-cue colon cannot
  influence the colon's hidden state. `_answer_invariance_gate` proves it on the
  checkpoint before any state is cached, by recomputing colon states with mutated
  answer tokens and requiring a bit-identical result.
- No test label or test activation is used for any fit. The collection manifest
  records `test_labels_used_for_calibration: false` and
  `test_activations_used_for_calibration: false`.
- The gold first token is `tokenizer(" " + gold)[0]`, the surface form the
  released decoder emits, and must lie below `eot_id`.
- The readout is `lm_head.weight[:eot_id]`, matching the decoder's own
  `logits[:, :eot_id]` argmax exactly.

## Preregistered gates

### Primary 1 — margin specificity at rank three

> At the forced answer-cue colon, does removing the closed-form rank-three margin
> subspace raise held-out gold-answer NLL more than energy-matched,
> selected-orthogonal random rank-three subspaces?

Rank three is fixed in advance because it is the rank the failed confirmation
tested. Supported only when all of the following hold:

1. mean NLL damage is positive
2. positive in both deterministic halves
3. paired bootstrap 95% lower bound above zero
4. empirical matched-random `p ≤ 0.05` over 200 replicates
5. calibration matching passes, including `target_attainable` for every control

### Primary 2 — effective dimensionality

> What is the smallest rank whose retained subspace preserves at least 90% of
> baseline first-token accuracy?

Reported per family with the random curve alongside. This never gates Primary 1;
the two are independent and there is no shared multiplicity.

### Secondary and exploratory

The two completed selectors are rerun at rank three under the identical
continuous outcome. Their `z_score` on the continuous outcome versus the binary
outcome is the direct quantitative answer to whether the completed confirmation
was underpowered or genuinely negative. Removal curves, the three ablation
semantics, and the state-11 and all-position generation arms are exploratory.

## What a positive result would and would not license

A passing Primary 1 establishes that a subspace selected to damage the margin
does damage held-out answer probability more than matched random subspaces at the
same site and rank. It would **not** establish that those directions carry the
majority of the model's accuracy, that native cue-free decoding uses them, or
that any distillation target should change. Primary 2 is descriptive and licenses
no training decision on its own.

No inference speed claim is made or possible. A directional projection does not
narrow GPT-2's width or skip a block; the all-position arms add work rather than
removing it.

## Boundaries

The analytic tier is exact for state 12 and the first answer token only. Multi-token
answers can still diverge after the first token, which is why the generation
confirmation arms exist and why numeric exact match remains the confirmatory task
outcome. Results are bounded to official CODI GPT-2, the forced answer cue, linear
subspaces, and GSM8K.

## Implementation

- configuration: `configs/official_codi_gpt2.yaml` under `endpoint_margin_geometry`
- mechanism: `src/mech/endpoint_margin_geometry.py`
- collection and parity gate: `scripts/collect_official_codi_endpoint_margin_states.py`
- analytic sweep: `scripts/run_official_codi_endpoint_margin_sweep.py`
- generation arms: `scripts/run_official_codi_endpoint_margin_generation.py`
- gates: `src/eval/official_codi_endpoint_margin_geometry_analysis.py`
- analysis CLI: `scripts/analyze_official_codi_endpoint_margin_geometry.py`
- tests: `tests/test_endpoint_margin_geometry.py`
- Kaggle notebook: `notebooks/kaggle_official_codi_endpoint_margin_geometry.ipynb`

`src/models/official_codi.py` gained one additive change: the generator honours an
`applies_to_all_positions` attribute on an endpoint intervention. It defaults to
absent/`False`, so every completed experiment keeps intervening on the single
answer-cue forward pass exactly as before.
