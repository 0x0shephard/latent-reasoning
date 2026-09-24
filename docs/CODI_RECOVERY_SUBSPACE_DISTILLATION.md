# Recovery test of distillation targets on the official CODI checkpoint

Ledger §94. Contract `official_codi_recovery_subspace_distillation_v1`.

## Why this design

§93 states what a decisive selector test needs: a student in the measurable-accuracy
regime, a distillation term far from converged, and a teacher on which the selectors
diverge. From-scratch students never reached the first condition at this quota
(§90, §92). This experiment damages one component of the official checkpoint and
measures which target restores GSM8K accuracy under continued training.

| damage | what changes | what stays |
|---|---|---|
| `projector_noise` (primary, §96) | each projector Linear weight gets `σ·std(W)·N(0,1)`, σ calibrated in the go/no-go, seeded per run | biases, LayerNorm, LoRA, embeddings, readout |
| `lora_half` (robustness) | every LoRA B matrix is scaled by 0.5 | projector, embeddings, readout |
| `projector` (non-default) | the projector is re-initialised; recovery proved bimodal across accounts (§96) | LoRA, embeddings, readout |

## Go/no-go before any counted seed

1. Headroom: `official − damaged ≥ 0.20` exact match on the 256-row selection split
   (§95). For `projector_noise`, σ is the smallest of {0.25, 0.5, 1, 2} meeting this.
2. Selector divergence: causal and variance share ≤ 8 of 12 PCs at the chosen rank.
3. Term not converged: variance and causal distillation loss on 256 fit rows ≥ 1.5× its
   value at the official weights (lowered from 2× in ledger §97).
4. Recovery pilots: two `none` runs (seeds 0 and 100), 2,000 steps, curve every 100.
   The recovery threshold is `damaged + 0.5 × (official − damaged)` on the selection
   split. Step budget = 1,000 if both pilots reach it by step 1,000, 2,000 if both by
   2,000, else STOP (a disagreeing pair is a STOP, §96).

`--preliminary-only` runs exactly this; the notebook flag `RUN_PRELIMINARY_ONLY`
exposes it.

## Stages

| stage | damage | arms | seeds | runs | ≈hours |
|---|---|---|---|---|---|
| primary | projector_noise | none, full, variance, relevance, causal, random | 1–5 | 30 | 25 |
| weights | projector_noise | variance_x0.3, variance_x3, causal_x0.3, causal_x3 | 1–3 | 12 | 6 |
| damage | lora_half | none, variance, causal, random | 1–3 | 12 | 6 |

Suggested split: account A runs primary seeds 1–3 then weights; account B runs
primary seeds 4–5 then damage. Shared artefacts (`teacher_cache.pt`,
`selectors.json`, `sampling.json`, `preliminary.json`) are reused from any attached
output; per-run records resume from this account's previous sessions.

## Training

AdamW with LoRA at 1e-4 and projector at 5e-4, warmup 50, cosine to zero over the
step budget, weight decay 0.1, clip 2.0, batch 16 as 2 × 8 exact accumulation.
Distillation: smooth-L1 on the selected teacher-PCA coordinates of the student's
latent decision state against the teacher's explicit-CoT decision state, scaled by
teacher coordinate std, gradient norm-matched to the answer cross-entropy; weight
arms multiply the matched gradient by 0.3 or 3. Each step records the cosine between
the CE gradient and the raw distillation gradient.

## Outcomes and gates

Primary: GSM8K test exact match on 1,319 questions at the final step, paired
bootstrap over questions on seed-mean correctness.

- R0: the `none` arm's final selection-split exact match (seed mean) reaches the
  recovery threshold. S1: `full − none` > 0.
- H1: `causal − variance` > 0. H2: `causal − relevance` > 0. H3: `causal − random` > 0.
- `variance − none` and `causal − none` two-sided.
- CONFIRMED = H1 ∧ H3. REVERSED if variance beats causal. NULL if H1 covers zero
  within 3 points. PARTIAL if H1 but not H3. INCONCLUSIVE otherwise.
- Secondary: steps to 30% selection exact match, test NLL, mean gradient cosine.

## Running

```bash
python scripts/run_codi_recovery_subspace_distillation.py \
  --reproduction-summary <official reproduction summary.json> \
  --output-dir outputs/codi_recovery_primary \
  --damage projector_noise --arms none,full,variance,relevance,causal,random \
  --seeds 1,2,3 --max-seconds 30600 [--preliminary-only] [--smoke]
```

Notebook: `notebooks/kaggle_codi_recovery_subspace_distillation.ipynb`
(builder `scripts/build_kaggle_codi_recovery_subspace_distillation_notebook.py`).
