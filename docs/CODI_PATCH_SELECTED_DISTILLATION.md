# Patch-selected two-term distillation

Ledger §101. Contract `official_codi_patch_selected_distillation_v1`. Designed after
the §100 primary and its per-question addendum.

## Idea

§100 found that at the decision state (a) teacher-side intervention and gradient
scores select the same directions, (b) a 12-direction answer-directed target repairs
as many never-solved questions as the full state, and (c) the full state wins only by
breaking fewer borderline questions. Both selectors here act on the **student's**
state, which no teacher-side score can see, and measure a nonlinear transfer effect
through the readout.

| set | selection | rank | pressure |
|---|---|---|---|
| teaching | transfer patching: `h' = s + V_S V_Sᵀ (t − s)`; greedy on the student's patched first-token accuracy (margin tie-break) | 12 | norm-matched ×1.0 |
| anchor | breakage patching: `h' = s₀ + V_S V_Sᵀ (s − s₀)` with `s₀` the official model's own latent state; greedy on the fraction of official answers broken; teaching set excluded | 24 | norm-matched ×0.3 |

Loss = CE + smooth-L1 on the teaching coordinates + smooth-L1 on the anchor
coordinates, each norm-matched to the CE gradient and scaled by its multiplier
(`student_training_step_multi`). Both targets are the teacher's coordinates.

## Arms and pairing

`patch`, `patch_anchor`, `patch_anchor_adaptive` (both sets re-selected every 250
steps on 1,024 select rows), `causal_anchor` (§100 causal set + breakage anchor).
Seeds 1–3, same damage (`projector_noise` σ from the primary's preliminary), same
data order, 1,000 steps. Baseline arms (`none`, `full`, `causal`, `relevance`, `random`,
`variance`) are read from the published §100 output's `predictions.jsonl` at the same
seeds; nothing is retrained.

## Go/no-go

Anchor breakage ≥ 2× the mean of 20 random 24-direction sets, else STOP. The Jaccard of
the teaching set with the §100 causal set is reported (≥ 0.75 means the `patch` arm is
expected to tie `causal`).

## Gates

P1 `patch_anchor − full`, P2 `− causal`, P3 `− relevance`, P4 `patch − causal`, P5
`adaptive − fixed`, P6 `causal_anchor − causal`; paired bootstrap over the 1,319 GSM8K
test questions on seed-mean correctness plus per-seed signs. CONFIRMED = P1 ∧ P2 ∧ P3
with 3/3 seeds; ANCHOR = P6 ∧ ¬P1; TIE within 3 points; else PARTIAL. Secondary,
predicted in advance: `patch_anchor` repairs ≥ 59 and breaks < 36.6 per seed vs `none`.

## Running

```bash
python scripts/run_codi_patch_selected_distillation.py \
  --reproduction-summary <official reproduction summary.json> \
  --shared-from <published codi_recovery_primary directory> \
  --output-dir outputs/codi_patch_selected --seeds 1,2,3 [--preliminary-only] [--smoke]
```

Notebook: `notebooks/kaggle_codi_patch_selected_distillation.ipynb`
(builder `scripts/build_kaggle_codi_patch_selected_distillation_notebook.py`).
