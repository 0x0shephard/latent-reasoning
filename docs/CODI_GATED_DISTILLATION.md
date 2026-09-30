# Counterfactually gated distillation

Ledger §103. Contract `official_codi_gated_distillation_v1`. Designed after §100–§102.

## Idea

In the repair regime every answer-directed target repairs the never-solved questions
about as well as the full state and loses only by breaking questions plain training
already gets right (§100 addendum, §102). Breaks come from copying pressure on examples
the student already answers correctly. The copying term is therefore gated **per
example, per step, by an intervention on the student's own decision state**, computed
from the forward pass the step already makes at the cost of one readout matmul:

```
before = argmax(W s) == gold
after  = argmax(W (s + V_S V_Sᵀ (t − s))) == gold      # full patch: after = argmax(W t) == gold
gate   = (not before) and after                          # "patch" gate
```

Copy only where the gate is true. The control is the error gate, `not before`, which
needs no intervention.

| arm | copies | gate |
|---|---|---|
| `gated_causal` | §100 causal 12 directions | patch of those 12 flips wrong → right |
| `gated_full` | all 768 | full patch flips wrong → right |
| `wrong_causal` | causal 12 | student currently wrong |
| `wrong_full` | all 768 | student currently wrong |

Pressure stays norm-matched to the CE gradient per step and is concentrated on the
gated examples; the gate fraction is logged per step. Seeds 1–3, paired with the §100
runs via the published primary output; 1,000 steps; σ from the primary's preliminary.

## Gates

G1 `gated_causal − full`, G2 `gated_full − full`, G3 `gated_causal − wrong_causal`, G4
`gated_full − wrong_full`; paired bootstrap over the 1,319 GSM8K test questions plus
per-seed signs. CONFIRMED = (G1 ∧ G3) or (G2 ∧ G4) with 3/3 seeds on the full
comparison; SELECTIVE = beats full but the counterfactual gate does not beat the error
gate; TIE within 3 points; else PARTIAL. Predicted in advance: each gated arm breaks
< 25 and repairs ≥ 54 per seed against `none`.

## Running

```bash
python scripts/run_codi_gated_distillation.py \
  --reproduction-summary <official reproduction summary.json> \
  --shared-from <published codi_recovery_primary directory> \
  --output-dir outputs/codi_gated --seeds 1,2,3 [--preliminary-only] [--smoke]
```

Notebook: `notebooks/kaggle_codi_gated_distillation.ipynb`
(builder `scripts/build_kaggle_codi_gated_distillation_notebook.py`).
