# Templated-task test of variance- versus intervention-selected distillation

Ledger §89. Contract `official_codi_templated_subspace_distillation_v1`.

## Question

Ledger §86–§88 found, twice, that distilling a fresh CODI student toward the
teacher's top-variance decision-state directions made the student *worse* than
no distillation at a fixed budget, while the intervention-selected (causal) set
cost less. Those students never solved GSM8K (≈4% exact match), so the finding
is about early loss, not final accuracy. This experiment moves the same teacher,
selectors and losses to a task the student can finish inside the quota: generated
two- and three-step arithmetic word problems in the GSM8k-Aug format.

It can decide whether the variance tax is **PERMANENT** (survives to final
accuracy, and causal selection avoids it), **TRANSIENT** (washes out), or
**REVERSED**. It cannot say anything about GSM8K difficulty.

## Data

`src/data/templated_arithmetic.py` generates problems from eight templates
(four two-step, four three-step) with positive-integer intermediates, seeded and
unique by question text. Rows are `{question, cot: "<<a+b=c>> <<c*d=e>>",
answer: "#### e"}`, so the released formatter drops the last equation exactly as
for GSM8k-Aug. Splits (all disjoint, drawn in order from one seed):

| split | rows | use |
|---|---|---|
| teacher_check | 500 | teacher CoT generation accuracy (go/no-go) |
| fit / select / validate | 2,048 each | PCA, selector fitting, rank rule |
| train | 48,000 | 3,000 steps × batch 16 |
| selection | 256 | learning curve every 300 steps |
| test | 2,000 | read once per trained student |

## Go/no-go before any training

1. Teacher greedy CoT generation from the question alone must reach ≥ 80% on
   the 500 held-out templated problems.
2. The rank rule (gap ≥ 0.05, causal retention ≥ 0.5, maximise gap × retention
   over r ∈ {8, 12, 16}) must select a rank.
3. Dense colon first-token accuracy on the validate split must be ≥ 80%.

`--preliminary-only` runs exactly this and writes `preliminary.json`; the
notebook flag `RUN_PRELIMINARY_ONLY` exposes it so the ~10-minute check is done
before spending the remaining quota.

## Arms, training, outcomes

Arms `none`, `variance`, `causal`; five seeds (split across accounts with
`--seeds`). Student = checkpoint embeddings/readout with fresh LoRA and projector
(§86). Loss = answer cross-entropy + smooth-L1 on the selected teacher-PCA
coordinates, with the distillation gradient norm-matched to the CE gradient.
AdamW 1e-4, cosine with 150 warmup steps, weight decay 0.1, clip 2.0, exact
micro-batch accumulation 2 × 8.

Primary: exact match on the 2,000 test problems, paired bootstrap over questions
on seed-mean correctness.

| gate | comparison | reading |
|---|---|---|
| T1 | `none − variance` | lower bound > 0 → tax persists; upper < 0 → reversed; covers 0 → transient |
| H1 | `causal − variance` | lower bound > 0 → causal selection beats variance |
| H1b | `causal − none` | two-sided |
| secondary | steps to 30% selection exact match | reported, not gated |

Claims: PERMANENT = T1 ∧ H1; PARTIAL = exactly one; REVERSED = T1 upper < 0;
TRANSIENT otherwise.

## Running

```bash
python scripts/run_codi_templated_subspace_distillation.py \
  --reproduction-summary <official reproduction summary.json> \
  --output-dir outputs/codi_templated_subspace_distillation \
  --seeds 1,2,3 --max-seconds 30600 [--preliminary-only] [--smoke]
```

Sessions pause cleanly at `--max-seconds` with a resumable checkpoint; publish
the output directory and pass it back with `--resume-from`. The other account's
finished output is pooled with `--aggregate-from`. Shared artefacts
(`problems.json`, `preliminary.json`, `teacher_cache.pt`, `selectors.json`) are
reused from any attached output.

Notebook: `notebooks/kaggle_codi_templated_subspace_distillation.ipynb`
(builder `scripts/build_kaggle_codi_templated_subspace_distillation_notebook.py`).
