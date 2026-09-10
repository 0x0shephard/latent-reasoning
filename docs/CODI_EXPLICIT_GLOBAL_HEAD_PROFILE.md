# CODI / explicit GPT-2 bottleneck notebook

Upload `notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb` to Kaggle.
Start a fresh session, select **Settings > Accelerator > GPU T4 x2**, turn Internet
on, and Run All. One T4 is used; the notebook stops if a different GPU is selected.
No attachments, reproduction summary, trained head, or configuration edits are needed.

The notebook automatically clones published base commit
`6a8d2e61950c67f012d0a9ba13ec8a70f3a25019` and embeds the current profiling helper.
Uploading this notebook works even before pushing the changes to GitHub.

## What runs

Both modes use the released, jointly trained CODI GPT-2 teacher/student weights.
Explicit generation starts after the question; CODI uses six latent passes and then
its answer cue. Each mode gets a separately fitted global low-rank head.

The fixed deployment setting is **rank 96, eager PyTorch, FP16, batch 1**. Fitting
retains the proposal's activation whitening, nested 32/64/96 training, KL/top-token/
margin losses, four clean epochs and two recovery epochs. The fit/selection/recovery
populations are 1,024/256/256 GSM8K train questions, with the previous notebook's
state caps and seed 89. Sixteen additional questions supply timing, with three repeats.
These are newly fitted heads; the PDF's reported numbers are not presumed reproduced.

There is no backend/rank/batch-size sweep or full-test accuracy experiment in this
bottleneck notebook. Decoder parity and held-out state agreement are still checked.
Visible generation caps are 64 tokens for CODI and 256 for explicit CoT; cap hits are
reported. Head fitting remains the slow setup step and reuses matching saved heads
when Run All is repeated in the same session/output folder.

## The output

One table, mean milliseconds per question:

| Explicit | CODI overall | CODI latent only | CODI visible only |
|---|---|---|---|
| Mean timings | Mean timings | Mean timings | Mean timings |

The row index names tokenization, CPU allocation, input transfer, embeddings, each
of the 12 transformer blocks, final norm, transformer bookkeeping, projector, low-rank
head, argmax, token/cache updates, EOS synchronization, output transfer, text decoding,
and Python/tracing gaps. Total average time is repeated at the top and bottom.

Latent includes the six reasoning passes and projector. Visible includes the answer
cue, visible generation and output handling. Overall additionally includes shared
question preparation, transfer and prompt prefill. Consequently latent + visible can
be smaller than overall; shared work is not arbitrarily assigned to either phase.

All detailed spans use one CUDA-stream clock, including CPU-only stages. Nested
intervals are partitioned exclusively so rows reconcile with their column's total.
This measures instrumented elapsed time, including host-induced stream idle time,
not pure kernel execution. CPU/GPU overlap is not counted twice. Profiling overhead
can affect component timings, so two short lines also show unprofiled dense/rank-96
wall-time means and their mean output lengths. These free-generation comparisons
can be affected by differing output lengths; they are not matched-token speedups.
Downloads, model loading, fitting and warmup are timed as one-time setup, outside the
per-question table. Their individual records remain available for debugging.

`bottleneck`, `profile_samples`, and `clean_samples` are directly inspectable notebook
variables. The only CSV is a copy of the displayed table. One optional `debug.jsonl.gz`
contains individual named-module/stage events, question/token/phase IDs, clean samples,
setup operations, training reports, partitions and package versions. The final cell
includes a short snippet for inspecting one question without downloading anything.
Intermediate `.pt` files are automatic state/head caches, not reports to review.

## Setup messages

The previous pip output mixed unrelated preinstalled-package conflicts with notebook
pins. The new setup reads canonical GSM8K JSONL directly, removes datasets/dill/pandas
installation pins, and uses `huggingface_hub>=0.34,<1` with `hf_xet`. It retains the
verified Transformers 4.52.4 / PEFT 0.15.2 behavior and Kaggle's installed PyTorch/CUDA.
Installation logs are retained, required imports are checked, and failures stop execution.

The official loader's embedding initialization notice and automatic PEFT Conv1D
orientation correction are recorded quietly. Checkpoint loading subsequently restores
the released weights. Other warnings remain visible. The new decoder starts GPT-2
with `DynamicCache`, avoiding legacy-tuple conversions; parity checks compare it with
the original CODI decoder and Hugging Face explicit generation. The old reference
itself may emit its single legacy-cache deprecation notice during verification.

## Push to Git

From the repository on `zain-kaggle`:

```powershell
git add notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb scripts/build_kaggle_codi_explicit_global_head_profile_notebook.py src/inference/global_head_comparison.py tests/test_global_head_comparison.py docs/CODI_EXPLICIT_GLOBAL_HEAD_PROFILE.md
git commit -m "Simplify T4 CODI bottleneck notebook"
git push origin zain-kaggle
```

Then import the updated `.ipynb` into Kaggle and Save Version > Save & Run All.
The original deployment benchmark and separate ablation notebook are unchanged.

Maintainers: regenerate after editing the helper or builder with
`python scripts/build_kaggle_codi_explicit_global_head_profile_notebook.py`.
