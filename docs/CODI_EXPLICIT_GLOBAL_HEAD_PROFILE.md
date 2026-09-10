# CODI versus explicit GPT-2: global head and full timing diagnostics

Upload `notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb` to Kaggle.
Enable a GPU and Internet, then Save Version > Save & Run All. No attached datasets,
old summaries, trained head artifacts, or manual path changes are required. The real
experiment runs by default; `SMOKE=True` is an optional small integration run.

The notebook clones published base commit `6a8d2e61950c67f012d0a9ba13ec8a70f3a25019`
and embeds the new runtime. It works before these files are pushed. The prior benchmark
and ablation notebook are unchanged. Both GPT-2 modes use the official CODI checkpoint
and share transformer weights: one uses six latent passes; the other generates explicit
CoT directly from the question, matching the teacher's training prefix. This is not a
separate CoT-SFT checkpoint and does not use pretrained, unfine-tuned GPT-2 as a teacher.

## Experiment

- Separate head fitting for each mode on its own dense trajectories.
- Canonical GSM8K train, 1,024 fit / 256 validation / 256 recovery questions; additional
  disjoint timing/warmup questions. Same partitions for both modes.
- Activation-whitened initialization, KL + top-token CE + margin, nested 32/64/96,
  four clean epochs and two rank-64 on-policy recovery epochs. Same recipe as the PDF;
  new training populations mean this is not a bitwise reproduction of its artifact.
- FP32 fitting; merged-LoRA FP16 deployment. Full 1,319-question quality evaluation.
- Dense, eager rank96, compiled rank96, original Triton arithmetic, and FP32-accumulating
  Triton with output rounding. Numerical agreement and regret are measured explicitly.
- Default clean timing: 64 questions, batch sizes 1/8/32, five repeats, randomized arm
  order; both free generation and identical dense-token replay. Replay is timing only.
- Detailed profiling: four questions, batch 1, two repeats per mode/head. Every module
  and explicit stage is logged. One additional question per mode/head produces a full
  CPU/CUDA operator trace. Increase sampling settings for more detail.

Explicit CoT is capped at 256 generated tokens (CODI: 64). Truncation and completed-answer
accuracy are reported. Do not interpret early-stop/truncation as a pure speed improvement.
The notebook checks both decoders against reference implementations before fitting.
GPU-specific execution is checked on Kaggle; local CPU tests cannot establish GPU speed.
An NVIDIA T4 is the closest match to the previous deployment report. One GPU is used.
No end-to-end GPU runtime estimate has been validated; the explicit mode can take much
longer than latent decoding. Completed heads and completed mode benchmarks can be reused
through `RESUME_DIR`, which must point at a matching saved run directory.

## Timing and raw values

Outputs: `/kaggle/working/codi_explicit_global_head/full/<run-id>/`.

| Output | Meaning |
|---|---|
| `timing_averages.csv` | Mean/median/SD/range and speedup by mode, head, protocol, batch |
| `timing_individual.csv` | Every clean batch/repeat, question IDs and token counts |
| `timing_speedup_intervals.csv` | Descriptive paired-repeat bootstrap speedup intervals |
| `head_averages.csv`, `head_individual.csv` | Head-only timing; each raw microbenchmark window contains 100 calls |
| `<mode>/layer_stage_averages.csv` | Named layer/submodule and explicit-stage timing averages |
| `<mode>/detailed_events.jsonl.gz` | Each call with layer, phase, question, token, repeat and parent ID |
| `<mode>/operators_<arm>.json` | Complete sampled operator/kernel timeline for Perfetto/Chrome |
| `<mode>/operator_events_<arm>.csv` | Individual operator events |
| `<mode>/operator_averages_<arm>.txt` | Operator averages by shape |
| `setup_events.jsonl.gz`, `setup_averages.csv` | Download, data parse, model/state loading, device movement, fitting and warmup |
| `profiling_overhead.csv` | Matched clean-versus-instrumented runtime |
| `quality_summary.csv`, `<mode>/quality_<arm>.json` | Accuracy/retention and every generated answer/CoT |

Layer indices are part of names, such as `transformer.h.0.attn`, `transformer.h.11.mlp`,
and `transformer.ln_f`. Both parent modules and their children are timed: **do not sum
inclusive parent/child times**. CUDA event intervals include stream idle time; host
module wall time generally measures dispatch. Actual kernel execution is in operator
traces. Profiling changes performance and is kept outside the reported clean speedups.
The compiled head stays intact during profiling; internal compiled/fused kernels appear
in operator traces rather than hooks that would alter compilation.

Timing includes prompt normalization/tokenization/padding in setup/detailed logs, H2D
input IDs and masks, cue construction, embedding lookup, prefill, every latent step and
projector, every visible-token transformer pass, head + argmax, KV-cache reference
updates, EOS host synchronization, D2H tokens/counts, and text decoding. Internal cache
concatenations, attention operations, allocations and CUDA copies appear in native
operator traces. Model/data downloads are cold-or-cached setup observations, not mixed
into repeated warm inference. Training is timed by initialization/clean/recovery phase;
per-minibatch training profiling is not enabled. Batch 1 provides per-question latency;
for larger batches, the raw value is a batch measurement shared by those questions.

Read individual layer calls in a Kaggle cell:

```python
import pandas as pd
calls = pd.read_json(RUN_DIR / 'explicit_cot' / 'detailed_events.jsonl.gz', lines=True)
display(calls[calls['name'].eq('transformer.h.0.attn')])
```

For large logs use line-by-line `gzip.open` rather than loading all rows. Open an
`operators_*.json` file in https://ui.perfetto.dev to inspect the CPU/GPU timeline.

## Git and Kaggle

From this repo on `zain-kaggle`:

```powershell
git status
git add notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb scripts/build_kaggle_codi_explicit_global_head_profile_notebook.py src/inference/global_head_comparison.py tests/test_global_head_comparison.py docs/CODI_EXPLICIT_GLOBAL_HEAD_PROFILE.md
git commit -m "Add CODI and explicit CoT global-head profiling notebook"
git push -u origin zain-kaggle
```

Import the local notebook into Kaggle (File > Import Notebook), enable GPU and Internet,
and Save Version > Save & Run All. Kaggle Save Version preserves a Kaggle run, not a Git
commit. To regenerate after runtime changes:
`python scripts/build_kaggle_codi_explicit_global_head_profile_notebook.py`.

References:
- Original CODI architecture/training: https://github.com/zhenyi4/codi
- Teacher sequence construction: `src/data/official_codi_training.py`
- Kaggle: https://www.kaggle.com/docs/notebooks
- Profiling: https://docs.pytorch.org/docs/stable/profiler.html
