# Transformer optimization notebook

New notebook: `notebooks/kaggle_codi_transformer_optimization.ipynb`.

Select **GPU T4 x2**, enable Internet, and Run All in a fresh Kaggle session. One GPU
is used. The notebook downloads the official CODI/GPT-2 checkpoint and GSM8K, fits the
same separate rank-96 heads for latent and explicit reasoning, and runs automatically.
No old results or dataset attachments are required. The original notebooks are retained.

## What is implemented

- Static preallocated K/V tensors with indexed writes, reusable decode masks, and
  preallocated hidden-state, token, count and EOS buffers. No growing decode cache or
  concatenated per-token attention mask.
- Contiguous standard Linear layouts for GPT-2's Conv1D projection weights.
- PyTorch scaled-dot-product attention with a correct explicit mask for unused static
  cache slots. The backend is selected by PyTorch for the actual GPU/shape; FlashAttention
  is not assumed. Prefill and CODI's answer cue remain batched.
- Full-graph `torch.compile` for transformer decode, answer cue, projector, head, token
  updates and embedding steps. Eligible pointwise operations can fuse; this is not a
  claim that an entire block becomes one kernel.
- Explicit CUDA Graph capture/replay for the repeating latent/cue/visible sequences.
  Automatic compiler CUDA graphs are disabled to avoid nesting graph managers around
  mutable cache buffers. Warmup and capture occur before timing.
- A four-token visible-decode graph candidate for explicit CoT, reducing host EOS checks.
  Extra work after EOS is masked and never counted as generated output. CODI defaults
  to one-token checks because its answers are short.
- Separate per-output-channel INT8 and packed INT4 weight-only candidates, using a
  custom Triton GEMV kernel with FP32 accumulation/dequantization for batch-1 decode.
  Prefill uses a once-dequantized copy of the same quantized weights. That retained
  copy means this experiment reduces decode weight traffic, not necessarily model VRAM.

The standard CUDA [FlashAttention-2 package](https://github.com/Dao-AILab/flash-attention)
and original [Marlin](https://github.com/IST-DASLab/marlin) target newer NVIDIA architectures.
A separate FlashAttention Turing port exists; it is not an automatic dependency here.
This notebook uses PyTorch SDPA and its own packed GEMV alternatives for the T4.
It does not claim to install every serving system or make dependent GPT-2 layers run
simultaneously. DeepSpeed/vLLM are not drop-in replacements for this custom latent loop.

## Output order and measurement boundaries

1. Equal-MAC experiment: one 768x9216 projection versus twelve 768x768 projections,
   using the same input/weights, with and without CUDA Graph replay. Shapes, occupancy
   and memory behavior also change, so the ratio is not a pure kernel-launch tax.
2. **Isolated block-substep table**, on one real cached position in original block 1:
   LayerNorm, QKV, attention/cache/layout, output projection, FC1, GELU, FC2 and residual/
   dispatch overhead. Attention's fused operations are kept together. Parent/child
   timings are made exclusive. This probe alone installs module hooks, then removes
   all handles. Its numbers include instrumentation overhead.
3. Compact candidate results and the selected implementation for each reasoning mode.
4. **Clean full-model latency**, without module hooks, internal timing events, diagnostic
   graphs or profiler contexts. Shows current versus optimized for dense and rank-96
   heads, output lengths and the latency saved by head compression.
5. Four coarse diagnostic tables: current/optimized per question and current/optimized
   per token/latent step. All use the rank-96 head and four columns: Explicit, CODI
   overall, CODI latent only, CODI visible only. Totals appear at top and bottom.
6. Full GSM8K accuracy for both modes, both transformer implementations and both heads.

The coarse model breakdown samples only two timing questions. It uses stage clocks,
not module hooks. The transformer is one compiled region; hooking its internal blocks
would change the execution being measured. Separate diagnostic CUDA graphs contain
external event nodes, while the main graphs stay uninstrumented. If the installed
PyTorch cannot capture timing events, a clearly labeled diagnostic-only fallback runs
compiled calls without graph replay. Diagnostic overhead is included in these tables;
**use clean latency for speed claims**.

CODI overall = shared preparation/prefill + latent + visible, printed above the question
tables. Per-token denominators are visible output tokens for explicit/visible, six
latent steps for latent, and latent plus visible steps for overall. Different per-token
columns cannot be added directly. Prompt/cue costs are amortized over generated steps.

The reported head fraction is diagnostic. The clean dense/rank-96 latency difference
is a separate practical control; different generated sequences can affect that difference.
The notebook reports generated lengths and accuracy rather than attributing all changes
to kernels. One optional PyTorch/CUPTI GPU trace is collected outside the main timing
runs and saved as `optimized_gpu_trace.json` when supported. Nsight hardware counters
are not automatically collected in Kaggle.

## Selection and correctness

- Head fitting retains the earlier 1,024/256/256 fit/selection/recovery train questions,
  four clean epochs, two recovery epochs, nested 32/64/96 training and seed 89.
- Candidates are tested on 64 head-selection questions from GSM8K train, never test.
- FP16 candidates undergo multi-step hidden-state comparisons against Hugging Face,
  then need at least 95% exact free-generation sequence agreement on validation.
- Compiled/graphed candidates must match their own eager implementation on separate
  warmup questions. Packed CUDA kernels are checked against the same dequantized
  weights before they are compiled/captured.
- Every candidate must match or exceed baseline validation correct-answer count.
  This small validation gate is not a guarantee of unchanged test accuracy.
- Timing selection uses separate warmup questions; a candidate needs an observed gain
  above 2% before replacing the current path. This threshold is a practical filter,
  not a statistical confidence claim. Fresh held-out timing uses 16 questions x 3 repeats.
- A winner is accepted only after its dense-head control also runs correctly with the
  same transformer configuration. Unsupported/failed candidates are reported; the
  current decoder remains available when none qualifies.
- Final accuracy uses all 1,319 canonical GSM8K test questions once per combination,
  batch 1, numeric exact match. CODI/explicit caps remain 64/256 tokens. Cap hits and
  every predicted/gold answer are recorded. Test results never select the winner.

Compilation, candidate checks and full accuracy add setup time. Full accuracy is eight
passes over the test set and can take substantially longer than head fitting. Runtime
and speedup on T4 have not been established by the local CPU tests; GPU-specific paths
are checked again by the notebook on Kaggle.

## Files and running

The notebook embeds both its reference decoder and new engine, and clones published
base commit `6a8d2e61950c67f012d0a9ba13ec8a70f3a25019` for the existing loader/head recipe.
It therefore runs before the new source files are pushed. Output is under
`/kaggle/working/codi_transformer_optimization/<run-id>/`.

Use `clean_samples`, `diagnostic_samples`, `substep_events`, `candidate_rows`, `tables`
and `accuracy_results` inside the notebook. One `debug.jsonl.gz` retains raw values,
settings, failures and predictions. `timing_tables.json` contains the four mean tables.
There is no collection of CSV reports to download.

From `zain-kaggle`, stage the five new files:

```powershell
git add notebooks/kaggle_codi_transformer_optimization.ipynb scripts/build_kaggle_codi_transformer_optimization_notebook.py src/inference/transformer_optimization.py tests/test_transformer_optimization.py docs/CODI_TRANSFORMER_OPTIMIZATION.md
git commit -m "Add T4 transformer optimization experiments"
git push origin zain-kaggle
```

Earlier uncommitted accuracy-notebook changes are separate from these new files.
Import the new notebook into Kaggle, then Save Version > Save & Run All.

Regenerate with `python scripts/build_kaggle_codi_transformer_optimization_notebook.py`.
The builder reuses the established bootstrap/fitting cells from the earlier notebook
and supplies full test-data loading when regenerating from its older revision.

References: [PyTorch GPT-fast](https://pytorch.org/blog/accelerating-generative-ai-2/),
[CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),
[static cache strategy](https://huggingface.co/docs/transformers/kv_cache),
[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile).
