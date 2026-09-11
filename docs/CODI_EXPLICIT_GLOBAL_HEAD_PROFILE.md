# CODI / explicit GPT-2: before and after global-head compression

Upload `notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb` to Kaggle.
Start a fresh session, choose **Settings > Accelerator > GPU T4 x2**, enable Internet,
and Run All. One T4 is used. No attachments or configuration changes are required.

The notebook clones published base commit
`6a8d2e61950c67f012d0a9ba13ec8a70f3a25019` and embeds its updated profiling helper.
Both modes use the released jointly trained CODI GPT-2 teacher/student weights:
explicit generates after the question; CODI uses six latent steps before its answer.

## Four tables and accuracy

Exactly four timing tables are displayed, with the same four numeric columns:
**Explicit | CODI overall | CODI latent only | CODI visible only**.

1. Before compression: mean milliseconds per question.
2. After compression: mean milliseconds per question.
3. Before compression: mean milliseconds per generated token / latent step.
4. After compression: mean milliseconds per generated token / latent step.

Each table repeats its total at the top and bottom. The rows include every transformer
block, embeddings, final norm, projector, LM head, argmax, token/cache updates, input
preparation, transfers, EOS synchronization, decoding and runtime gaps. Each question
table prints the actual reconciliation equation immediately above it:

`CODI overall = shared prompt/runtime work + latent reasoning + visible answer`

The shared contribution already appears in the component rows, including prompt
prefill in the block rows. It is not added again as a duplicate subtotal row. Latent
and visible intentionally exclude this shared work. For the previously supplied run:

`221.347 = 24.945 shared + 144.222 latent + 52.180 visible ms/question`.

Per-token values use **summed elapsed time / summed generated step count**:

- Explicit: generated visible tokens, including EOS.
- CODI overall: six latent steps plus generated visible tokens per question.
- CODI latent: six latent steps per question.
- CODI visible: generated visible tokens, including EOS.

Prompt and forced-cue costs are amortized over those steps. Prompt/cue tokens do not
increase the denominator. Columns in the per-token tables therefore have different
denominators and cannot be added directly. These are amortized costs, not isolated
single-token kernel benchmarks. In particular, block rows still include prompt prefill.

After timing, the notebook evaluates **all 1,319 GSM8K test questions**, once for each
mode/head. It prints dense and rank-96 final-answer numeric exact-match accuracy,
correct counts and percentage-point changes. This is generation accuracy, not token
agreement. Test questions are never used for fitting or model selection. Accuracy
uses batch 1 and the same 64-token CODI / 256-token explicit generation caps as timing.
Cap hits and individual predictions/correctness decisions are retained and reported.

## Method and measurement

Before = the original dense vocabulary projection. After = a separately fitted rank-96
head for each reasoning mode. The transformer, merged weights, dtype, decoder, prompt,
KV-cache behavior and decoding policy are the same on both sides. The dense baseline
uses the same body-only decoder that computes the head only at needed positions; it
is not an artificially slow full-model forward at every position.

The proposal's head recipe is retained: activation whitening, nested 32/64/96 training,
KL/top-token/margin losses, four clean epochs and two recovery epochs; 1,024/256/256
fit/selection/recovery GSM8K train questions. State caps and seed 89 match the previous
notebook. Deployment is eager PyTorch, FP16, rank 96, batch 1. Timing uses 16 held-out
train questions, three repeats, both heads and both modes. There is no backend/rank/
batch-size sweep. These are newly fitted heads, not the PDF's saved artifacts.

Profiling now hooks only each whole transformer block and the displayed components.
It no longer hooks all nested attention/MLP/linear modules or creates record_function
markers for the timing pass. This reduces the previous profiler's distortion, though
CUDA-event instrumentation still costs time. Both dense and compressed heads receive
the same instrumentation. The notebook checks that profiling preserves their tokens.

All spans use one CUDA-stream clock, including host-induced stream idle time. Nested
spans are partitioned exclusively so each column's component rows add to its total.
These are instrumented elapsed times, not pure GPU kernel durations. Separate clean
wall-time means and profiler inflation factors are printed for each head and mode.
Do not infer production speedups from profiled totals. Free generations may differ
in length or answer, so even clean totals are not matched-token speedup measurements.

The supplied previous run demonstrated substantial instrumentation distortion:
explicit 595.266 / 191.08 = 3.12x; CODI 221.347 / 74.50 = 2.97x. Old detailed numbers
should not be compared directly with the revised profiler's breakdown.

## A concrete cost calculation

Assume batch 1, FP16, GPT-2 width d=768, eligible vocabulary V=50,259, rank r=96,
and cached attention context S=100. The base configuration is documented in the
[GPT-2 config](https://huggingface.co/openai-community/gpt2/blob/main/config.json).
CODI adds special tokens; the vocabulary boundary changes the rounded numbers only
slightly. A multiply-accumulate (MAC) is counted as two FLOPs.

For one cached decode position, ignoring small bias/normalization/nonlinear operations:

- Block projections: QKV 3d^2 + attention output d^2 + MLP 8d^2 = 12d^2 MACs.
- Block attention: QK plus AV = 2Sd MACs.
- Dense head: dV MACs.
- Rank-96 head: dr + rV MACs.

| Component, one decode position | MACs | FP16 weight / KV bytes | Optimistic streaming time at 320 GB/s |
|---|---:|---:|---:|
| One transformer block | 7.23 million | 14.46 MB | 45.2 microseconds |
| Dense head | 38.60 million | 77.20 MB | 241.2 microseconds |
| Rank-96 head | 4.90 million | 9.80 MB | 30.6 microseconds |

Block bytes approximate 24d^2 for weights plus 4Sd for cached keys/values. Head bytes
are twice their weight count. This assumes weights are streamed once; intermediate
activation traffic, writes, dispatch, kernel launches and synchronization are omitted.
These are conditional memory-only estimates, not predicted end-to-end runtimes.

The [T4 specifications](https://www.nvidia.com/en-gb/data-center/tesla-t4/) give about
320 GB/s memory bandwidth and 65 FP16 TFLOPS peak. Dividing these small matvec FLOP
counts by peak TFLOPS is misleading: batch-1 inference often cannot reuse weights
enough to reach that compute throughput.

An explicit illustrative model is:

`time = bytes / effective_bandwidth + kernel_count * launch/dispatch_cost`.

Assume 160 GB/s effective bandwidth, 10 microseconds of launch/dispatch per kernel,
and approximately 20 kernels per block, one for the dense head, two for rank 96.
That gives approximately **0.29 ms/block, 0.49 ms/dense head, 0.081 ms/rank-96 head**
per cached position. Kernel count/fusion, bandwidth and host dispatch are assumptions,
not measurements from the user's T4; actual values may differ substantially. Argmax
is separate. The calculation explains why a compressed head can cost less than a
whole transformer block even though the original dense head is much larger.

The supplied table's 42-50 ms/block is **per question**, not one layer invocation.
Assume P=88 prompt tokens and N=24 output tokens (close to the supplied output mean).
The block processes P prompt positions in one prefill, then N-1 cached positions:

`12d^2(P+N-1) + attention terms ~= 0.80 billion MACs per block per question`.

The compressed head performs N predictions:

`N*r*(d+V) ~= 0.118 billion MACs per question`.

That is about 6.8x as much arithmetic for a block across the whole question. Prompt
positions are processed together, reusing weights, so their latency is not the prompt
length multiplied by the single-token latency. Per-question operation counts, memory
traffic, multiple kernels and profiling overhead must all be distinguished.

## Inspection and running

Four DataFrames are in `tables`; individual runs in `profile_samples` and `clean_samples`;
accuracy totals in `accuracy_results`. One optional `debug.jsonl.gz` contains individual
component/stage calls with head/mode/phase/question/token IDs, setup timings, model/head
reports, generated predictions, gold answers and correctness decisions. A snippet in
the notebook shows how to inspect one question. All four mean tables are also saved
together in `timing_tables.json`. Intermediate `.pt` files are automatic fitting caches.

The setup reads canonical GSM8K JSONL directly, avoids datasets/dill/pandas pins, and
uses `huggingface_hub>=0.34,<1` with `hf_xet`, Transformers 4.52.4 and PEFT 0.15.2.
Kaggle's PyTorch/CUDA is retained. Failed installs/imports stop execution. Expected
embedding-initialization/Conv1D notices are logged; GPT-2 uses DynamicCache and retains
parity checks against the original CODI path and Hugging Face explicit generation.

From the repository on `zain-kaggle`:

```powershell
git add notebooks/kaggle_codi_vs_explicit_cot_global_head_profile.ipynb scripts/build_kaggle_codi_explicit_global_head_profile_notebook.py src/inference/global_head_comparison.py tests/test_global_head_comparison.py docs/CODI_EXPLICIT_GLOBAL_HEAD_PROFILE.md
git commit -m "Add before-after CODI timing and GSM8K accuracy"
git push origin zain-kaggle
```

Import the updated notebook into Kaggle, then Save Version > Save & Run All.
Maintainers can regenerate it with
`python scripts/build_kaggle_codi_explicit_global_head_profile_notebook.py`.
The original deployment benchmark and separate ablation notebook are unchanged.
