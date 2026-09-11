# Model and dataset ablations of the global LM head

Notebook: `notebooks/kaggle_model_dataset_global_head_ablations.ipynb`.

Import into Kaggle, select GPU T4 x2, turn Internet on, and Run All. One T4 is used.
No attachments or old head artifacts are required. The notebook embeds the reference
decoder and model/dataset adapter, and downloads the existing pinned repository base
for its model loader and head-fitting functions. It runs before these new files are
pushed. Existing notebooks are unchanged.

## Experiment

| Model | Source | Available reasoning modes |
| --- | --- | --- |
| GPT-2 | Existing verified author-released CODI checkpoint | CODI and explicit CoT on the same weights |
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M`, fixed revision | Explicit CoT; CODI N/A |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B`, fixed revision | Explicit CoT; CODI N/A |

The two added models are the agreed base checkpoints. There are no trained CODI
weights for them in this repository, so this prepared notebook does not manufacture
latent reasoning or silently substitute another model. Adding trained CODI versions
requires a separate training/checkpoint decision. It does not fine-tune the backbone.
Absolute math accuracy across these checkpoints is not a controlled model comparison:
GPT-2 is math-trained, while the new models are base pretrained checkpoints. The
within-model dense-versus-compressed comparison is the principal head experiment.

Each model is evaluated on all three datasets:

- GSM8K: full 1,319-question test split.
- SVAMP: all 1,000 questions from the original release.
- ASDiv: 2,084 single-scalar numeric-answer questions from the original 2,305.
  The existing numeric scorer cannot score the 221 time, ratio, name, multiple-answer
  and other non-scalar rows correctly. Their IDs, gold answers and exclusion reason
  are recorded. This is explicitly **ASDiv numeric subset**, not full-ASDiv accuracy
  or the ASDiv-A cross-validation protocol. Unit annotations are removed before
  parsing, so numbers in units never replace the actual answer.

Only the backbone and evaluation dataset vary. The original head recipe remains:
activation-whitened initialization; nested training ranks 32/64/96; four clean epochs;
two recovery epochs using rank-64 on-policy states; deployment rank 96; seed 89.
State caps are 4,096/1,024/2,048; fit/selection/recovery populations are 1,024/256/256
GSM8K training questions; collection and distillation batches are 8. No adaptive
selection, compiler, CUDA Graph, quantization, or batch-size sweep is included.

Heads are fitted once per model/mode on the identical GSM8K split and reused across
all datasets. The test datasets are never used to fit or choose a head. Exact
question overlaps with head-fitting/selection/recovery/warmup data stop the run.
This check concerns this experiment's data use, not unknown pretraining contamination.

Inference is eager FP16, batch 1, greedy generation, with the unchanged 64 CODI / 256
explicit output-token caps. GPT-2 keeps six latent steps and its original prompt and
answer cue. Explicit models receive the same question-only prompt without an added
chat template or few-shot examples. Dense decoding is checked against Transformers
generation before fitting. Each architecture uses its own tokenizer, embeddings,
native vocabulary and transformer blocks; the LM head is called only for visible
generation. Parameters and dimensions are logged. Qwen is about four times GPT-2's
parameter count; rank 96 represents different compression ratios across models.

## Reporting and runtime

Timing uses 16 fixed questions x 3 repeats per pair. GSM8K retains the original
held-out training timing questions; SVAMP/ASDiv timing questions are a deterministic
sample of their evaluation set. Timing does not select a head or model. Full eligible
accuracy is run separately, batch 1, without timing hooks. There are 35,224 accuracy
generations across the available model/mode/head/dataset combinations, plus fitting
and timing; plan for a substantial Kaggle run. No full T4 run has been performed locally.

Every pair produces the original four mean-only tables:

1. Dense, milliseconds per question.
2. Rank 96, milliseconds per question.
3. Dense, milliseconds per generated token/latent step.
4. Rank 96, milliseconds per generated token/latent step.

Each table retains Explicit / CODI overall / CODI latent only / CODI visible only,
with total at the top and bottom. Missing CODI results are N/A, not zero. Layer rows
follow the actual backbone depth. All nine result panels are available in the notebook;
the first is expanded initially, and the other panels can be expanded individually.
A compact summary reports clean mean latency and accuracy before/after compression.

Component timing uses hooks on whole blocks and displayed components, not recursive
submodule hooks. Clean timing and accuracy run separately with no hooks attached.
Profiler inflation, generated lengths, cap hits, and answer accuracy are reported.
Use clean latency for speed claims; free-generation length changes contribute to
latency changes. Per-token numbers use each model's tokenizer, so milliseconds per
question is preferable for comparisons between models.

CODI overall includes shared preparation/prefill plus latent and visible work; the
equation is displayed above per-question tables. Per-token denominators are visible
tokens for Explicit/visible, six latent steps for latent, and latent + visible steps
for overall. Columns with different denominators do not add together.

Output: `/kaggle/working/codi_model_dataset_ablations/<fingerprint>/`.
`all_results[(model_key, dataset)]` contains four tables, accuracy summaries, and each
clean/profiled question measurement. The single `debug.jsonl.gz` contains individual
events, predicted/gold answers, setup and exclusions. Per-pair `results.json` files
support reuse; they do not need to be opened. Fitted heads/state caches are reused
in the same working directory. The raw log is preserved when completed pairs are
reused. Preserve Kaggle working outputs to retain caches across session resets.

## Git

On `zain-kaggle`:

```powershell
git add notebooks/kaggle_model_dataset_global_head_ablations.ipynb scripts/build_kaggle_model_dataset_global_head_ablation_notebook.py src/inference/model_dataset_ablation.py tests/test_model_dataset_ablation.py docs/MODEL_DATASET_GLOBAL_HEAD_ABLATIONS.md
git commit -m "Add fixed global-head model and dataset ablations"
git push origin zain-kaggle
```

Regenerate with:

```powershell
python scripts/build_kaggle_model_dataset_global_head_ablation_notebook.py
```

Sources: [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B),
[SVAMP](https://github.com/arkilpatel/SVAMP),
[ASDiv](https://github.com/chaochun/nlu-asdiv-dataset),
[released CODI models](https://github.com/zhenyi4/codi).
