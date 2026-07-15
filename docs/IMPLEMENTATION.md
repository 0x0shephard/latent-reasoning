# Implementation Walkthrough

An in-depth, file-by-file account of the code written so far, why each piece exists, and
where it sits in the overall research plan ([`docs/PLAN.md`](PLAN.md)). Read this alongside
`PLAN.md` (the *what/why* of the study) — this doc is the *how* of the code.

---

## Where we are in the overall plan

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 0** | Scaffolding + session-safe trainer (checkpoint / resume / determinism) | ✅ **Complete** — 3/3 tests green |
| **Phase 1a** | Data + answer-extraction + prompt layer (the "measuring instrument") | ✅ **Complete** — CPU unit-tested |
| **Phase 1b** | Generic trainer + real HF SFT baselines (No-CoT / CoT) + `run_eval.py` + Kaggle notebook | ✅ **Complete** — CoT-SFT checkpoint evaluated on all four sets |
| **Phase 2** | Latent-LM (`<bot>`/`<eot>` continuous thoughts) + CODI & KaVa losses + R-KV compression | ✅ **Implemented and CPU-validated** — Kaggle GPU runs next |
| **Phase 3** | Mechanistic analysis (probes, CKA/SVCCA, ablation, patching) | ◻ Not started |
| **Phase 4** | Supervision-granularity continuum + writing | ◻ Not started |

**Current milestone:** Phase 1 produced a step-24,102 CoT-SFT checkpoint and a reproducible
200-example evaluation (GSM8k 26.0%, SVAMP 30.0%, MultiArith 74.44%, GSM-Hard 5.0%). Phase
2's implementation and real-data contract pass locally; the remaining work is running the
matched CODI and KaVa configs on Kaggle and evaluating their checkpoints.

**Run everything so far:**
```bash
python -m pytest -q                                   # all phase gates (CPU, no downloads)
python -m src.train.kaggle_run --config configs/phase0.yaml   # the session-safe trainer
```

---

## Design philosophy (applies to every phase)

1. **One session-safe loop, reused everywhere.** Checkpoint/resume/time-budget logic is
   written once and every future task (SFT, CODI, KaVa) plugs into it. This is why Phase 0
   builds the loop against a throwaway synthetic task *before* any modeling.
2. **Step-deterministic batching.** Batches are a pure function of the step index, so a run
   killed by Kaggle's session cap resumes bit-for-bit — no RNG drift, no duplicated or
   skipped data. This is the property the Phase 0 tests lock down.
3. **The measuring instrument comes before the models.** Answer extraction and eval are
   built and unit-tested first, because a subtle bug there silently corrupts every accuracy
   number in the study.
4. **Config over code.** Dataset IDs, prompt format, budgets, and seeds live in YAML so the
   controlled comparison never depends on an edit buried in a `.py` file.

---

## Phase 0 — Scaffolding & session-safe trainer

Goal: prove the training harness (checkpoint, resume, determinism, wall-clock guard) on a
trivial task so it can be trusted before spending GPU time.

### `requirements.txt`
Core stack: `torch`, `transformers`, `datasets`, `peft`, `accelerate` (modeling + LoRA +
device handling for later phases), plus `pyyaml`, `numpy`, `tqdm`, and `pytest`. Versions
are lower-bounded, not pinned, to stay compatible with Kaggle's preinstalled CUDA stack.

### `.gitignore`
Ignores venvs, `__pycache__`, and — importantly — `outputs/`, `checkpoints/`, `*.pt`, and
HF caches. Training artifacts and model weights never enter git; datasets are staged as
Kaggle Datasets instead.

### `configs/phase0.yaml`
The smoke config: `run_name`, `seed`, `offline` (toggles HF offline env vars),
`output_dir`, a `train` block (`total_steps`, `batch_size`, `lr`, `ckpt_every`,
`keep_last`, `max_seconds`), and a `dummy` block for the synthetic task. `max_seconds:
32400` is the ~9h Kaggle guard.

### `src/utils/config.py`
- **`Config`** — a `dict` subclass giving recursive attribute access (`cfg.train.lr`) while
  staying a plain dict for serialization. `.to_dict()` returns a clean nested dict for
  checkpoint metadata / logging.
- **`load_config(path, overrides)`** — loads YAML, then applies CLI dot-overrides like
  `train.lr=0.02`. `_coerce` uses `ast.literal_eval` so `0.02`→float, `true`→bool,
  `[1,2]`→list; `_apply_override` walks/creates the nested path. This is how you sweep a
  hyperparameter from the command line without editing YAML.

### `src/utils/seeding.py`
- **`set_seed(seed, deterministic=True)`** — seeds Python / NumPy / Torch (+CUDA) and pins
  cuDNN determinism. Determinism is a *feature* here because seed variance is a reported
  quantity in the study (§6).
- **`rng_state()` / `load_rng_state()`** — capture and restore all RNG states so a resumed
  run continues identically. Belt-and-suspenders alongside step-deterministic batching.

### `src/utils/time_budget.py`
- **`TimeBudget(max_seconds, safety_margin=0.05)`** — computes a monotonic-clock deadline
  set 5% early so there's time to flush a final checkpoint before the session is killed.
  `should_stop()` is polled each step.

### `src/utils/checkpoint.py`
- **`Checkpointer(output_dir, keep_last)`** — checkpoints live in
  `output_dir/checkpoints/step_XXXXXXXX.pt`.
  - **`save`** writes to a `.tmp` file then `replace()`s it into place — **atomic**, so a
    crash mid-write can never leave a corrupt "latest" checkpoint. Then prunes to
    `keep_last` (rolling window, bounds disk on ephemeral Kaggle storage).
  - **`latest_step` / `load_latest`** — find and load the newest checkpoint; the basis of
    idempotent resume.
- **`clear_checkpoints`** — used by tests / fresh runs.

### `src/train/kaggle_run.py`
The session-safe entrypoint and (currently) the training loop.
- **`_set_offline`** — sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` when `offline: true`,
  so Kaggle never blocks on network.
- **`DummyTask`** — synthetic linear-regression task. `batch(step)` seeds a generator with
  the **step index**, making every batch a deterministic function of the step — the crux of
  resume continuity.
- **`build_task(cfg)`** — returns `(model, optimizer, step_fn)`. This is the seam later
  phases swap: Phase 1b returns a real GPT-2 + SFT step, but the loop around it is unchanged.
- **`run(cfg)`** — the reusable loop: set offline/seed → build task → **resume from latest
  checkpoint if present** → step from `start_step` to `total_steps`, checkpointing on
  cadence, and **saving + exiting early if the time budget is hit**. Returns
  `(per-step losses, exit_code)`.
- **Exit codes** — `EXIT_COMPLETE=0` (reached `total_steps`), `EXIT_RESUME_NEEDED=42` (hit
  wall-clock; re-run to resume). Kaggle automation can key off `42` to relaunch.

### `tests/test_phase0.py`
The Phase 0 gate:
- **`test_determinism`** — two fresh runs, same seed → identical loss trajectories.
- **`test_resume_has_no_discontinuity`** — train 10 steps, "resume" to 20, and assert the
  losses equal an uninterrupted 20-step run **bit-for-bit**. This is the property that makes
  surviving Kaggle session caps safe.
- **`test_task_actually_learns`** — sanity check that the loop is wired (loss decreases).

---

## Phase 1a — Data, answer extraction & prompts (the measuring instrument)

Goal: build and unit-test the scoring/eval foundation before any real training, since a bug
here corrupts every downstream number. All CPU-only, no downloads.

### `configs/data.yaml`
Single source of truth for data:
- **`train`** — `eq_only` → `zen-E/GSM8k-Aug` (equation-only CoT) and `natural_language` →
  `zen-E/GSM8k-Aug-NL` (natural-language CoT): the two trace styles the study contrasts
  (§5.2). `fields` maps source columns (`question`/`cot`/`answer`) to the canonical schema.
- **`eval`** — four sets, each with `hf_id`, `split`, and a `kind` tag: `gsm8k` (in-domain),
  `svamp`, `multiarith`, `gsm_hard` (OOD). `kind` drives both the adapter and gold
  normalization.
- **`prompt`** — `question_prefix`, `cot_prefix`, `answer_prefix: "The answer is:"`. The
  trailing `:` is deliberate — it is CODI's distillation token position in Phase 2, so
  prompt format and loss target stay coupled.

### `src/data/answer_extract.py`
The scoring core, dependency-free and heavily unit-tested.
- **`normalize_number(token)`** — turns `"1,234"`, `"$50"`, `"3.0"`, `"12%"` into exact
  `Decimal` values;
  returns `None` for junk.
- **`extract_final_number(text)`** — parses a model generation into its answer. Strategy:
  **prefer the number right after an answer cue** (`"answer is: N"`), else **fall back to
  the last number** (final answers come last in a CoT). `_NUMBER_RE` matches signed/decimal/
  comma-grouped/`$`-prefixed numbers.
- **`normalize_gold(raw, kind)`** — normalizes heterogeneous gold answers: for `gsm8k_main`
  it strips everything before `####`; for the others it parses a bare number.
- **`answers_match(pred_text, gold, tol)`** — performs exact numeric comparison with
  `Decimal`, so large GSM-Hard answers cannot receive a magnitude-scaled tolerance. An
  optional absolute tolerance exists only for explicit diagnostics.

### `src/data/prompts.py`
Shared formatting so inputs are **byte-identical across methods** (a §5.3 fairness control).
- **`PromptStyle`** — frozen dataclass of the three prefixes; `from_config` builds it from
  `configs/data.yaml`.
- **`eval_prompt`** — prompt fed at eval time; ends exactly at the answer cue (no answer).
- **`cot_eval_prompt`** — variant that invites an explicit CoT (CoT-SFT / teacher).
- **`sft_target`** — the full supervised string. No-CoT: `Question: …\nThe answer is: <a>`;
  CoT: `Question: …\n<cot> The answer is: <a>`.
- **`answer_span`** — splits the target into `(prefix, completion)` at the cue, so a trainer
  can optionally mask the loss to answer tokens only.

### `src/data/datasets.py`
Normalizes every dataset into a common shape so the eval harness is dataset-agnostic.
- **`_row(d, *keys)`** — returns the first present column among candidates (schemas differ
  across mirrors); fails loudly if none match.
- **Adapters** (`_adapt_gsm8k_main`, `_adapt_svamp`, `_adapt_multiarith`, `_adapt_gsm_hard`)
  — each maps a raw HF row to `(question_text, gold_raw)`. SVAMP concatenates `Body` +
  `Question`; GSM-Hard reads `input`/`target`; etc. `ADAPTERS` is the `kind → adapter`
  registry.
- **`load_eval_set(name, spec)`** — loads a split, adapts each row, and normalizes gold.
  Empty questions, schema mismatches, and unparseable gold answers fail loudly rather than
  silently changing the evaluation denominator. Returns `[{question, gold}]`.
- **`load_train_set(data_cfg, trace_style)`** — loads a training split and renames columns
  to the canonical `question`/`cot`/`answer` schema.

### `tests/test_answer_extract.py` & `tests/test_prompts.py`
The Phase 1a gate: parametrized coverage of number normalization, answer extraction
(including "cue beats trailing number"), per-`kind` gold parsing, tolerant matching, prompt
formatting, `answer_span` round-tripping, and each dataset adapter — all without downloads.

---

## Phase 1b — Generic trainer + real SFT baselines + eval

Goal: turn the Phase 0 harness into a real GPT-2 fine-tuner, implement the two non-latent
baselines, and build the eval that scores them — the reproduction gate before any latent
method. All code is CPU-verified; the actual training runs on Kaggle GPU.

### `src/train/trainer.py`
The generic loop, extracted from `kaggle_run.py` so every method reuses it.
- **`Trainer(cfg, model, optimizer, step_fn)`** — owns resume, checkpoint cadence, the
  wall-clock guard, logging, and the exit-code contract. It infers the device from the
  model, so the same loop drives the CPU dummy task and a GPU GPT-2 unchanged.
- **`fit()`** — resume from latest checkpoint (loading model/optimizer/RNG state), then step
  `start_step → total_steps`, checkpointing every `ckpt_every` and on completion, or saving
  + returning `EXIT_RESUME_NEEDED` (42) if the budget is hit. `EXIT_COMPLETE`/
  `EXIT_RESUME_NEEDED` now live here and are re-exported from `kaggle_run` (tests unchanged).
- **Run provenance** — `run_manifest.json` records the effective and resume-critical config,
  data config, executable-source hash, git state, and dependency versions. Checkpoints carry
  the manifest fingerprint; changed science settings or source code require a new output
  directory, while runtime controls such as a larger `total_steps` may be extended safely.

### `src/train/kaggle_run.py` (refactored)
Now a thin **dispatcher**: `build_task(cfg)` reads `cfg.task` (`dummy` → `sft` → `latent`)
and returns `(model, optimizer, step_fn)`; `run(cfg)` wires it into a
`Trainer`. The Phase 0 `DummyTask` path is byte-for-byte unchanged, so its determinism /
resume tests still pass.

### `src/train/batching.py`
The step-deterministic data machinery, dependency-free (stdlib `random` only) so it is
unit-testable anywhere.
- **`StepBatcher(num_examples, batch_size, seed)`** — `batch_indices(step)` maps a step to
  example indices via a per-epoch permutation. It is a **pure function of the step**, so a
  resumed run draws exactly the batches it would have without interruption — no duplicated
  or skipped data. This extends the Phase 0 resume guarantee to real data.
- **`build_labels(input_ids, prompt_len)`** — masks the first `prompt_len` tokens with
  `-100` so cross-entropy is computed on the completion only.

### `src/train/sft.py`
The SFT task builder for both baselines (§5.4). Same backbone, tokenizer, prompt format, and
batcher for both — they differ **only** in the target:
- **`texts_for(row, method, style)`** — for `cot_sft` the prompt is `Question: …\n` and the
  completion is `<cot> The answer is: <ans>` (the model learns to produce the cue + CoT);
  for `nocot_sft` the prompt is `Question: …\nThe answer is:` and the completion is just the
  answer.
- **`build_sft_task(cfg)`** — loads GPT-2 + tokenizer (pad = eos), the normalized training
  set, an `AdamW` optimizer, and a `StepBatcher`. Returns a `step_fn` that: sets a warmup/
  constant LR (a **pure function of step**, so no scheduler state to checkpoint), gathers the
  step's rows, collates with prompt-masked labels + attention mask + left-padding-free
  packing, and runs a next-token CE step (HuggingFace computes the shifted loss and ignores
  `-100`). Runs single-GPU by default to keep resume exact and deterministic.
- **Sequence safety** — the prompt, full answer span, and EOS are always retained. If a CoT
  exceeds `max_length`, only its reasoning portion is shortened and the truncation rate is
  logged. An example that cannot retain prompt + answer fails rather than training on a
  partial target.

### `src/eval/run_eval.py`
The scoring harness (§6), dataset-agnostic thanks to the Phase 1a adapters.
- **`load_eval_model(cfg)`** — rebuilds the backbone, loads the latest checkpoint, sets
  `padding_side="left"` (correct batched decoder-only generation).
- **`generate_answers(...)`** — greedy decode (reproducible), returning only the newly
  generated tokens.
- **`evaluate(cfg, limit)`** — validates the evaluation config against the training manifest,
  picks the prompt by method, generates per eval set, and reports numeric exact-match. It
  writes every question/gold/generation/correctness record plus a JSON summary under
  `eval/step_XXXXXXXX/`. `--limit` caps examples for a fast sanity pass.

### `scripts/validate_phase1.py`
Preflight for the real Hugging Face artifacts: validates model/tokenizer resolution and
dataset schemas, samples answer parseability and sequence construction, measures reasoning
truncation, reports the effective epoch count, and rejects exact train/eval question overlap.

### `configs/sft_cot.yaml`, `configs/sft_nocot.yaml`
The two baseline runs: `task.method` (`cot_sft`/`nocot_sft`), `backbone: gpt2`,
`trace_style`, `max_length`, training budget (`epochs` or an explicit `total_steps`,
`batch_size`, `lr`, warmup,
`ckpt_every`, `max_seconds`), and an `eval` block. `offline: false` for the first Kaggle run
(direct HF download); flip to `true` with staged data. `data_config` points at
`configs/data.yaml`.

### `scripts/dataset_prep.py`
Populates a local HF cache (`hf_cache/`) with the pinned GPT-2 artifacts and also writes
normalized `save_to_disk` datasets beneath `hf_cache/prepared/`. Upload it as a Kaggle
Dataset and set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `CODIKAVA_DATA_ROOT=<HF_HOME>/prepared`.
The explicit prepared paths avoid relying on Hugging Face's private cache-key derivation.

### `notebooks/kaggle_phase1_sft.ipynb`
Clone/pull → install → (optional offline data) → **train CoT-SFT** (re-runnable, resumes) →
**evaluate** → optional No-CoT-SFT, with an explicit resume-past-the-session-cap recipe.

### `tests/test_batching.py`
CPU gate for the new machinery: batcher determinism, seed sensitivity, epoch-is-a-
permutation (full coverage, no repeats), resume-independence (indices are pure in `step`),
and label masking incl. clamping.

**Phase 1 gate result:** the saved CoT-SFT checkpoint completed 24,102 steps and was
evaluated with the shared exact-match harness; Phase 2 implementation then proceeded.

---

## Phase 2 — Shared latent model, CODI, and KaVa

The primary comparison fixes GPT-2, tokenizer, data, `M=6`, autoregressive latent updates,
optimizer, steps, and greedy decoding. `configs/codi.yaml` and `configs/kava.yaml` pass an
automated peer-config equality check over those controlled fields. They use separate output
directories and differ scientifically only by the additional KaVa KV trajectory target.

### `src/data/teacher_cache.py`

- Adds the matched teacher/student sequence contract and records every token boundary.
- Removes the final explicit reasoning step before tokenization, then truncates only the
  remaining trace if required; question, answer, and EOS are never discarded.
- Extracts detached all-block hidden states at the answer-cue colon plus explicit-CoT keys,
  values, masks, and answer-to-trace attention importance.
- Supports both Transformers v4 tuple caches and v5 `DynamicCache` objects.

### `src/models/latent_lm.py`

- Registers `<bot>` and `<eot>`, resizes the backbone, and adds CODI's two-layer GELU +
  LayerNorm projection.
- Autoregressive mode repeatedly feeds the projected final activation as the next
  continuous token. A Jacobi fixed-point mode is available only as an explicitly configured
  ablation.
- Returns answer CE, all-layer endpoint states, every latent hidden state, and the exact
  all-layer/head latent K/V trajectory. Evaluation uses this same path before greedy answer
  decoding; it cannot silently use ordinary text generation.

### `src/losses/trajectory_match.py`

- CODI: L1 hidden matching at the answer-cue endpoint over all blocks, teacher detached,
  normalized by each teacher layer's current-batch standard deviation.
- KaVa: CODI plus masked key/value trajectory matching over all layers, heads, and latent
  positions. Layer subsets and L1/MSE/SmoothL1 metrics are configurable for later continuum
  experiments.

### `src/losses/kv_compress.py`

Implements R-KV (`0.1 × importance + 0.9 × non-redundancy`) independently per layer/head.
Top-scoring tokens are returned to chronological order before matching student steps.
Short or empty traces are padded and masked, never treated as real zero-valued targets.
Uniform and seeded-random selectors provide the compression-quality ablations.

### `src/train/latent.py` and evaluation integration

The generic trainer now dispatches `task.type: latent`. Each step performs a detached
explicit-teacher target pass, a teacher CoT+answer CE pass, and the continuous student pass;
the shared model receives both CE gradients plus the configured trajectory terms. LR is a
step-derived warmup/cosine schedule, preserving resume determinism. `run_eval.py` rebuilds
the wrapper from the run manifest, loads its checkpoint, executes latent inference, and
writes the same exact-match artifacts as the SFT baselines.

### Phase 2 validation and tests

`scripts/validate_phase2.py` checks real tokenizer/data sequence construction, special and
colon token identities, trace-length/truncation statistics, context limits, method/loss
settings, and controlled equality against the peer config. The current 512-example
preflight passes for both methods with no truncation or construction failures.

The complete CPU suite has 78 passing tests. Phase 2 coverage includes boundary/masking
checks, explicit teacher extraction from a real tiny causal LM, R-KV shape/order/masks,
deterministic random compression, stop-gradient and layer selection, autoregressive and
Jacobi forward contracts, latent determinism, projection backpropagation, and one-batch
overfitting.

### GPU checkpoint RNG compatibility

`scripts/resume_training.py` is the Phase-2 GPU entrypoint. It handles checkpoints created
by the original trainer, whose `map_location='cuda'` also moved the saved CPU RNG byte tensor
to CUDA. The wrapper converts only RNG state tensors back to CPU before restoration; model,
optimizer, loss history, batch index, and experiment fingerprint remain unchanged. It lives
outside the provenance-hashed `src/` tree specifically so already-running experiments can
adopt the resume fix without invalidating their manifests.
