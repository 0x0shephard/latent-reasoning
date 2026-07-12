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
| **Phase 1b** | Generalize trainer for real HF SFT; No-CoT / CoT-SFT baselines; `run_eval.py`; Kaggle notebook | ⏳ **Next** |
| **Phase 2** | Latent-LM (`<bot>`/`<eot>` continuous thoughts) + CODI & KaVa losses + R-KV compression | ◻ Not started |
| **Phase 3** | Mechanistic analysis (probes, CKA/SVCCA, ablation, patching) | ◻ Not started |
| **Phase 4** | Supervision-granularity continuum + writing | ◻ Not started |

**Current milestone:** the local, CPU-only foundation is done and trustworthy. The next
step (1b) is the first code that requires a GPU and is therefore the first thing to run on
Kaggle. Nothing before Phase 1b needs Kaggle.

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
- **`normalize_number(token)`** — turns `"1,234"`, `"$50"`, `"3.0"`, `"12%"` into floats;
  returns `None` for junk.
- **`extract_final_number(text)`** — parses a model generation into its answer. Strategy:
  **prefer the number right after an answer cue** (`"answer is: N"`), else **fall back to
  the last number** (final answers come last in a CoT). `_NUMBER_RE` matches signed/decimal/
  comma-grouped/`$`-prefixed numbers.
- **`normalize_gold(raw, kind)`** — normalizes heterogeneous gold answers: for `gsm8k_main`
  it strips everything before `####`; for the others it parses a bare number.
- **`answers_match(pred_text, gold, tol)`** — extracts the prediction and compares to gold
  within a relative tolerance (robust to `360` vs `360.00004`).

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
- **`load_eval_set(name, spec)`** — loads a split, adapts each row, normalizes gold, and
  **skips rows whose gold can't be parsed** (rather than silently miscounting). Returns
  `[{question, gold}]`. `load_dataset` is imported lazily so importing this module needs no
  network — that's why the unit tests can exercise the adapters offline.
- **`load_train_set(data_cfg, trace_style)`** — loads a training split and renames columns
  to the canonical `question`/`cot`/`answer` schema.

### `tests/test_answer_extract.py` & `tests/test_prompts.py`
The Phase 1a gate: parametrized coverage of number normalization, answer extraction
(including "cue beats trailing number"), per-`kind` gold parsing, tolerant matching, prompt
formatting, `answer_span` round-tripping, and each dataset adapter — all without downloads.

---

## What Phase 1b will add (next)

- **`src/train/trainer.py`** — extract the generic loop out of `kaggle_run.py` into a
  reusable `Trainer`, with `kaggle_run` dispatching on `cfg.task` (`dummy` → `sft` → later
  `codi`/`kava`). The Phase 0 dummy path stays numerically identical so its tests keep
  passing.
- **`build_task` for SFT** — real GPT-2 + tokenizer (via `transformers`), a
  step-deterministic dataloader over `load_train_set`, and a next-token CE `step_fn`.
  Implements the **No-CoT-SFT** and **CoT-SFT** baselines (§5.4).
- **`src/eval/run_eval.py`** — batched generation → `answers_match`, reporting exact-match
  overall and broken down by step-count / trace-style, across in-domain + OOD sets (§6).
- **`scripts/dataset_prep.py`** — flesh out offline staging of GPT-2 + datasets as a Kaggle
  Dataset.
- **Kaggle notebook** — runs the **CoT-SFT reproduction gate**: sane GSM8k accuracy proves
  the data/tokenizer/prompt/eval pipeline is trustworthy before any latent method.

**Gate to exit Phase 1:** CoT-SFT on GPT-2 reaches sane GSM8k exact-match, and the eval
harness reproduces the same number across two runs on a fixed checkpoint.
