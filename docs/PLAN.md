# Plan: CODI vs KaVa — Controlled & Mechanistic Study of Latent Reasoning Supervision

## Context

The repo (`CODI:KAVA`) currently holds only the proposal PDF
(`KAVA_vs_CODI_SProj_Research_Proposal.pdf`) — no code. The goal is to build, from
scratch, a **single shared training/eval harness** in which two latent-reasoning
supervision methods differ *only* in their distillation loss, so the study isolates the
effect of **supervision granularity** rather than comparing two unrelated codebases
(which would reintroduce every confound in the proposal's §9).

The two methods are real, published, and — importantly — **nested**:

- **CODI** (arXiv 2502.21074, EMNLP 2025): self-distillation on GPT-2. 6 autoregressive
  continuous "thought" tokens between `<bot>`/`<eot>`. Distillation loss = **L1 between
  teacher and student hidden states at the `":"` token** of "The answer is:", summed over
  **all layers**, each layer normalized by the teacher hidden-state std within the batch,
  stop-gradient on the teacher. Total: `L = α·L_student_CE + β·L_KD + γ·L_teacher_CE`
  (α=β=γ=1 on GPT-2).
- **KaVa** (arXiv 2510.02312, Oct 2025): `L_KaVa = student_CE + teacher_CE + α₁·L_CODI +
  α₂·L_KV`, i.e. **CODI plus a KV-trajectory loss**. `L_KV = (1/2M)(‖sg[K̃_t]−K_s‖_p^p +
  ‖sg[Ṽ_t]−V_s‖_p^p)` over all layers/heads, L1 default. Teacher KV-cache is compressed
  from CoT-length to M latent slots via **R-KV** redundancy-aware eviction (Cai et al.
  2025; score `S = λI + (1−λ)R`, λ=0.1). KaVa uses M=24 latent tokens produced by **Jacobi
  iteration** (T=3), on LLaMA3.2-1B/3B and Qwen2.5-0.5B with LoRA r=128.

**Key consequence for design:** since KaVa = CODI + KV loss, the proposal's §8 continuum
(final hidden state → all-layer hidden → selected latent states → values-only → keys-only
→ keys+values@selected-layers → full KV trajectory) is *one configurable distillation
module*, not seven implementations.

**Decisions locked with the user:** clean shared harness; full proposal scope incl. the
§8 continuum; compute = Kaggle **T4×2 / P100** → GPT-2 is the primary backbone, Qwen2.5-0.5B
+LoRA is an optional "scales to larger backbone" arm only after the pipeline works.

## Guiding technical decisions

1. **Losses are composable modules over one identical student forward pass.** A single
   `TrajectoryMatchLoss` parameterized by `(target ∈ {hidden,key,value}, layers, positions
   ∈ {endpoint, all_latent_steps})` yields CODI (hidden/all-layers/endpoint), KaVa
   (key+value/all-layers/all-steps), and every §8 midpoint. This *is* the core abstraction.
2. **Control the latent-generation mechanism.** CODI = autoregressive continuous thoughts;
   KaVa = Jacobi (T=3). These are an architecture confound. For the primary controlled
   comparison, fix ONE mechanism for both (default: **autoregressive**, GPT-2-proven,
   simpler), and treat Jacobi as a separate, explicitly-labeled ablation arm — never mix
   them silently in a single comparison table.
3. **Match the latent budget.** CODI uses M=6, KaVa M=24. Pick a single default (**M=6**
   for the GPT-2 reproduction), then sweep M ∈ {6, 12, 24} as a controlled variable. Log M
   as first-class config; never compare across different M in a headline result.
4. **Teacher = same backbone (self-distillation).** Run the frozen backbone on the explicit
   CoT, cache (a) per-layer hidden states at the `":"` position, and (b) full per-layer/head
   KV. Compress the KV once via R-KV. Stop-gradient on all teacher tensors.
5. **Kaggle robustness is a first-class requirement, not an afterthought.** 9–12h session
   caps + ephemeral disk → the trainer must checkpoint every N steps and resume idempotently;
   datasets/models are pre-cached as Kaggle Datasets and loaded in `HF_HUB_OFFLINE=1` mode;
   training is chunked to fit a session.

## Proposed repo structure

```
configs/                  # YAML: backbone, data, method, latent budget, seeds, kaggle
src/
  data/
    datasets.py           # GSM8k-Aug / -Aug-NL train; GSM8k/SVAMP/MultiArith/GSM-Hard eval
    teacher_cache.py      # explicit-CoT forward pass -> hidden@":" + per-layer/head KV
    answer_extract.py     # shared answer parsing + exact-match (the "measuring instrument")
  models/
    latent_lm.py          # HF causal LM + <bot>/<eot> + continuous-thought block (AR|Jacobi)
  losses/
    trajectory_match.py   # the one configurable distillation module (§8 continuum)
    kv_compress.py        # R-KV redundancy-aware eviction (CoT-len -> M)
    registry.py           # name -> loss config: no_cot, cot_sft, codi, kava, continuum_*
  train/
    trainer.py            # shared loop; checkpoint/resume; loss = sum of registered terms
    kaggle_run.py         # session-safe entrypoint (offline, resume, time-budget guard)
  eval/
    run_eval.py           # exact-match, by-step-count, by-trace-style, in-domain + OOD
    metrics.py            # efficiency, calibration (ECE), robustness perturbations
  mech/
    probes.py             # linear probes on latent states (§7.1)
    similarity.py         # CKA / SVCCA teacher-vs-student (§7.2)
    ablation.py           # zero/replace/shuffle latent states (§7.3); patching (§7.4)
tests/                    # shape tests, overfit-one-batch, determinism, kv-compress sanity
scripts/                  # dataset-prep, launch, aggregate-results
README.md                 # this plan, condensed, + how to run on Kaggle
```

## Phased execution (maps to proposal §10)

### Phase 0 — Scaffolding & Kaggle harness (before any modeling)
- Repo skeleton above; `requirements.txt` (PyTorch, `transformers`, `datasets`, `peft`,
  `accelerate`, `pyyaml`); deterministic seeding util; simple YAML config loader.
- `train/kaggle_run.py`: offline HF mode, checkpoint-every-N-steps to `/kaggle/working`,
  resume-from-latest, and a wall-clock guard that saves + exits cleanly before the cap.
- Pre-cache GPT-2 + GSM8k-Aug/-Aug-NL + eval sets as Kaggle Datasets; script the download
  locally (`scripts/dataset-prep`).
- **Deliverable:** a no-op training loop that checkpoints, dies, and resumes correctly.

### Phase 1 — Data + eval + non-latent baselines (reproduction milestone, §10 wk1–4)
- `data/datasets.py`, `data/answer_extract.py`, `eval/run_eval.py` wired to all four eval
  sets (GSM8k in-domain; SVAMP/MultiArith/GSM-Hard OOD).
- Baselines from §5.4 that need **no** latent block yet: **No-CoT-SFT** and **CoT-SFT**.
- **Gate:** CoT-SFT reaches sane GSM8k accuracy → data, tokenizer, prompt, and eval are
  trustworthy. Do not proceed to latent methods until this passes. This de-risks 90% of
  reproduction failure modes.

### Phase 2 — Latent block + CODI + KaVa (controlled comparison, §10 wk5–8)
- `models/latent_lm.py`: `<bot>`/`<eot>`, M continuous thoughts (autoregressive default;
  Jacobi T=3 behind a flag), answer decoded from `<eot>`. Training drops the final CoT step
  (CODI's anti-shortcut trick).
- `data/teacher_cache.py`: teacher forward on explicit CoT → hidden@`":"` (all layers) + full
  KV; `losses/kv_compress.py`: R-KV → M slots.
- `losses/trajectory_match.py`: the configurable module. Register **CODI** (hidden/all/
  endpoint, L1, std-normalized, stop-grad) and **KaVa** (=CODI + key&value/all/all-steps).
- Additional §5.4 baselines: latent-student-no-distillation; random-KV-targets; uniform-KV-
  targets (isolates *meaningful compression* from *mere target density* — directly answers
  §9 "unequal loss density").
- Run matched: same backbone/tokenizer/M/optimizer/steps/decoding, **≥3 seeds**. Produce the
  §6 table: accuracy (+ by-step-count, by-trace-style, in-domain/OOD), efficiency, optimization
  stability, robustness, calibration. Both trace styles (GSM8k-Aug eq-only + -Aug-NL).

### Phase 3 — Mechanistic analysis (§10 wk9–12, §7)
- `mech/probes.py`: linear probes on each latent state → intermediate result / op-type /
  final answer / remaining depth.
- `mech/similarity.py`: CKA/SVCCA of explicit-CoT teacher states vs CODI vs KaVa latents.
- `mech/ablation.py`: zero/replace/shuffle latent states; activation patching (swap a latent
  from a wrong run into a correct run) to locate decisive positions.
- Pick **one** as the primary analysis per §10; others as time allows.

### Phase 4 — Supervision-granularity continuum + writing (§10 wk13–16, §8)
- Sweep `TrajectoryMatchLoss` configs along the continuum: final-hidden → all-layer-hidden
  (CODI) → selected latent states → values-only → keys-only → K+V@selected-layers → full KV
  (KaVa). Plot accuracy / cost vs granularity to find the **point of diminishing returns** —
  the paper's headline result.
- Optional: repeat the core comparison on Qwen2.5-0.5B+LoRA (§5.1 optional backbone) to test
  the "scales to larger backbone" claim, only if T4×2 time permits.
- Consolidate: SProj report, poster, reproducible eval scripts, seed-variance error bars.

## Verification (how each phase is checked end-to-end)

- **Unit / sanity (continuous, in `tests/`):**
  - Shape tests for `trajectory_match` across every `(target, layers, positions)` config.
  - **Overfit-one-batch**: each method drives train loss → ~0 on a single batch (catches
    wiring/masking bugs before any long run).
  - Determinism: same seed → identical loss for K steps.
  - `kv_compress` sanity: output has exactly M slots per layer/head; random-KV baseline is
    measurably worse than R-KV (confirms compression *quality* matters, not just density).
- **Phase 1 gate:** CoT-SFT GPT-2 hits sane GSM8k exact-match; eval harness reproduces the
  same number on a fixed checkpoint across two runs.
- **Phase 2:** CODI reproduces roughly its reported GPT-2 GSM8k behavior; KaVa (=CODI+KV) is
  compared under identical M/backbone/seeds; results table auto-generated by `run_eval.py`
  with ≥3-seed error bars.
- **Kaggle:** kill a session mid-training and confirm `kaggle_run.py` resumes from the last
  checkpoint with no metric discontinuity.
- **Mechanistic/continuum:** probe accuracy and CKA reported with seed variance; continuum
  plot shows a monotone-or-plateauing trend that a reader can interpret.

## Open risks (tracked, not blocking)

- GSM8k-Aug / -Aug-NL sourcing — confirm exact release from the CODI/KaVa authors vs
  regenerating; pin the version.
- R-KV eviction is the most involved single component; budget extra time and unit-test it in
  isolation before wiring into training.
- Full scope on T4×2 is ambitious — Phases 1–2 are the must-ship core; Phases 3–4 are the
  research contribution but can be scoped down (one mechanistic analysis, partial continuum)
  if compute/time run short.

## References

- CODI — *Compressing Chain-of-Thought into Continuous Space via Self-Distillation*,
  arXiv 2502.21074 (EMNLP 2025).
- KaVa — *Latent Reasoning via Compressed KV-Cache Distillation*, arXiv 2510.02312 (2025).
- R-KV — redundancy-aware KV eviction, Cai et al. 2025 (used by KaVa for teacher-cache
  compression).
