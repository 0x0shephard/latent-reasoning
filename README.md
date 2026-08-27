# CODI vs KaVa — Controlled & Mechanistic Study of Latent Reasoning Supervision

A single shared harness in which two latent-reasoning supervision methods differ *only*
in their distillation loss, so the study isolates the effect of **supervision
granularity**. See [`docs/PLAN.md`](docs/PLAN.md) for the full research plan and
[`KAVA_vs_CODI_SProj_Research_Proposal.pdf`](docs/KAVA_vs_CODI_SProj_Research_Proposal.pdf) for
the proposal.

Before extending the study, read the
[research context ledger](docs/RESEARCH_CONTEXT_LEDGER.md). It records the original
question, instructor criticism, TSV-inspired pivot, completed gates, negative
spectral-causality result, and current decision point.

The current corrective follow-up is the
[test-like correctness detector replication](docs/OFFICIAL_CODI_CORRECTNESS_DETECT_REPLICATION.md).
The [correct-versus-wrong covariance intervention](docs/OFFICIAL_CODI_CORRECTNESS_CONTRASTIVE_COVARIANCE.md)
is **complete and `not_confirmed`** (ledger §47): the 28 correct-specific directions of
`C_correct v = λ C_wrong v` are genuinely class-specific but retain only 0.121 accuracy
against the PCA controls' 0.321, and removing the 28 wrong-specific directions changes
exactly nothing. Correctness-conditioned covariance is descriptive, not a correction
channel. Neither follow-up changes the completed steer and project results.

- **CODI** (arXiv 2502.21074): endpoint hidden-state distillation.
- **KaVa** (arXiv 2510.02312): CODI **+** compressed KV-trajectory distillation.

## Status

**The primary seed for Phases 0–2 is trained and the capped Phase-3 causal gate is complete.**
The completed checkpoints are CoT-SFT step 24,102 and matched CODI/KaVa step 96,405. Their
saved capped evaluations (200 examples per set, all 180 MultiArith examples) are:

| Method | GSM8K | SVAMP | MultiArith | GSM-Hard | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| CoT-SFT | 26.0% | 30.0% | 74.44% | 5.0% | 33.86% |
| CODI | 15.5% | 9.5% | 18.89% | 3.0% | 11.72% |
| KaVa | 16.0% | 11.0% | 25.56% | 3.0% | 13.89% |

KaVa is +2.17 macro percentage points over CODI in this seed, but both latent methods are
below CoT-SFT. These are provisional quick-gate results, not the final research table:
full-set evaluation on one pinned dataset snapshot, the three Phase-2 control baselines,
and at least three seeds remain.

The matched Phase-3 interventions support continuing: KaVa is more sensitive than CODI
to batch-mean replacement (DID -2.31 points, 95% paired bootstrap CI -4.14 to -0.54)
and cross-example shuffling (DID -4.60 points, CI -6.85 to -2.42). Zeroing has the same
direction but is inconclusive at this gate (DID -2.03 points, CI -4.31 to +0.21).

## Setup

The project folder name contains a colon (`CODI:KAVA`), which `venv` rejects for an
in-tree `.venv` (`:` is the shell `PATH` separator). Create the venv **outside** the
colon path instead — it can live anywhere:

```bash
# from the repo root
python3 -m venv ~/.venvs/codikava
source ~/.venvs/codikava/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # run from the repo root
```

A colon in the current working directory is harmless; only a colon inside the venv's own
path breaks activation. (Alternatively, rename the folder to `CODI-KAVA` to avoid the
quirk entirely.)

## Run the Phase 0 smoke trainer

```bash
python -m src.train.kaggle_run --config configs/phase0.yaml
```

Kill it mid-run and re-run the same command — it resumes from the latest checkpoint with
no metric discontinuity. Exit codes: `0` = complete, `42` = hit the wall-clock budget
(state saved, re-run to resume).

## Tests

```bash
pytest -q
```

Covers deterministic resume, exact numeric scoring, prompt/data adapters, answer-preserving
SFT/latent collation, provenance guards, R-KV selection, stop-gradient trajectory losses,
both latent mechanisms, determinism, and tiny-model overfitting. CPU-only, no downloads.

## Validate the author-released CODI checkpoint

The local one-epoch CODI/KaVa runs are compute-limited pilots and are far below the
published CODI accuracy. Before extending the KV-subspace experiments, use the isolated
[official CODI validation notebook](notebooks/colab_official_codi_validation.ipynb). It
loads the pinned `zen-E/CODI-gpt2` weights with the released LoRA, projection, latent
generation, prompt, and benchmark protocol. It never overwrites the pilot outputs.

The first meaningful gate is the complete 1,319-example GSM8K evaluation:

```bash
python -u -m src.eval.official_codi \
  --config configs/official_codi_gpt2.yaml \
  --datasets gsm8k \
  --limit 0 \
  --device cuda
```

See [the official-checkpoint validation contract](docs/OFFICIAL_CODI_VALIDATION.md) for
the pinned revisions, expected benchmark counts, output paths, and interpretation rules.

After the complete GSM8K gate passes, use the
[official CODI KV subspace notebook](notebooks/colab_official_codi_kv_subspaces.ipynb)
to run the one-batch alignment audit, 2,000-example paired cross-subspace analysis, and
held-out reduced-rank test. Its exact teacher/student sequence reconstruction, R-KV
alignment, shuffled null, durable output paths, and interpretation boundary are in the
[official KV subspace contract](docs/OFFICIAL_CODI_KV_SUBSPACES.md).

After the independent 5,000-example reduced-rank result is recorded, use the
[official selector-specificity notebook](notebooks/colab_official_codi_selector_specificity.ipynb)
to compare R-KV with uniform and four seeded-random teacher-trace selectors in one
matched pass. The primary score subtracts each selector's own shuffled-pairing null
before comparing selectors. See the
[selector-specificity contract](docs/OFFICIAL_CODI_SELECTOR_SPECIFICITY.md).

If R-KV beats random selection but fails against uniform selection, use the
[boundary-aware selector notebook](notebooks/colab_official_codi_boundary_selector.ipynb)
for the preregistered confirmation. It excludes every example in the original
5,000-example selector collection, forces the first and last teacher trace tokens, and
uses R-KV for four interior targets. The exact gate and decision rule are in the
[boundary-selector contract](docs/OFFICIAL_CODI_BOUNDARY_SELECTOR.md).

If the selector gates do not establish a stronger token selector, stop selector design
and test the learned spectral directions directly with the
[official CODI KV causal notebook](notebooks/colab_official_codi_kv_causal.ipynb).
It performs centered rank-four retain and remove interventions on the frozen official
checkpoint, compares them with energy-matched random directions, and evaluates paired
full-GSM8K accuracy by latent position. Positions 4 and 5 are the multiplicity-corrected
primary tests. See the
[spectral-causality contract](docs/OFFICIAL_CODI_KV_CAUSAL.md).

When Colab GPU quota is unavailable, run the equivalent
[Kaggle causal notebook](notebooks/kaggle_official_codi_kv_causal.ipynb). Attach a
Kaggle input dataset containing the completed
`official_codi_kv_subspaces/n5000_seed1/statistics.pt`, enable Internet and a T4 GPU,
pin the repository commit, and use Save Version with outputs enabled. The current
Kaggle PyTorch 2.10 build does not support the P100 `sm_60` architecture. Its 29
conditions are condition-level resumable from a previously saved Kaggle output.

That full-GSM8K experiment is complete and did not establish greater causal value for
the learned rank-four directions than for energy-matched random directions. Before
implementing another subspace method, use the
[operational answer-causal signal definition](docs/ANSWER_CAUSAL_SIGNAL_DEFINITION.md).
It separates structural, predictive, answer-causal, student-accessible, and
transferable signal. It also defines a marginal distillation-utility screen that asks
whether each target family's matched update lowers held-out answer loss before another
spectral method is fitted.

The executable screen is
[`scripts/run_official_codi_kv_target_utility.py`](scripts/run_official_codi_kv_target_utility.py).
Begin with key-versus-value granularity and refine only helpful families by latent
position and layer band. The full update-matching, split, classification, resume, and
output contract is in
[`OFFICIAL_CODI_KV_TARGET_UTILITY.md`](docs/OFFICIAL_CODI_KV_TARGET_UTILITY.md).
Run the complete default screen on a Kaggle T4 with
[`kaggle_official_codi_kv_target_utility.ipynb`](notebooks/kaggle_official_codi_kv_target_utility.ipynb).
It creates the official full-GSM8K gate if no passed summary is attached, runs the smoke
and kind-level screens, resumes atomic batch outputs, and packages the results for
Kaggle Save Version.

That kind-level screen completed with neither pooled key nor pooled value targets
passing the predefined utility gate. The subsequent
[sparse answer-aligned KV gradient notebook](notebooks/kaggle_official_codi_kv_gradient_signal.ipynb)
also completed with a negative primary gate. Its five-percent mask retained
pairing-specific information relative to shuffled targets, but did not reproducibly
beat answer-only, full KV, random sparse, or complement updates. The complete protocol
and interpretation boundary are defined in
[`OFFICIAL_CODI_KV_GRADIENT_SIGNAL.md`](docs/OFFICIAL_CODI_KV_GRADIENT_SIGNAL.md).

## Validate and run the Phase 1 baseline

Before using a GPU, validate the real model/dataset contract (requires the Hugging Face
artifacts online or in the configured offline cache):

```bash
python scripts/validate_phase1.py --config configs/sft_cot.yaml
python -m src.train.kaggle_run --config configs/sft_cot.yaml
python -m src.eval.run_eval --config configs/sft_cot.yaml --limit 200
```

On Kaggle, add a read-only Hugging Face token as a notebook secret named `HF_TOKEN`.
The Phase-1 notebook loads it without printing the value; public downloads still work
without it, but authenticated requests avoid anonymous rate limits. Evaluation files use
their pinned source JSON/JSONL blobs instead of auto-converted Xet Parquet, avoiding CDN
signing failures without changing the configured benchmark splits.

Each training run records `run_manifest.json` with its immutable config, resolved artifact
identities, source hash, and package versions. Evaluation writes per-example predictions
and a summary under the run's `eval/step_XXXXXXXX/` directory.

## Validate and run Phase 2

CODI and KaVa use the same GPT-2 backbone, tokenizer, `M=6` autoregressive latent block,
optimizer, data, and decoding. The controlled difference is `distillation.kv_weight`:
CODI uses the all-layer hidden endpoint; KaVa adds all-layer/head key/value trajectories
compressed to six slots with R-KV.

```bash
python scripts/validate_phase2.py --config configs/codi.yaml --peer-config configs/kava.yaml
python -u scripts/resume_training.py --config configs/codi.yaml
python -m src.eval.run_eval --config configs/codi.yaml --limit 200

python scripts/validate_phase2.py --config configs/kava.yaml --peer-config configs/codi.yaml
python -u scripts/resume_training.py --config configs/kava.yaml
python -m src.eval.run_eval --config configs/kava.yaml --limit 200
```

Use [`notebooks/kaggle_phase2_latent.ipynb`](notebooks/kaggle_phase2_latent.ipynb) on a
Kaggle P100 or T4. Keep each method in its own output directory and re-run the same training
command after a wall-clock exit (code 42); the latest checkpoint resumes automatically.

### Resume Phase 2 on Colab with Google Drive

Use the fixed-purpose [`CODI Colab notebook`](notebooks/colab_phase2_codi.ipynb) and
[`KaVa Colab notebook`](notebooks/colab_phase2_kava.ipynb). Each notebook is locked to one
method and Drive output directory. They train from fast VM-local storage and mirror every
completed atomic checkpoint to `MyDrive/CODI_KAVA/outputs/`. Before the first session,
upload the complete extracted checkpoint folders:

```text
MyDrive/CODI_KAVA/uploads/step_00080000.pt/   # CODI
MyDrive/CODI_KAVA/uploads/step_00024000.pt/   # KaVa
```

Upload the entire folders, including `data.pkl`, `data/`, `byteorder`, `version`, and hidden
metadata files. The notebook restores the matching archived manifest, reconstructs and
PyTorch-verifies a real `.pt` archive, and uses `scripts/colab_runner.py` to audit a
Kaggle-to-Colab dependency change
while still requiring identical executable source, pinned data config, and scientific
settings. Training stays in an active cell; after the resume message appears, the browser
may be closed while Colab continues server-side. Drive persistence protects completed
checkpoints, but managed Colab runtimes can still terminate and must then be restarted.

## Close Phase 2 and run the Phase 3 causal test

The dedicated [`colab_phase3_ablations.ipynb`](notebooks/colab_phase3_ablations.ipynb)
contains this complete post-training workflow with safe defaults and Drive-persistent logs.
It is separate from both training notebooks and never updates model weights.

First produce a strict paired comparison of the existing capped results. The analyzer pairs
rows by exact question and
normalized gold answer, handles different row order, and refuses dataset-version drift:

```bash
python scripts/analyze_phase2.py \
  --run codi=/content/drive/MyDrive/CODI_KAVA/outputs/codi/eval/step_00096405 \
  --run kava=/content/drive/MyDrive/CODI_KAVA/outputs/kava/eval/step_00096405 \
  --output /content/drive/MyDrive/CODI_KAVA/reports/codi_vs_kava.json
```

The first scoped Phase-3 analysis causally changes each continuous state before it enters
the latent slot, so the intervention affects the cache, subsequent latent states, and
answer. Run all-position zeroing, batch-mean replacement, and deterministic cross-example
shuffling on the 200-example gate first:

```bash
python -u scripts/colab_ablation_runner.py --method codi --limit 200
python -u scripts/colab_ablation_runner.py --method kava --limit 200
```

Results persist below `eval/step_00096405/ablations/`. Use `--positions 0`, `--positions 5`,
or another comma-separated subset only after the all-position gate shows a meaningful
effect. A batch of one cannot support cross-example shuffling and is left unchanged.

Compare an ablation to its own unmodified checkpoint with the same paired analyzer:

```bash
python scripts/analyze_phase2.py \
  --run baseline=/content/drive/MyDrive/CODI_KAVA/outputs/kava/eval/step_00096405 \
  --run zero=/content/drive/MyDrive/CODI_KAVA/outputs/kava/eval/step_00096405/ablations/zero_all \
  --run mean=/content/drive/MyDrive/CODI_KAVA/outputs/kava/eval/step_00096405/ablations/batch_mean_all \
  --run shuffle=/content/drive/MyDrive/CODI_KAVA/outputs/kava/eval/step_00096405/ablations/batch_shuffle_all \
  --output /content/drive/MyDrive/CODI_KAVA/reports/kava_latent_ablation.json
```

For the next mechanistic gate, use the Phase-3 notebook's opt-in settings:

```python
RUN_KAVA_ABLATIONS = False
RUN_CODI_ABLATIONS = False
RUN_KAVA_POSITION_SWEEP = True
RUN_MATCHED_BATCH_DID = False
RUN_FULL_BASELINES = False
```

The position cell runs six single-position KaVa shuffle interventions, holds the realized
cross-example permutation fixed across positions, and writes
`kava_shuffle_position_sweep_limit200.{json,md}`. The DID cell first compares the existing
batch-independent zero interventions, then reruns a tagged baseline, mean, and shuffle for
both methods with matched evaluation batch size 8. Matched outputs receive `_bs8` tags, so the original
ablations are not overwritten. `analyze_intervention_effects.py` estimates
`KaVa_effect - CODI_effect` with a question-paired bootstrap; negative values mean KaVa is
more sensitive. This uncertainty is conditional on the two checkpoints and does not replace
additional training seeds.

Run the position cell first, then switch to `RUN_KAVA_POSITION_SWEEP = False` and
`RUN_MATCHED_BATCH_DID = True` for the direct comparison. Do not enable both expensive
cells in one pass.

## Localize accuracy-critical CODI endpoint directions

The frozen-checkpoint follow-up to the answer-colon ablation uses 100 method-specific
activation-energy-matched random controls and hierarchical state, single-direction, and
joint-minus-one tests. Its complete contract is documented in
[`docs/OFFICIAL_CODI_ENDPOINT_ACCURACY_LOCALIZATION.md`](docs/OFFICIAL_CODI_ENDPOINT_ACCURACY_LOCALIZATION.md),
and the resumable Kaggle workflow is
[`notebooks/kaggle_official_codi_endpoint_accuracy_localization.ipynb`](notebooks/kaggle_official_codi_endpoint_accuracy_localization.ipynb).

This experiment is a causal localization test at the forced answer-cue colon. It does
not update model weights and does not claim architectural inference speedup.

The single-hypothesis follow-up for the parameter-aware state-12 rank-three group is
frozen in
[`docs/OFFICIAL_CODI_PARAMETER_STATE12_CONFIRMATION.md`](docs/OFFICIAL_CODI_PARAMETER_STATE12_CONFIRMATION.md).
Its
[`Kaggle run-all notebook`](notebooks/kaggle_official_codi_parameter_state12_confirmation.ipynb)
uses disjoint GSM8K-train calibration, 500 matched controls, and an explicit evaluation
RMS transport gate.

That confirmation is **complete and negative**: `not_confirmed`. Five of six conditions
passed — the selected subspace cost 1.5163 accuracy points with a positive bootstrap
lower bound and McNemar `p = 0.00227` — but 77 of 500 energy-matched random subspaces
were at least as damaging, so the empirical matched-random test failed at `p = 0.1557`.
The evaluation RMS transport gate passed at ratio 1.0497, so the failure is not
explained by a conservative null. See ledger sections 33 and 34.

## Answer-colon margin geometry and effective dimensionality

Before proposing another selector, a source audit asked what the state-12 design could
detect at all. `ln_f` runs after every transformer block, so its output never enters the
key/value cache: a state-12 edit changes one token's logits and nothing propagates.
Combined with a binary outcome on 1,319 questions and selectors scored by first-order
gradients but tested with finite projections, the completed experiment could only ever
confirm a very large effect.

The follow-up removes each of those limits rather than adding a heuristic. Because
`lm_head` is bias-free and consumes the `ln_f` output, a state-12 edit is exactly
`z' = z - (W U)(Uᵀ(h - centre))`, so one cached colon state per question makes every
state-12 arm a matrix product instead of a full greedy decode. A parity gate checks the
analytic first token against the released decoder before any sweep runs.

The primary subspace is the closed-form maximiser of the measured objective — the top-`k`
eigenvectors of `sym(E[c gᵀ])` — rather than a gradient heuristic, the outcome is
per-example gold-answer NLL, retention arms sweep rank 1 → 512 for sufficiency, three
ablation semantics are separated, and state-11 and all-position arms cover the
propagating cases the analytic tier cannot reach.

```bash
python scripts/build_kaggle_official_codi_endpoint_margin_geometry_notebook.py
```

The frozen contract, gates, and interpretation boundary are in
[`OFFICIAL_CODI_ENDPOINT_MARGIN_GEOMETRY.md`](docs/OFFICIAL_CODI_ENDPOINT_MARGIN_GEOMETRY.md),
and the resumable workflow is
[`kaggle_official_codi_endpoint_margin_geometry.ipynb`](notebooks/kaggle_official_codi_endpoint_margin_geometry.ipynb).
No weight is updated and no inference speed is claimed.

After the capped ablation report is saved, optionally run a full baseline evaluation from
each checkpoint. This overwrites the root prediction JSONLs with the full benchmark, while
the capped ablation directories and report remain preserved:

```bash
python -u scripts/colab_ablation_runner.py --method codi --mode baseline --limit 0
python -u scripts/colab_ablation_runner.py --method kava --mode baseline --limit 0
```

## Close the remaining controls and seed variance

Before the Phase-4 supervision continuum, use
[`colab_controls_and_seeds.ipynb`](notebooks/colab_controls_and_seeds.ipynb) to run the
three missing seed-zero controls and two additional matched CODI/KaVa seeds:

```text
latent_nodistill_seed0
kava_random_seed0
kava_uniform_seed0
codi_seed1
kava_seed1
codi_seed2
kava_seed2
```

Each experiment starts from the pinned GPT-2 backbone, trains in an active Colab cell,
and mirrors atomic checkpoints to
`MyDrive/CODI_KAVA/outputs/controls_and_seeds/<experiment>/`. No checkpoint upload is
needed. Re-running the same experiment resumes it; changing the selector starts or resumes
only the newly selected isolated directory. All new capped evaluations use batch size 8;
the final report uses the saved `baseline_bs8` Phase-3 runs for seed zero, keeping the
decoding condition matched. Run the download-free contract check before GPU work:

```bash
python scripts/validate_controls.py
```

After all seven statuses are `complete`, the notebook creates a paired five-way seed-zero
control report and a matched three-seed CODI-vs-KaVa report. The seed report lists all
individual values and sample standard deviations; three seeds are not presented as a
high-confidence asymptotic method-level interval.

### Run Kaggle and Colab in parallel

Use [`kaggle_controls_and_seeds.ipynb`](notebooks/kaggle_controls_and_seeds.ipynb) as a
second worker while the Colab notebook runs. Never assign the same experiment to both.
Keep each matched method pair on one platform to avoid confounding the paired seed delta
with GPU/software environment:

```text
Kaggle: kava_random_seed0 -> codi_seed1 -> kava_seed1
Colab:  latent_nodistill_seed0 -> kava_uniform_seed0 -> codi_seed2 -> kava_seed2
```

On Kaggle, enable GPU and Internet, pin the same Git commit as Colab, and use **Save
Version → Save & Run All** with outputs enabled. Exit 42 requires another version with the
previous experiment directory attached and `RESUME_INPUT` set; exit 0 includes the capped
evaluation. The Colab notebook's opt-in Kaggle import cell downloads the saved notebook
output, verifies method/seed/manifest/checkpoint identity, and atomically installs it into
the standard Drive tree. The importer refuses non-empty targets, preventing accidental
merges of independently trained copies.

If a completed CODI seed-2 checkpoint was preserved without its evaluation artifacts, use
[`kaggle_codi_seed2_eval_only.ipynb`](notebooks/kaggle_codi_seed2_eval_only.ipynb). Attach
the `jonraza15/codi-seed-2-resume-dataset` input and run the notebook on a Kaggle GPU. It
validates step 96,405, evaluates without invoking training, keeps only the final checkpoint,
records a SHA-256 audit, and uploads `jonraza15/codi-seed2-final-step96405` for the normal
verified Drive import.

## Stage 1 spectral feasibility gate

Before training a TSV-inspired distillation variant, determine whether the trained KaVa
teacher-minus-student KV residuals contain stable low-rank directions. The workflow reuses
the exact R-KV alignment from training, treats keys and values independently, and compares
split-half spectra against cross-example shuffling and energy-matched isotropic noise.
It accumulates exact covariance sufficient statistics and never updates model weights.

Use [`colab_stage1_kv_subspaces.ipynb`](notebooks/colab_stage1_kv_subspaces.ipynb) for the
restartable Drive-backed run. Start with 2,000 examples. A later 5,000-example run extends
the same deterministic sample prefix without repeating completed examples. The complete
experimental contract, gate, output schema, and direct CLI commands are in
[`STAGE1_KV_SUBSPACE.md`](docs/STAGE1_KV_SUBSPACE.md).

If residual directions remain as stable after shuffling as under correct pairing, continue
with [`colab_stage1b_kv_cross_subspaces.ipynb`](notebooks/colab_stage1b_kv_cross_subspaces.ipynb).
Stage 1b whitens the teacher and student marginals, decomposes their centered
cross-covariance, and compares canonical correlations and canonical-direction stability
against repeated within-split derangements. Its complete protocol is in
[`STAGE1B_KV_CROSS_SUBSPACE.md`](docs/STAGE1B_KV_CROSS_SUBSPACE.md).

When Stage 1b finds strong position-resolved signal but its pooled gate fails, run the
CPU-only held-out prediction test with
[`colab_stage1c_kv_reduced_rank.ipynb`](notebooks/colab_stage1c_kv_reduced_rank.ipynb).
Stage 1c learns reduced-rank student-to-teacher maps on one calibration split and tests
them on the other, separately for every layer, head, latent position, and KV kind. Its
predefined rank-four gate and interpretation limits are documented in
[`STAGE1C_KV_REDUCED_RANK.md`](docs/STAGE1C_KV_REDUCED_RANK.md).

If rank-four keys pass Stage 1c, export the frozen signal and random-control bases and run
the four-arm warm-started comparison with
[`colab_stage1d_key_projection.ipynb`](notebooks/colab_stage1d_key_projection.ipynb).
It starts every arm from the completed CODI seed-one checkpoint and separates continued
training, full key supervision, learned rank-four key supervision, and random rank-four
supervision. The complete leakage, compute, loss, and decision contract is in
[`STAGE1D_KEY_PROJECTION_TRAINING.md`](docs/STAGE1D_KEY_PROJECTION_TRAINING.md).

## Answer-conditioned CODI endpoint follow-up

The corrected answer-cue endpoint run found strongly concentrated residual spectra but
no rank-77 answer utility. The fresh follow-up therefore fits residual PCs on one new
partition and selects block-state directions on a second new partition using stable
positive alignment with the gold-answer NLL gradient. It excludes the embedding state
and every normalized question used by the completed seed-11 endpoint experiment.

Run the complete restartable workflow with
[`kaggle_official_codi_endpoint_answer_conditioned.ipynb`](notebooks/kaggle_official_codi_endpoint_answer_conditioned.ipynb).
The frozen selection rule, matched controls, early-stop behavior, and final gate are in
[`OFFICIAL_CODI_ENDPOINT_ANSWER_CONDITIONED.md`](docs/OFFICIAL_CODI_ENDPOINT_ANSWER_CONDITIONED.md).

## Parameter-aware CODI endpoint follow-up

The answer-conditioned run dynamically reduced rank 77 to six PCs in the final two
block states, but their induced update was effectively orthogonal to the answer
gradient. The parameter-aware follow-up therefore selects from the first 64 residual
PCs in those two states using their induced LoRA-parameter-gradient cosine with the
gold-answer gradient. Candidate norms use deterministic Hutchinson sketches, and all
questions from both completed endpoint experiments are excluded before fresh seed-41
sampling.

Run the restartable workflow with
[`kaggle_official_codi_endpoint_parameter_aware.ipynb`](notebooks/kaggle_official_codi_endpoint_parameter_aware.ipynb).
The parameter geometry, split-stable selection rule, controls, and final utility gate
are documented in
[`OFFICIAL_CODI_ENDPOINT_PARAMETER_AWARE.md`](docs/OFFICIAL_CODI_ENDPOINT_PARAMETER_AWARE.md).

## Same-question paired CODI correction

The correct-versus-wrong covariance comparison mixes correctness with question
identity, and retaining a correct-only subspace discards 740 coordinates. The paired
follow-up instead creates controlled variants of each fitting question by perturbing
state 11, captures the resulting state 12 before answer selection, and learns an
additive question-conditioned correction restricted to PCs 4–31. Global-average and
shuffled-target maps are matched controls; fit, selection, and final-test questions are
disjoint.

Run the restartable GPU workflow with
[`kaggle_official_codi_paired_correction.ipynb`](notebooks/kaggle_official_codi_paired_correction.ipynb).
The intervention, leakage boundary, controls, and gate are documented in
[`OFFICIAL_CODI_PAIRED_CORRECTION.md`](docs/OFFICIAL_CODI_PAIRED_CORRECTION.md).

## Layout

```
configs/   YAML run configs
src/data    datasets, teacher caching, answer extraction   (Phase 1+)
src/models  latent-LM wrapper (<bot>/<eot> continuous thoughts)  (Phase 2+)
src/losses  configurable TrajectoryMatch + R-KV compression  (Phase 2+)
src/train   session-safe entrypoint + shared training loop
src/eval    numeric exact-match, OOD, paired comparison + uncertainty
src/mech    causal zero/replace/shuffle ablation (probes/CKA/patching planned)
tests/      Phase gates
```
