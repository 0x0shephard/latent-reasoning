# CODI vs KaVa — Controlled & Mechanistic Study of Latent Reasoning Supervision

A single shared harness in which two latent-reasoning supervision methods differ *only*
in their distillation loss, so the study isolates the effect of **supervision
granularity**. See [`docs/PLAN.md`](docs/PLAN.md) for the full research plan and
[`KAVA_vs_CODI_SProj_Research_Proposal.pdf`](docs/KAVA_vs_CODI_SProj_Research_Proposal.pdf) for
the proposal.

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
