# CODI vs KaVa — Controlled & Mechanistic Study of Latent Reasoning Supervision

A single shared harness in which two latent-reasoning supervision methods differ *only*
in their distillation loss, so the study isolates the effect of **supervision
granularity**. See [`docs/PLAN.md`](docs/PLAN.md) for the full research plan and
[`KAVA_vs_CODI_SProj_Research_Proposal.pdf`](docs/KAVA_vs_CODI_SProj_Research_Proposal.pdf) for
the proposal.

- **CODI** (arXiv 2502.21074): endpoint hidden-state distillation.
- **KaVa** (arXiv 2510.02312): CODI **+** compressed KV-trajectory distillation.

## Status

**The primary seed for Phases 0–2 is trained; Phase 3 causal ablation is implemented.**
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

Run a full baseline evaluation from each completed Drive checkpoint without re-entering
the training loop. The runner restores to Colab's local SSD and atomically syncs predictions
back to Drive:

```bash
python -u scripts/colab_ablation_runner.py --method codi --mode baseline --limit 0
python -u scripts/colab_ablation_runner.py --method kava --mode baseline --limit 0
```

Then produce a strict paired comparison. The analyzer pairs rows by exact question and
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
