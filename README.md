# CODI vs KaVa — Controlled & Mechanistic Study of Latent Reasoning Supervision

A single shared harness in which two latent-reasoning supervision methods differ *only*
in their distillation loss, so the study isolates the effect of **supervision
granularity**. See [`docs/PLAN.md`](docs/PLAN.md) for the full research plan and
[`KAVA_vs_CODI_SProj_Research_Proposal.pdf`](docs/KAVA_vs_CODI_SProj_Research_Proposal.pdf) for
the proposal.

- **CODI** (arXiv 2502.21074): endpoint hidden-state distillation.
- **KaVa** (arXiv 2510.02312): CODI **+** compressed KV-trajectory distillation.

## Status

**Phases 0–2 implementation complete.** Phase 1's CoT-SFT checkpoint passed the real-data
gate and scores 26.0% GSM8k / 33.86% macro exact-match on the saved 200-example evaluation.
Phase 2 now provides one shared continuous-thought GPT-2 wrapper, CODI endpoint matching,
KaVa KV-trajectory matching, R-KV compression, matched configs, latent-aware evaluation,
and a Kaggle preflight. Long CODI/KaVa GPU runs remain to be executed.

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

## Layout

```
configs/   YAML run configs
src/data    datasets, teacher caching, answer extraction   (Phase 1+)
src/models  latent-LM wrapper (<bot>/<eot> continuous thoughts)  (Phase 2+)
src/losses  configurable TrajectoryMatch + R-KV compression  (Phase 2+)
src/train   session-safe entrypoint + shared training loop
src/eval    numeric exact-match + OOD (efficiency/calibration planned)
src/mech    probes, CKA/SVCCA, ablation, patching            (Phase 3+)
tests/      Phase gates
```
