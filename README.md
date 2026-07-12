# CODI vs KaVa — Controlled & Mechanistic Study of Latent Reasoning Supervision

A single shared harness in which two latent-reasoning supervision methods differ *only*
in their distillation loss, so the study isolates the effect of **supervision
granularity**. See [`docs/PLAN.md`](docs/PLAN.md) for the full research plan and
[`KAVA_vs_CODI_SProj_Research_Proposal.pdf`](KAVA_vs_CODI_SProj_Research_Proposal.pdf) for
the proposal.

- **CODI** (arXiv 2502.21074): endpoint hidden-state distillation.
- **KaVa** (arXiv 2510.02312): CODI **+** compressed KV-trajectory distillation.

## Status

**Phase 0 — scaffolding & session-safe trainer.** A tiny synthetic task exercises the
harness (deterministic, checkpoint/resume, wall-clock guard) before any real modeling.
Phases 1–4 add data/eval, the latent-LM + CODI/KaVa losses, mechanistic analysis, and the
supervision-granularity continuum.

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

## Tests (Phase 0 gate)

```bash
pytest -q
```

Covers determinism (same seed → identical losses) and resume continuity (interrupt +
resume matches an uninterrupted run bit-for-bit). CPU-only, no downloads.

## Layout

```
configs/   YAML run configs
src/data    datasets, teacher caching, answer extraction   (Phase 1+)
src/models  latent-LM wrapper (<bot>/<eot> continuous thoughts)  (Phase 2+)
src/losses  configurable TrajectoryMatch + R-KV compression  (Phase 2+)
src/train   session-safe entrypoint + shared training loop
src/eval    exact-match / OOD / efficiency / calibration     (Phase 1+)
src/mech    probes, CKA/SVCCA, ablation, patching            (Phase 3+)
tests/      Phase gates
```
