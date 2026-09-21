# Run teacher-code distillation on Kaggle

Notebook: [kaggle_teacher_code_distillation.ipynb](../notebooks/kaggle_teacher_code_distillation.ipynb).
Protocol: [LOW_RANK_TEACHER_CODE_DISTILLATION.md](LOW_RANK_TEACHER_CODE_DISTILLATION.md).

This notebook implements the primary explicit-CoT teacher-to-DistilGPT2 experiment.
The later CODI latent-student extension is not included. No full GPU result is claimed
by the local CPU checks.

## First run

1. Create a Kaggle notebook, choose File > Import Notebook, and upload the notebook.
2. Enable Internet and select a GPU (T4 x2 is suitable; only one GPU is used).
3. Keep STAGE="pilot", SMOKE=False, SEEDS=[89,90,91], RUN_SECONDARY=False.
4. Choose Save Version > Save & Run All. No attached model/data artifacts are needed.
5. Inspect the displayed pilot report. Preserve the output folder printed as RUN_DIR.

The pilot performs real teacher collection, codec fitting, serialized-target audits,
gradient comparisons, five student pilot fits and development evaluation. It does
not generate final-test answers. It can take substantial time; no complete T4 runtime
has been measured locally. SMOKE=True is an optional small integration run with its
own output identity and no scientific interpretation.

Setup clones immutable base commit 6a8d2e61950c67f012d0a9ba13ec8a70f3a25019 and writes
embedded helpers into that checkout. GitHub does not need your new commit before
the uploaded notebook can run. The official teacher checkpoint is checksum-verified.
DistilGPT2 is pinned to revision 2290a62682d06624634c1f46a6ad5be0f47f38aa.

## Full run after the pilot

If the pilot reports proceed=true, set STAGE="full". Keep the same configuration.
In the same session, outputs are reused automatically. In a new session:

1. Attach the previous saved notebook outputs using Add Input.
2. Set RESUME_ROOT to the exact directory containing manifest.json, for example:
   /kaggle/input/<your-saved-output>/teacher_code_distillation/full_<fingerprint>
3. Set STAGE="full" and Save Version > Save & Run All.

The directory path shown by Kaggle varies: locate manifest.json rather than copying
the example literally. Restored output is copied to writable /kaggle/working.

The full default grid is five arms x three seeds = 15 student fits. Setting
RUN_SECONDARY=True before starting the original pilot adds rank32, rank64 and
weight-SVD96, bringing the full grid to 24 fits. This changes experiment identity,
so adding it later starts a different run rather than modifying a locked result.

The development gate requires positive full-KD gain over supervised-only training.
This is a feasibility screen; the final statistical utility gate is stricter.
A failed gate saves a report and stops instead of silently launching the full grid.
Common pilot hyperparameters can be revised deliberately in the configuration cell;
there is no automatic hyperparameter search or per-arm tuning.

## Long runs, partial runs and resume

SESSION_HOURS=9 leaves a buffer before Kaggle runtime limits. A planned budget stop
writes session_pause.json. Save Version output persistence is essential; an unsaved
interactive session can lose its writable files if the platform terminates it.

Teacher collection checkpoints completed questions. Target exporters checkpoint
blocks. Codec fitting resumes completed epochs. Student training saves optimizer,
RNG, scaler and position every 20 optimizer steps, and at a planned budget stop.
An abrupt termination may replay up to 19 optimizer steps. Completed student models
and individual evaluation questions are reused.

To split the full grid across sessions without changing the experiment:
- ARM_FILTER=["sft","full","lr96"] runs those configured arms.
- SEED_FILTER=[89] runs that configured seed.
- None runs all configured arms/seeds.

Use one cumulative saved output chain when resuming. Concurrent independent output
branches are not merged by the notebook. A filtered run reports partial and cannot
produce a full completion marker until the entire configured grid is present.

Settings, source code, package versions, GPU type and checkpoint identity are locked.
Changing an identity field intentionally starts a new folder or rejects RESUME_ROOT.
Changing only stage, execution filters or session budget is supported.

## What is saved

Under /kaggle/working/teacher_code_distillation/full_<fingerprint>/:
- manifest.json and partitions.json: identities, eligible questions and exclusions.
- states/: per-split FP16 teacher states and collection cursors.
- codecs/: codec parameters, selection history and untouched audit metrics.
- targets/: standalone dense-hidden-state, low-rank, tail-aware top-k and sampled packages.
- storage.json: actual standalone target package bytes, including a fixed-size schema
  and any shared decoder weights. Research intermediates are additional.
- target_strata.json: target fidelity by question, token position and token type.
- gradient_audit_initial.json and gradient_audit_sft.json: KD-only and combined-loss
  gradient comparisons at shared student states.
- students/pilot/ and students/full/: weights, training histories, per-question
  predictions and development gold-token NLL.
- pilot_report.json: feasibility decision and an approximate precision forecast.
- results.json: complete paired analysis after all full arms/seeds finish.
- completion.json: written only for a completed full grid; inspect its smoke flag.
- sessions.json: elapsed time per session.

Student final checkpoints are saved in FP16 for output size, then loaded back into
the common model for evaluation. Interrupted optimizer checkpoints retain FP32
training state. This serialization policy is identical across arms. Final optimizer
states are removed only after final weights and training metrics have been written.

The printed tables distinguish accuracy from teacher agreement and count malformed
outputs and token-cap hits. A negative scientific gate is not an implementation error.
Three-seed and question uncertainty remain limited, especially for a 1 pp
noninferiority margin on GSM8K. The results retain both conditional and crossed
bootstrap intervals and never interpret a smoke run as evidence.

The first implementation records actual training throughput/memory, package bytes,
per-step times, and total session times. It does not yet implement a separate cold/warm
I/O microbenchmark, universal deployment speed claims, or independently replicated
codec-fitting seeds. The primary student seed replication uses one fixed codec.

## Local files and rebuilding

Sources:
- src/mech/teacher_code_distillation.py: target formats, losses, pairing, evaluation.
- scripts/run_teacher_code_distillation.py: collection, fitting, training, gates and resume.
- scripts/build_kaggle_teacher_code_distillation_notebook.py: standalone notebook builder.
- tests/test_teacher_code_distillation.py: CPU numerical and tiny-model pipeline tests.

After changing a helper or test, regenerate the notebook:
python scripts/build_kaggle_teacher_code_distillation_notebook.py

Run focused tests in an environment with PyTorch, Transformers 4.52.4, NumPy,
pytest and nbformat:
python -m pytest -q tests/test_teacher_code_distillation.py

The notebook also runs CPU tests before the GPU experiment. Tests use synthetic
tiny models and require no model downloads. Local passing tests do not validate
CUDA behavior or establish teacher/student mathematical accuracy.

## Commit to zain-kaggle

Run in the repository terminal:

git switch zain-kaggle
git add docs/LOW_RANK_TEACHER_CODE_DISTILLATION.md docs/KAGGLE_TEACHER_CODE_DISTILLATION.md notebooks/kaggle_teacher_code_distillation.ipynb scripts/build_kaggle_teacher_code_distillation_notebook.py scripts/run_teacher_code_distillation.py src/mech/teacher_code_distillation.py tests/test_teacher_code_distillation.py
git diff --cached --stat
git commit -m "Add Kaggle teacher-code distillation experiment"
git push -u origin zain-kaggle

Only the seven named files are staged. Local test dependencies under tmp/ are ignored.
Kaggle Save Version preserves Kaggle outputs; it does not commit or push Git changes.

