# Running CODI global-head ablations on Kaggle

Notebook: `notebooks/kaggle_codi_global_head_ablations.ipynb`.

1. Create a Kaggle notebook and use **File > Import Notebook** to upload this file.
2. Set **Accelerator** to a GPU and enable **Internet**. One GPU is used, even if Kaggle allocates two.
3. No data attachments are needed. The notebook downloads the official model checkpoint,
   pinned dependencies, and training/evaluation datasets automatically. It checks the
   checkpoint hash and decoder parity itself, then evaluates a fresh dense baseline.
4. The defaults `SMOKE=False`, `SUITE="core"`, and `SEEDS=[89]` run the real core experiment.
   `SMOKE=True` is optional: it runs a small setup check on eight evaluation questions.
5. Choose **Save Version > Save & Run All** to execute and preserve output files.
6. For final fitting-seed comparisons use `SEEDS=[89, 90, 91]`. Run optional suites as
   separate saved notebooks/versions, or set `SUITE="all"` if your runtime budget permits.
   No GPU runtime estimate has been validated. Core runs five fits and fifteen
   compressed-head evaluations per seed per dataset; all runs fifteen fits and forty-five.

## Suites

| Setting | Comparison |
|---|---|
| `core` | Baseline, fixed U28, fixed random-28, first-token fitting, matched-count trajectory fitting |
| `initialization` | Activation-aware versus ordinary weight SVD, identical distillation |
| `losses` | Full loss, no margin term, no top-token term |
| `recovery` | Clean baseline, extra clean updates, teacher recovery states, on-policy recovery states |
| `data` | 128/256/512/1024 nested unique-question subsets with matched state presentations |
| `all` | Union of all suites, deduplicating the baseline |

All arms evaluate nested ranks 32, 64, 96. Core heads have four clean epochs; recovery
arms start from that same clean checkpoint and take two additional matched-budget epochs.
Coverage samples one state per question in both arms. Data-size comparisons replay
smaller state pools to match optimizer updates while including every selected question.
The baseline retains the original method's nested-prefix loss and validation selection.
Smaller ranks are prefixes, not independently optimized models.

Set `EVAL_DATASETS=["gsm8k", "svamp"]` before starting to evaluate frozen heads on
both datasets. This is data transfer, not refitting. Model/architecture transfer is
not implemented by this CODI-specific notebook; the repository's existing Qwen notebook
is a separate experiment and does not inherit CODI's U28 coordinates.

## Inputs and provenance

No historical low-rank checkpoint or colon-state dataset is required. Every comparison
head is refitted on the same partitions. U28 is recomputed using PCs 4:32 of the first
answer-token training states. This measures the U28 construction, not replication of
the exact reported historical basis. Training is split by normalized unique question,
not augmented row. The notebook checks that selected training/validation/recovery
questions do not appear in evaluation. The already-used GSM8K test supports follow-up
comparisons; do not describe it as a new untouched confirmatory holdout.

The notebook embeds `src/mech/global_head_ablations.py` and clones immutable base commit
`6a8d2e61950c67f012d0a9ba13ec8a70f3a25019`. Uploading does not require committing or
pushing the local branch. Regenerate after helper changes with
`python scripts/build_kaggle_codi_global_head_ablations_notebook.py`.

## Outputs and resume

Download `/kaggle/working/codi_global_head_ablations/full_<run-id>/`:

- `results.csv`: exact-match accuracy, retention and paired differences against dense.
- `paired_comparisons.csv`: controlled arm-to-arm comparisons, including U28 rank 64
  versus unconstrained rank 96. Confidence intervals are unadjusted for multiplicity.
- `seed_summary.csv`: mean and standard deviation across fitting seeds; one seed has no SD.
- `head_*.json`: training history and first/later-token agreement and KL on validation states.
- `head_*.pt`: fitted checkpoints; `eval_*.json`: question-level outputs and paired flags.
- `manifest.json`, `partitions.json`, `u28_train_only.pt`: locked settings, splits and basis.
- `completion.json`: written only after every configured evaluation completes.

Same-session reruns skip completed fits and evaluation arms. To continue in another
session, attach the saved output as an input and point `RESUME_ROOT` at its exact run
folder. Keep all settings and runtime identity the same; mismatches are rejected.
An interrupted fit/evaluation arm restarts from its beginning, not its last batch.
Changing batch size creates a different run identity. Keep output below Kaggle's
storage limits by downloading/archiving completed suites separately.

Kaggle instructions: https://www.kaggle.com/docs/notebooks

## Commit and push to GitHub

From the repository terminal (the five new ablation files are the only current changes):

```sh
git switch zain-kaggle
git add docs/CODI_GLOBAL_HEAD_ABLATIONS.md notebooks/kaggle_codi_global_head_ablations.ipynb scripts/build_kaggle_codi_global_head_ablations_notebook.py src/mech/global_head_ablations.py tests/test_global_head_ablations.py
git commit -m "Add standalone Kaggle CODI head ablations"
git push -u origin zain-kaggle
```

Kaggle's Save Version saves a Kaggle run; it does not push to GitHub. Import the local
notebook into Kaggle after regenerating it when helpers change. The notebook itself
clones GitHub automatically at the pinned base commit, rather than following branch
changes. Its new ablation helper is embedded, so a Git push is optional for running it.

`core` is the five-fit starter suite. `all` runs all fifteen fits per seed.
`SEEDS=[89]` runs one reproducible fit per arm; the number 89 is simply a random seed.
Three seeds measure sensitivity to training randomness and cost about three times
as many fits. No full GPU runtime estimate is validated.
