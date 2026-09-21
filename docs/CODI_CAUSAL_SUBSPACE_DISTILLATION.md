# Causal versus variance selection of the distillation subspace

Preregistered as ledger §86. Contract `official_codi_causal_subspace_distillation_v1`.

## Question

Subspace distillation methods choose the teacher subspace by statistics of the
teacher: variance and second moments (LoRi, June 2026), or gradient relevance
(Flex-KD, SubDistill). Ledger §40 showed that at CODI's decision state the
top-variance directions are inert and accuracy lives in a low-variance band, and
§43–§52 showed gradient-selected directions repeatedly failed matched causal
tests. Nobody has asked whether a variance- or relevance-selected distillation
target wastes its rank on directions the answer never uses.

> When a latent student is distilled toward a rank-r subspace of the teacher's
> decision state, does selecting that subspace by causal retain-and-remove on the
> teacher transfer more accuracy than selecting it by variance, by gradient
> relevance, or at random, at matched rank?

## Why the student is not the checkpoint

The checkpoint is used for everything except the part that has to learn. The
teacher is its explicit-CoT path, frozen. Its embedding table and tied readout are
shared with the student, so the special tokens are meaningful and the causal
subspace defined through the readout is the student's own readout. Only the LoRA
adapters (A random, B zero) and the projector start fresh, which is the checkpoint
with its latent adaptation removed and nothing else. §85 established why the
finished checkpoint cannot be the student: on a converged student every
distillation target looks identical within noise, because the target's shaping has
already happened. This question is about which directions transfer accuracy while
the latent task is being learned.

## Teacher side, before any training

The teacher's post-`ln_f` colon state (state 12) and the gold first answer token are
precomputed for every row. Principal components are fitted on 2,048 rows. Every
subspace arm is an index set over those PCs, so arms differ only in *which* PCs are
kept:

| arm | selector |
| --- | --- |
| `variance` | the first r PCs |
| `relevance` | the r PCs with the largest Fisher score E[(g·v)²], g the gradient of gold first-token NLL at state 12 through the readout |
| `causal` | greedy forward selection over the top-128 PCs maximising retain-only first-token accuracy, ties broken by mean answer margin, on a disjoint 2,048-row split (the §36 analytic tier); greedy is nested, so one pass serves every rank |
| `random` | r seeded PCs |
| `full` | all 768 dimensions |
| `none` | answer cross-entropy only |

Rank is chosen on a third disjoint 2,048-row split from {8, 12, 16}: among ranks
where the causal set beats the variance set by at least five points and retains at
least half of the teacher's dense first-token accuracy under retain-only, the rank
maximising gap times retention. (Amended before training from a 75% retention
requirement; see ledger §86.) Ranks near 32
would make the comparison vacuous, since §40's band (PCs 4–31) sits inside the
variance top-32; small ranks force variance selection to spend a third or more of
its budget on the inert PCs 0–3. If no rank satisfies both conditions the selectors
are not distinguishable on the teacher and the run stops before training. Retention,
variance share and pairwise Jaccard overlap of every set are reported.

## Student side

Six arms, each seed a fresh reset. Loss: answer cross-entropy plus the state-12
distillation term, smooth-L1 over the selected teacher coordinates (all 768 for
`full`) scaled by the teacher's coordinate standard deviation, with the distillation
gradient norm-matched to the cross-entropy gradient every step. The reference term
is being learned, not converged, so the §85 failure does not apply. 10,000 steps at
batch 16 (160,000 GSM8k-Aug rows, about 0.4 epoch), AdamW at 1e-4 cosine with 500
warm-up steps, weight decay 0.1, gradient clip 2.0, float32: the pilot's CODI recipe.
Every 1,000 steps the selection split (256 rows) records teacher-forced answer NLL
and native-decoding exact match. Each run ends with one read of the full GSM8K test:
exact match and teacher-forced NLL.

## Sessions and accounts

Runs checkpoint every 500 steps and stop cleanly two minutes before `--max-seconds`
(8.5 hours by default), leaving a resumable state. The notebook resumes from an
attached copy of the previous output. Two accounts split the seeds: one runs
`--seeds 1`, the other `--seeds 2`; `--aggregate-from` combines both outputs into
the final summary and gates.

## Gates

Paired bootstrap over questions on per-question seed-mean correctness, 10,000
resamples:

1. **S1, sanity:** `full − none` lower bound above zero. If distillation itself is
   not detectable at this budget, the subspace comparisons are uninformative.
2. **H1:** `causal − variance` lower bound above zero.
3. **H2:** `causal − relevance` lower bound above zero.
4. **H3:** `causal − random` lower bound above zero; `variance − random` reported.
5. `causal − full`, two-sided: whether a rank-r causal target matches the full one.

Headline requires S1, H1 and H3. Teacher-forced NLL differences are reported as the
secondary outcome for every pair.

## Expectations

Fresh adapters at 0.4 epoch may reach low absolute accuracy (the §2 pilot reached
13% after one epoch of joint training), so the primary comparison may be
underpowered even at n = 1,319; the selection-split NLL curve is the sensitive
secondary. A positive result is a targeted correction to a June 2026 method. A null
with S1 passing says the inert directions cost nothing during training, which is
also worth knowing.

## Inputs and outputs

Attach the completed official CODI reproduction dataset containing
`official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json`. For a resumed
session also attach the previous output. Outputs under
`/kaggle/working/codi_causal_subspace_distillation`: `summary.json`, `sampling.json`,
`teacher_cache.pt`, `selectors.json`, per-run records and checkpoints under `runs/`,
and after aggregation `causal_subspace_distillation.pt` and `predictions.jsonl`.

## Files

- implementation: `src/mech/causal_subspace_distillation.py`
- runner: `scripts/run_codi_causal_subspace_distillation.py`
- notebook builder: `scripts/build_kaggle_codi_causal_subspace_distillation_notebook.py`
- notebook: `notebooks/kaggle_codi_causal_subspace_distillation.ipynb`
- tests: `tests/test_causal_subspace_distillation.py`,
  `tests/test_causal_subspace_distillation_runner.py`,
  `tests/test_codi_causal_subspace_distillation_notebook.py`
