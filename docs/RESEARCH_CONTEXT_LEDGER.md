# Research context ledger

Last updated: 2026-08-08 (rev 2)

## Purpose

This is the durable source of truth for continuing the CODI–KaVa project when the
conversation becomes too long or is compacted. Read this file before proposing another
experiment. It records the original question, the instructor's criticism, the
TSV-inspired pivot, completed experimental gates, current evidence, and the next
decision.

The latest result is a completed full-GSM8K causal intervention on the official CODI
checkpoint. Learned rank-four KV directions did **not** show greater causal value than
energy-matched random rank-four directions.

## 1. Original problem and research question

The project began in efficient test-time reasoning and efficient LLM inference:

- test-time reasoning can improve quality by spending more inference compute
- efficient inference tries to spend that compute only when useful and execute it with
  less memory, latency, and energy
- long reasoning traces increase generated tokens and KV-cache memory
- latent-reasoning methods try to replace explicit text reasoning with compact
  continuous computation

The starting methods were CODI, KaVa, and R-KV:

- CODI compresses chain-of-thought into continuous latent states through
  self-distillation
- KaVa adds compressed key/value trajectory supervision
- R-KV selects KV-cache positions using redundancy-aware compression

The first broad question was:

> Which method is better for latent mathematical reasoning, CODI or KaVa?

It was refined into:

> Under an identical architecture, dataset, optimizer, decoding procedure, latent
> budget, and training budget, does KaVa's compressed KV-trajectory supervision improve
> over CODI's endpoint hidden-state distillation?

The mechanism question was:

> If KaVa improves, is the improvement specifically caused by R-KV-compressed
> trajectory supervision?

## 2. Initial controlled pilot

CODI and KaVa were implemented in a shared GPT-2 harness with:

- pretrained GPT-2 initialization
- 385,620 training examples
- six autoregressive latent positions
- one epoch
- batch size four
- 96,405 optimizer steps
- identical evaluation and numeric exact-match scoring

The pilot found:

- full seed-zero CODI macro accuracy: 11.28 percent
- full seed-zero KaVa macro accuracy: 13.29 percent
- paired gain: +2.01 percentage points
- 95 percent paired-bootstrap interval: +0.44 to +3.60 points
- capped matched-seed differences: +2.17, +0.97, and +1.08 points
- most of the gain came from MultiArith
- cross-example latent shuffling harmed KaVa more than CODI in the capped pilot

These were real results for the trained checkpoints, but the absolute accuracies were
far below the papers.

## 3. First major criticism and correction

The instructor raised three objections:

1. KaVa accuracy around 13 percent was too low for strong conclusions about the
   methods.
2. Official CODI checkpoints were available and should have been used as an evaluator
   and reproduction gate.
3. The local setup was not sufficiently paper-aligned to be treated as a reproduction.

The pilot used one epoch and batch size four because of compute and time constraints.
Published work used substantially more training, larger effective batches, and, for
KaVa, larger instruction models and a different latent mechanism. The pilot was
therefore reclassified as a compute-limited controlled experiment rather than a
reproduction.

Corrective principle:

> Do not make mechanistic claims from low-accuracy pilot checkpoints when a
> paper-accuracy official checkpoint can be used.

## 4. Updated teacher-target question

The project moved from the broad method comparison toward the supervision target:

> Under a fixed teacher-target budget and a paper-aligned latent-reasoning setup, which
> parts of the teacher KV trajectory are necessary and sufficient for transferring
> mathematical reasoning, and can a value-aware or answer-causal selector outperform
> R-KV as a distillation target?

This contains two selection problems:

1. **Token-position selection**  
   Which teacher reasoning tokens should be retained under a fixed budget?

2. **Information-direction selection**  
   Within the selected high-dimensional keys and values, which directions contain
   transferable task signal rather than noise or redundant representation structure?

R-KV was the proposed position selector. A TSV-inspired spectral method was proposed
for selecting information directions.

## 5. Instructor skepticism and the TSV suggestion

The instructor's central skepticism was:

> It may be difficult to find a stable correct-answer causal pattern because hidden
> states and KV representations contain substantial noise.

The suggested analogy was Task Singular Vectors from model merging. TSV-style methods
use spectral structure to separate recurring task signal from noisy or interfering
directions.

Applied here, the proposal became:

> First test whether differences between teacher and student KV trajectories contain
> stable low-rank signal subspaces. Let R-KV choose teacher token positions, and let a
> TSV-inspired spectral step identify which directions within those KV representations
> should be distilled.

Intuitively:

```text
R-KV chooses where to look.
The spectral step chooses what information to keep.
```

The working hypothesis was:

```text
teacher/student KV relationship
    = stable low-rank transferable signal
    + noise and redundant directions
```

## 6. Why the hypothesis was split into gates

Expensive distillation was postponed. The hypothesis was divided into increasingly
strong tests:

1. Does low-rank structure exist?
2. Is it paired teacher–student structure rather than marginal covariance?
3. Does it predict teacher KV information on untouched examples?
4. Is R-KV uniquely better than matched position-selection controls?
5. Do learned spectral directions affect answers more than random directions?
6. Only if earlier gates pass, does the target improve downstream student training?

These properties are different:

- **stable** means a pattern repeats across data splits
- **predictive** means student states reconstruct teacher information
- **causal** means changing the information changes answers
- **transferable** means supervising with it improves a newly trained student

One does not automatically imply the next.

## 7. Spectral diagnostics

### Stage 1 residual covariance

Teacher-minus-student KV residuals appeared low-rank and split-stable, but shuffled
teacher–student pairings were similarly stable.

Result:

> Not supported by the preregistered gate.

Residual covariance mixed marginal teacher and student structure with their paired
relationship. Stable covariance alone did not isolate transferable signal.

### Stage 1b paired cross-subspaces

Whitened teacher–student cross-covariance was used to isolate paired dependence. The
pooled layer-head gate failed because pooling six latent positions mixed different
position-specific relationships. Position-resolved results were much stronger than
their shuffled controls.

Interpretation:

> Latent-position identity matters and should not be pooled away.

### Stage 1c held-out reduced-rank prediction

Maps were fitted on one split and evaluated on the untouched split while preserving
latent position.

Key 5,000-example results:

- key rank-four held-out R-squared: 0.2629
- key actual-minus-shuffle R-squared: 0.2735
- key rank/full retention: 0.8272
- value rank-four held-out R-squared: 0.1730
- value actual-minus-shuffle R-squared: 0.1813
- value rank/full retention: 0.7182

The key gate passed. The value gate missed the preregistered 80 percent full-rank
retention requirement.

Interpretation:

> Stable position-conditioned rank-four **predictive** structure existed,
> particularly for keys. This did not establish answer causality.

## 8. Short pilot projection training

A 10,000-step pilot compared continued CODI training, full key supervision, learned
rank-four key supervision, and random rank-four key supervision. Learned rank-four
supervision did not outperform full or random-rank supervision.

This was not decisive because it inherited the low-quality local checkpoint and short
continuation budget. It motivated moving important tests to an official checkpoint.

## 9. Official CODI reproduction gate

The author-released CODI GPT-2 checkpoint was loaded with its released LoRA,
projection, prompts, latent-generation procedure, and scoring protocol.

Full GSM8K:

- correct: 576 of 1,319
- accuracy: 43.669 percent
- published reference: 43.7 percent
- gate: passed

This removed undertraining and evaluator-compatibility confounds.

## 10. Official low-rank replication

The paired collection and reduced-rank analysis were repeated on the official
checkpoint using:

- 2,000-example seed-zero calibration
- independent 5,000-example seed-one calibration
- exact layer, head, and latent-position alignment
- R-KV-selected teacher trace positions
- shuffled-pairing nulls
- held-out reduced-rank prediction

The predictive low-rank result replicated on the paper-accuracy checkpoint. It was not
an artifact of the undertrained pilot, but it still answered prediction rather than
answer causality.

## 11. R-KV selector specificity

The official 5,000-example experiment compared R-KV, uniform selection, and four fixed
random selectors under matched examples, states, splits, and shuffled nulls.

Held-out position-conditioned signal R-squared:

| Selector | Key | Value |
| --- | ---: | ---: |
| R-KV | 0.1686 | 0.0872 |
| Uniform | 0.1770 | 0.0955 |
| Random controls | approximately 0.092 | approximately 0.041 |

Result:

> R-KV selector specificity not supported.

R-KV clearly beat random selection but did not beat uniform selection. It was not
established as the uniquely best source of predictable teacher KV signal.

## 12. Boundary-aware selector

A fresh disjoint 5,000-example experiment tested a selector that always retained the
first and last valid trace tokens and used R-KV for four interior positions.

Aggregate results sometimes favored the candidate, but the predefined per-group
margins and win fractions against all structured controls were not met.

Result:

> Boundary-aware R-KV specificity not supported.

Decision:

> Stop designing token selectors based only on predictable linear KV signal.

## 13. Official spectral-causality experiment

The next experiment tested whether learned rank-four student KV directions were more
causally important than energy-matched random rank-four directions.

For centered vector `x - mean` and learned projector `P`:

```text
retain learned = mean + P(x - mean)
remove learned = mean + (I - P)(x - mean)
```

Random controls used groupwise random orthonormal rank-four bases scaled to match
expected projected calibration energy. Keys and values were intervened on together,
immediately after a selected latent KV entry was appended and before later computation
consumed it.

Evaluation:

- frozen official CODI checkpoint
- full 1,319-example GSM8K
- unchanged baseline reproduced first
- positions 0 through 5 and all positions jointly
- retain tested sufficiency
- remove tested necessity
- primary family: retain/remove at positions 4 and 5
- 10,000 paired-bootstrap samples
- exact McNemar tests
- Holm correction across four primary comparisons

Primary results:

| Position | Intervention | Baseline | Learned | Random | Learned minus random | 95% CI | Holm p |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | retain | 0.4367 | 0.3821 | 0.3806 | +0.0015 | [+0.0000, +0.0038] | 1 |
| 4 | remove | 0.4367 | 0.4329 | 0.4359 | -0.0030 | [-0.0076, +0.0015] | 1 |
| 5 | retain | 0.4367 | 0.4344 | 0.4352 | -0.0008 | [-0.0045, +0.0023] | 1 |
| 5 | remove | 0.4367 | 0.4352 | 0.4359 | -0.0008 | [-0.0038, +0.0015] | 1 |

Result:

> Learned subspace causality not supported.

Position-sweep observations:

- retaining rank four at position 0 reduced accuracy by about 13 points for both
  learned and random directions
- retaining rank four at position 4 reduced accuracy by about 5.5 points for both
- position 5 retention remained close to baseline for both
- retaining rank four at all six positions reduced accuracy from 43.67 percent to
  about 13 percent for both learned and random directions
- removing learned rank-four directions caused only small changes
- learned-minus-random effects were tiny throughout the sweep

Interpretation:

> Four dimensions are often insufficient as the only KV information, but the learned
> four dimensions are not meaningfully better than matched random dimensions. Removing
> them also causes little unique damage.

## 14. Current scientific conclusion

The completed evidence supports:

> Stable, position-conditioned, teacher-predictive low-rank KV structure exists in the
> official CODI checkpoint, especially for keys. However, R-KV was not superior to
> uniform position selection, and full-GSM8K interventions found no evidence that the
> learned rank-four directions were more necessary or sufficient for correct answers
> than energy-matched random directions.

The TSV-inspired method found recurring structure, but recurring structure was not the
same as answer-causal signal.

Intuitive analogy:

> Engine sound is a stable and predictable sign that a car is moving, but removing the
> sound does not stop the car. The discovered KV directions may describe computation
> without steering it.

## 15. Status of the updated question

What has been answered:

- stable low-rank predictive teacher–student KV structure exists
- latent-position identity matters
- R-KV is stronger than random but not stronger than uniform for predictable linear
  signal
- tested learned rank-four directions are not uniquely necessary or sufficient for
  official CODI GSM8K answers

What remains unanswered:

- which teacher KV tokens are answer-causal
- whether keys and values have different causal roles
- whether an answer-conditioned selector can beat R-KV
- whether such a selector improves downstream student training
- whether useful causal information is higher-rank, nonlinear, example-specific, or
  distributed redundantly across layers and heads

The evidence does **not** justify spectral distillation training with the learned
rank-four targets.

## 16. Possible next question

If the project continues, the cleanest narrower question is:

> Under a fixed six-target budget, can teacher KV positions selected by their measured
> causal effect on correct-answer probability outperform R-KV, uniform, and random
> selection on held-out mathematical reasoning examples?

The definition of signal would change:

```text
Old:
information that reconstructs the teacher KV representation

New:
information whose removal changes correct-answer probability
```

Minimal HOW:

1. Run the teacher and record correct-answer log-probability.
2. Modify or remove each eligible reasoning token's key and value.
3. Measure the change in correct-answer log-probability.
4. Score keys and values separately.
5. Use gradient-times-activation for inexpensive screening.
6. Validate high- and low-scoring tokens with exact interventions on untouched data.
7. Compare fixed-budget answer-causal, R-KV, uniform, and random selectors.
8. Require a held-out causal advantage before new student training.
9. Only after a positive gate, compare downstream distillation with identical compute.

This is a possible next direction, not an already approved experiment.

## 17. Decisions not to reverse without new evidence

- Do not treat the low-accuracy local CODI/KaVa models as paper reproductions.
- Do not use the pilot's 13 percent accuracy for strong method-level claims.
- Do not claim that predictive low-rank KV directions are answer-causal.
- Do not claim that R-KV is uniquely superior to uniform selection.
- Do not start rank-four spectral distillation based on the completed gates.
- Do not design more selectors using only the same predictable-linear-signal objective.
- Use official or paper-accuracy checkpoints for mechanistic claims.
- Keep target count, model, data, decoding, and compute fixed across selectors.
- Separate exploratory observations from preregistered confirmation tests.

## 18. Durable artifacts and code

External Google Drive roots:

```text
MyDrive/CODI_KAVA/
  outputs/official_codi_gpt2/
  outputs/official_codi_kv_subspaces/n5000_seed1/
  outputs/official_codi_selector_specificity/n5000_seed1/
  outputs/official_codi_boundary_selector/n5000_seed2/
  reports/
  logs/
```

Completed Kaggle causal export:

```text
/kaggle/working/official_codi_kv_causal_export/
```

Completed Kaggle kind-level target-utility export:

```text
/kaggle/working/official_codi_kv_target_utility_export/
```

The Kaggle result should be saved as a Kaggle dataset or imported to Drive before its
notebook version is deleted. Large external statistics and prediction files are not
necessarily present in the local repository.

Protocol documents:

- `docs/OFFICIAL_CODI_VALIDATION.md`
- `docs/OFFICIAL_CODI_KV_SUBSPACES.md`
- `docs/OFFICIAL_CODI_SELECTOR_SPECIFICITY.md`
- `docs/OFFICIAL_CODI_BOUNDARY_SELECTOR.md`
- `docs/OFFICIAL_CODI_KV_CAUSAL.md`
- `docs/ANSWER_CAUSAL_SIGNAL_DEFINITION.md`
- `docs/OFFICIAL_CODI_KV_TARGET_UTILITY.md`
- `docs/OFFICIAL_CODI_KV_GRADIENT_SIGNAL.md`

Execution notebooks:

- `notebooks/colab_official_codi_validation.ipynb`
- `notebooks/colab_official_codi_kv_subspaces.ipynb`
- `notebooks/colab_official_codi_selector_specificity.ipynb`
- `notebooks/colab_official_codi_boundary_selector.ipynb`
- `notebooks/colab_official_codi_kv_causal.ipynb`
- `notebooks/kaggle_official_codi_kv_causal.ipynb`
- `notebooks/kaggle_official_codi_kv_target_utility.ipynb`
- `notebooks/kaggle_official_codi_kv_gradient_signal.ipynb`

Core code:

- `scripts/collect_official_codi_kv_subspaces.py`
- `scripts/export_official_codi_student_subspaces.py`
- `scripts/run_official_codi_kv_causal.py`
- `scripts/analyze_official_codi_kv_causal.py`
- `src/mech/kv_reduced_rank.py`
- `src/mech/official_codi_kv_intervention.py`
- `src/eval/official_codi_kv_causal_analysis.py`
- `src/models/official_codi.py`
- `scripts/run_official_codi_kv_target_utility.py`
- `src/mech/kv_target_utility.py`
- `src/eval/official_codi_kv_target_utility_analysis.py`
- `scripts/run_official_codi_kv_gradient_signal.py`
- `src/mech/kv_gradient_signal.py`
- `src/eval/official_codi_kv_gradient_signal_analysis.py`

## 19. Instructions for future continuation

Before continuing:

1. Read this file.
2. Confirm that the completed Kaggle causal export is stored durably.
3. Treat the rank-four spectral-causality experiment as complete and negative.
4. Do not repeat the learned-versus-random rank-four experiment.
5. State exactly which remaining question any proposed experiment answers.
6. Require a cheap held-out gate before recommending expensive training.
7. Bound conclusions to official CODI GPT-2, rank-four linear subspaces, six latent
   positions, and GSM8K.

Current decision point:

> Either close the TSV-inspired rank-four direction as a rigorous negative result and
> write it up, or design a separate answer-conditioned causal-selection gate. Do not
> proceed directly to distillation training.

## 20. Operational signal definition

The project chose the answer-conditioned path and froze the vocabulary and evidence
hierarchy in `docs/ANSWER_CAUSAL_SIGNAL_DEFINITION.md`.

The central definition is:

> Answer-causal KV signal is information whose matched intervention produces a
> reproducible change in held-out gold-answer probability beyond an
> intervention-matched null.

A candidate teacher target must be both:

1. answer-causal under held-out intervention
2. accessible from the student's latent state beyond shuffled pairing
3. optimization-aligned, meaning its matched update lowers held-out answer loss

The label **transferable supervision signal** is reserved for a later compute-matched
training improvement. Stable, low-rank, or teacher-predictive structure alone is now
called structural or predictive signal rather than task signal.

The immediate next step is a hierarchical marginal-utility screen over KV kind, latent
position, and layer group. It compares held-out answer loss after matched functional
updates with and without each target family. Raw distillation loss is not the criterion,
because deleting a non-negative loss term lowers the reported objective mechanically.
Only target families with positive held-out utility proceed to answer-conditioned
direction discovery. Gold-answer log-probability is the differentiable outcome. Numeric
exact match under the released CODI generation protocol remains the confirmatory task
outcome.

No new distillation training is authorized by this definition alone.

## 21. Target-utility implementation

The hierarchical marginal-utility screen is implemented in:

- `scripts/run_official_codi_kv_target_utility.py`
- `src/mech/kv_target_utility.py`
- `src/mech/official_codi_target_utility.py`
- `src/eval/official_codi_kv_target_utility_analysis.py`
- `docs/OFFICIAL_CODI_KV_TARGET_UTILITY.md`

It compares answer-only, correctly paired KV-target, and shuffled KV-target parameter
updates. All updates have the same parameter L2 norm and are evaluated on disjoint
normalized-question groups through stateless functional calls. It starts with key
versus value targets, then permits position and layer-band refinement only for helpful
branches.

The implementation does not itself authorize another TSV decomposition or training
run. Its first required execution is a small GPU smoke test followed by the kind-level
screen.

## 22. Completed kind-level target-utility result

The official-CODI kind-level screen completed on Kaggle with the preregistered
configuration:

- checkpoint revision `fd641b3d3edc`
- 128 discovery and 128 disjoint validation examples
- 32 paired update batches of size four
- all 12 layers and all six latent positions pooled within each KV kind
- L1 KV loss
- relative total update norm `1e-4`
- 10,000 paired update-batch bootstrap samples

Results:

| Target | Candidate vs no target | 95% CI | Candidate vs shuffled | 95% CI | Median gradient cosine | Class |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| key | +0.002446 | [-0.000756, +0.005895] | +0.000882 | [-0.000354, +0.002395] | +0.000906 | neutral or inconclusive |
| value | +0.001083 | [-0.001940, +0.003780] | +0.000647 | [-0.000668, +0.002090] | -0.000361 | neutral or inconclusive |

These are held-out gold-answer NLL changes, not exact-match accuracy points.

Predefined outcome:

> No helpful target family at kind granularity.

Decision:

- do not run position or layer-band refinement under this hierarchy
- do not begin spectral distillation training
- do not interpret the small positive key point estimates as established utility

Bounded conclusion:

> At the final official CODI checkpoint, globally pooled key or value trajectory
> targets did not provide reproducible incremental answer-loss improvement over
> answer-only updates or shuffled KV targets.

This strengthens the distinction between predictable representational structure and
useful optimization signal.

## 23. Sparse answer-aligned gradient follow-up

The next fixed question is:

> Does useful KV supervision exist only as a sparse, consistently answer-aligned
> gradient component that is obscured when complete KV targets are distilled together?

The experiment changes the denoising object from KV activations to the auxiliary
parameter gradient. On a fresh calibration split it ranks trainable parameter
coordinates by repeated positive contributions:

```text
answer gradient * KV gradient
```

The top five percent with at least 60 percent positive batch consistency form one
frozen mask. A fresh update split produces full, learned-sparse, random-sparse,
shuffled-sparse, and complement KV gradient components. A third split measures
one-step answer loss. Auxiliary-gradient energy and total parameter-update norm are
matched.

The primary key gate requires the learned sparse component to beat answer-only, full
KV, random sparse, shuffled sparse, and complement updates with positive paired
bootstrap lower bounds. The word **only** additionally requires the complement to have
a non-positive upper confidence bound versus answer-only.

All 256 normalized question groups from the completed kind-level screen are excluded.
Value results are secondary and cannot change the primary key gate.

Implementation:

- `scripts/run_official_codi_kv_gradient_signal.py`
- `src/mech/kv_gradient_signal.py`
- `src/eval/official_codi_kv_gradient_signal_analysis.py`
- `docs/OFFICIAL_CODI_KV_GRADIENT_SIGNAL.md`
- `notebooks/kaggle_official_codi_kv_gradient_signal.ipynb`

Status:

> Completed on Kaggle. The primary sparse component was not supported.

No exact-match or long-run training claim is authorized by a positive one-step result.

## 24. Completed sparse answer-aligned gradient result

The fresh three-split experiment completed with:

- 128 calibration examples
- 128 update examples
- 128 held-out validation examples
- 32 update batches of size four
- zero normalized-question overlap among the three splits
- exclusion of all 256 question groups from the earlier kind-level screen
- five-percent learned coordinate mask
- cardinality-matched random mask
- matched auxiliary-gradient energy
- matched total parameter-update norm
- 10,000 paired update-batch bootstrap samples

Primary key results:

| Comparison | Mean advantage | 95% CI |
| --- | ---: | ---: |
| sparse vs no target | +0.000949 | [-0.000129, +0.002545] |
| sparse vs full KV | -0.001161 | [-0.003045, +0.000093] |
| sparse vs random sparse | +0.001914 | [-0.000228, +0.005461] |
| sparse vs shuffled sparse | +0.001798 | [+0.000089, +0.005044] |
| sparse vs complement | -0.000302 | [-0.000871, +0.000179] |
| full KV vs no target | +0.002110 | [-0.000109, +0.005113] |
| complement vs no target | +0.001250 | [-0.000109, +0.003128] |

Secondary value results:

| Comparison | Mean advantage | 95% CI |
| --- | ---: | ---: |
| sparse vs no target | +0.000631 | [-0.000122, +0.001712] |
| sparse vs full KV | -0.000888 | [-0.002356, +0.000050] |
| sparse vs random sparse | +0.001760 | [-0.000035, +0.005149] |
| sparse vs shuffled sparse | +0.001850 | [+0.000052, +0.005227] |
| sparse vs complement | -0.000234 | [-0.000667, +0.000058] |
| full KV vs no target | +0.001519 | [-0.000129, +0.004137] |
| complement vs no target | +0.000865 | [-0.000108, +0.002118] |

Gradient diagnostics:

| KV kind | Median sparse cosine | Positive batch fraction |
| --- | ---: | ---: |
| key | +0.018981 | 0.7812 |
| value | +0.039987 | 0.7812 |

Predefined outcome:

> Primary sparse component not supported.

The sparse mask beat the shuffled-pairing sparse condition for both keys and values.
This supports a narrow observation that the mask retained some example-pairing
information. It does not support useful sparse optimization signal because:

- sparse did not reproducibly beat answer-only
- sparse did not beat complete KV gradients
- sparse did not beat cardinality-matched random coordinates
- sparse did not beat the coordinate complement
- the complement was not demonstrably unhelpful

The positive gradient cosines therefore did not translate into the required held-out
utility pattern.

Bounded conclusion:

> At the final official CODI checkpoint, a fixed five-percent coordinate mask selected
> by calibration-batch answer-gradient alignment did not isolate a uniquely useful KV
> supervision component. It isolated pairing-sensitive structure, but not a component
> that improved held-out answer loss beyond the full, random, complement, and
> answer-only controls.

Decision:

- close this coordinatewise answer-alignment definition
- do not tune sparsity or consistency thresholds on these results
- do not begin distillation training with this mask
- preserve the result as a rigorous negative gate

This result does not rule out nonlinear, example-conditional, higher-rank, or
training-stage-dependent KV utility. Testing any of those would require a new
question, fresh data, and a separately preregistered gate.

## 25. Pivot to predictable KV-compression risk

Because paper-level KaVa weights and a complete reproduction path are not publicly
available, the project stepped back from method comparison and adopted a cheaper
inference-only question:

> Is KV-compression failure a stable, problem-specific property that could be
> predicted, or is it mainly sampling noise and ordinary problem difficulty?

The preregistered pilot uses DeepSeek-R1-Distill-Qwen-1.5B, first selects a dataset
whose full-cache accuracy is between 60 and 85 percent, and then compares full-cache
decoding with 90, 50, 25, and 10 percent generated-token cache retention on 150
disjoint questions. Its decisive viability test is whether compression failures are
nested as retention tightens and exceed the stochastic full-cache noise floor.

The first Kaggle execution is invalid and must not be interpreted. On a T4 it forced
the BF16-origin model into float16. All screened answers collapsed into one repeated
punctuation token until the 2,048-token limit, and recorded entropy was non-finite.
The resulting zero accuracies therefore failed numerical validation before they
could test the research hypothesis.

The repaired workflow now requires, before any scientific screen:

- automatic BF16 on supported GPUs and float32 on T4-class GPUs
- finite logits and predictive entropy
- exact greedy-token parity between the custom full-cache decoder and
  `transformers.generate`
- rejection of repeated-token collapse
- a fixed eight-example GSM8K parsing and accuracy gate
- a finite, non-degenerate compressed-cache smoke decode

Status:

> The third repaired Kaggle screen is numerically valid and complete. No candidate
> passed all preregistered dataset-selection criteria, so the compression sweep has
> not run.

The valid third-run screen used float32 on a T4. Exact decoder parity, finite logits
and entropy, functional GSM8K performance, and non-degenerate compressed decoding
all passed. All 158 screen records were complete and finite.

Observed screen:

| Dataset | Correct | Accuracy | Median generated tokens | Length-limited |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 50 / 64 | 78.125% | 405.0 | 0 / 64 |
| MATH-500 | 32 / 64 | 50.000% | 1,948.5 | 31 / 64 |
| AIME 2024 | 1 / 30 | 3.333% | 2,048.0 | 30 / 30 |

GSM8K failed only the minimum 512-token reasoning-length rule. MATH-500 failed the
60% minimum accuracy rule but was visibly constrained by the 2,048-token ceiling.
AIME failed accuracy and cannot leave 150 fresh problems. The primary research
question remains unanswered because there is not yet an eligible dataset on which
to measure compression-risk structure.

The earlier float16 and bfloat16 T4 runs remain diagnostic failures and must not be
used as scientific evidence or resume sources.

## 26. MATH-500 generation-budget diagnostic

Before moving to a larger model, the next bounded question is:

> Did MATH-500 fail the dataset screen because the 2,048-token generation ceiling
> truncated the 1.5B model's reasoning?

The paired diagnostic holds the model, revision, float32 precision, prompt, greedy
decoding, grader, and exact 64 questions fixed. Only `max_new_tokens` changes from
2,048 to 4,096. It records the paired accuracy change, answer flips, recovery among
previously length-limited examples, and remaining length truncation.

The original eligibility gate is not relaxed. The candidate cap must recover:

- 60% to 85% accuracy
- at least 512 median generated tokens
- at least 150 disjoint examples remaining

If the paired diagnostic passes, a fresh disjoint 64-question MATH-500 confirmation
must independently pass the same rule before the 150-question compression sweep is
authorized. If the paired diagnostic fails, no confirmation or compression run is
started.

Implementation:

- `configs/kv_risk_math_token_budget.yaml`
- `scripts/run_kv_risk_math_token_budget.py`
- `notebooks/kaggle_kv_risk_math_token_budget.ipynb`
- `docs/KV_RISK_MATH_TOKEN_BUDGET.md`

Status:

> Completed. The candidate cap remained binding and did not reach the unchanged
> accuracy gate.

Observed paired result:

| Metric | 2,048 tokens | 4,096 tokens |
| --- | ---: | ---: |
| Correct | 32 / 64 | 37 / 64 |
| Accuracy | 50.000% | 57.8125% |
| Length-limited | 31 / 64 | 20 / 64 |
| Median generated tokens | 1,948.5 | 1,948.5 |

The accuracy change was +7.8125 points with a 95% paired-bootstrap interval from
-1.5625 to +17.1875 points. There were seven incorrect-to-correct and two
correct-to-incorrect changes. Among the 23 previously length-limited incorrect
answers, seven recovered.

All 33 previously completed EOS generations reproduced exactly. Every one of the
31 censored 2,048-token sequences was an exact prefix of its 4,096-token
continuation. The paired change is therefore a deterministic token-budget effect,
not sampling noise.

The original minimum requires 39 correct answers out of 64. The candidate produced
37, and 20 questions still reached the new ceiling. Fresh confirmation and the
compression sweep were correctly blocked.

## 27. Final 8,192-token eligibility extension

The final bounded question for the 1.5B configuration is:

> When only the 20 still-censored MATH-500 questions are allowed to continue from
> 4,096 to 8,192 tokens, does the composed 64-question screen cross the unchanged
> eligibility gate?

The 44 completed greedy outputs are reused. Only the 20 length-limited questions
are regenerated. Each new output must preserve the complete 4,096-token sequence
as an exact prefix before it can enter the composition.

The unchanged gate still requires:

- 60% to 85% accuracy
- at least 512 median generated tokens
- at least 150 disjoint questions remaining

If the composed result passes, a fresh disjoint 64-question confirmation runs at
8,192 tokens. Only a passing fresh confirmation authorizes the 150-question
compression-risk pilot. Any failure closes this 1.5B configuration. No further
token-cap escalation is allowed.

Implementation:

- `configs/kv_risk_math_token_budget_8192.yaml`
- `scripts/run_kv_risk_math_token_budget_8192.py`
- `notebooks/kaggle_kv_risk_math_token_budget_8192.ipynb`
- `docs/KV_RISK_MATH_TOKEN_BUDGET_8192.md`

Status:

> Implemented and statically validated. Kaggle execution remains.

## 28. Historical teacher-colon versus student-pre-cue TSV-C diagnostic

The earlier spectral work operated on teacher and student KV trajectories. It established
stable low-rank relationships, but the learned directions were not more causally useful
than matched random directions. R-KV was not superior to uniform token selection, and
complete, sparse, and answer-gradient-selected KV targets failed their held-out utility
gates. Those results do not directly test CODI's native distillation target because CODI
matches hidden states at an answer-cue endpoint rather than KV trajectories.

The bounded historical question was:

> At the paper-accuracy official CODI checkpoint, do leading singular directions of the
> teacher-student endpoint hidden-state residual provide specifically useful answer
> updates beyond answer-only, random, bottom-spectrum, and shuffled controls?

The original TSV-C method compresses per-layer weight-difference matrices. This is an
explicit activation-space adaptation and will be called TSV-C-inspired filtering.

The implementation separated transformer depth, but a later pinned-source audit found
that its trajectory alignment was not the native CODI match:

- `endpoint_all_layers` used teacher colon states but student pre-cue latent-six states.
- `endpoint_layer11` used the same cross-location pairing at block 11.

The fixed historical contract used the official 43.67-percent GSM8K checkpoint, 5,000 calibration
questions, 256 update questions, 256 disjoint validation questions, uncentered per-layer
SVD, rank 77, equal auxiliary-gradient norms, equal total-update norms, and 10,000 paired
update-batch bootstrap samples. Its preregistered rule required the all-layer primary
gate to pass four prespecified comparisons with Holm correction.

Implementation:

- `configs/official_codi_gpt2.yaml` under `endpoint_tsvc`
- `scripts/collect_official_codi_endpoint_tsvc.py`
- `scripts/run_official_codi_endpoint_tsvc_utility.py`
- `scripts/analyze_official_codi_endpoint_tsvc.py`
- `notebooks/kaggle_official_codi_endpoint_tsvc.ipynb`
- `docs/OFFICIAL_CODI_ENDPOINT_TSVC.md`

Status:

> Complete as a historical cross-location diagnostic. It cannot authorize or block a
> training study based on CODI's native answer-cue endpoint.

Completed result:

- `endpoint_all_layers` failed all four required comparisons. Learned top-77 versus
  answer-only had mean advantage -0.000466 with 95% CI [-0.001758, +0.000791].
  Learned versus random, bottom-spectrum, and shuffled controls was also non-positive
  or inconclusive. Median answer-gradient cosine was -0.005703.
- `endpoint_layer11` showed a small positive mean advantage over answer-only of
  +0.000215, but its 95% CI [-0.000035, +0.000526] crossed zero and Holm p was 0.3508.
  It did not beat random, bottom-spectrum, or shuffled controls. Median cosine was
  +0.000565.
- Learned top-77 did not beat the full endpoint target in either scope.
- The combined decision was `endpoint_tsvc_not_supported`, with
  `training_authorized=false`.

Durable provenance:

- Kaggle dataset:
  `jonraza15/official-codi-endpoint-tsv-c-inspired-experiment`, version 1
- Run commit: `8b70b0d95e6b28ce3dfc512929bd0ac942f8a427`
- All files in the purpose-built export passed SHA-256 verification after download on
  2026-08-03.

Bounded historical conclusion:

> At the final official CODI checkpoint, the leading rank-77 directions of the
> teacher-colon versus student-pre-cue latent-six residual did not isolate locally useful
> update signal beyond matched controls. This does not answer the native CODI endpoint
> question because the student was sampled before EOT and the answer cue, the embedding
> state was excluded, and the diagnostic used L1 rather than the released SmoothL1 loss.

## 29. Corrected source-native CODI endpoint TSV-C experiment

A pinned-source audit of official CODI revision
`2c2314662c63e9f482ebc46614ffe9af17a241e5` corrected three material details:

1. The student target is gathered at the colon in its decoded `The answer is:` cue,
   after six continuous latents and EOT, not at latent step six itself.
2. The released distillation loop consumes all 13 Hugging Face hidden-state entries for
   GPT-2, including the embedding state and 12 transformer-block outputs.
3. The released GPT-2 run uses SmoothL1, divides each state loss by the unbiased teacher
   standard deviation, and averages over the 13 states.

The corrected primary question is:

> At the paper-accuracy official CODI checkpoint, do leading rank-77 singular directions
> of the source-native teacher-student answer-cue endpoint residual produce locally
> useful answer updates beyond answer-only, random, bottom-spectrum, and shuffled-pairing
> controls?

The corrected primary scope is `endpoint_all_states`. The secondary localization scope
is `endpoint_layer11`, which maps to hidden-state tuple index 12. The original completed
artifacts are preserved; the corrected experiment writes to separate output, report,
and log trees.

A mandatory four-example parity gate runs before calibration. It verifies matching
colon token IDs, `[B,13,768]` finite tensors, detached teacher states, native loss error
at most `1e-7`, parameter-gradient relative L2 error at most `1e-6`, and gradient cosine
at least `0.999999`. A failed parity check blocks the 5,000-example calibration.

The remaining contract is frozen at 5,000 calibration examples, 256 update examples,
256 paired validation examples, rank 77, sampling seed 11, random-basis seed 20260803,
64 paired update batches, equal auxiliary-gradient norms, equal total-update norms, and
10,000 paired bootstrap samples. Only a primary all-state gate pass can authorize a
separately preregistered training experiment.

Implementation:

- `configs/official_codi_gpt2.yaml` under `endpoint_tsvc_corrected`
- `src/mech/endpoint_tsvc_corrected.py`
- `scripts/collect_official_codi_endpoint_tsvc_corrected.py`
- `scripts/run_official_codi_endpoint_tsvc_corrected_utility.py`
- `scripts/analyze_official_codi_endpoint_tsvc_corrected.py`
- `notebooks/kaggle_official_codi_endpoint_tsvc_corrected.ipynb`
- `docs/OFFICIAL_CODI_ENDPOINT_TSVC_CORRECTED.md`

Status:

> Implemented. Local syntax validation is complete. Kaggle parity smoke, calibration,
> utility execution, and the combined decision remain pending.

## 30. Rank-matched endpoint retention experiment

The three endpoint selectors rank residual directions differently: by residual Gram
energy, by split-stable alignment with the gold-answer loss gradient
(answer-conditioned), and by induced trainable-parameter-gradient alignment
(parameter-aware). The retention experiment asked whether each rule's selected
directions are sufficient for accuracy, and whether they beat the directions the
same rule discards.

It filters the **teacher auxiliary residual during fine-tuning**. It does not reduce
the student's 768-dimensional inference state.

Completed on Kaggle as `jonraza15/official-codi-endpoint-rank-matched-experiment`:
24 runs, three seeds, rank three at states 11 and 12, one fresh training partition,
all 109 exported checksums verified.

| Selector | Selected accuracy | Selected − full | Selected − complement | Selected − answer-only |
| --- | ---: | ---: | ---: | ---: |
| Energy | 43.341% | −0.076 pp | −0.051 pp | −0.025 pp |
| Answer-conditioned | 43.290% | −0.126 pp | −0.177 pp | −0.076 pp |
| Parameter-aware | 43.417% | 0.000 pp | +0.025 pp | +0.051 pp |

Every interval includes zero. Two separable conclusions:

- **Auxiliary-target compression: yes.** Six directions preserve the full residual
  target's accuracy within the registered one-point non-inferiority margin.
- **Accuracy-critical directions: no.** No selection beat its own discarded
  complement or plain answer-only training.

The decisive control is `full target − answer-only = +0.051 pp`, 95% interval about
[−0.303, +0.379]. Even the complete two-block residual target was not shown to add
accuracy over ordinary answer training.

Throughput was statistically flat at about 34 examples/second across arms, as the
contract predicted. Top-k changed only the training target, never inference compute.

## 31. Interpretation correction: marginal training utility is not causal ablation

The retention result initially read as "six directions are responsible for 43%
accuracy, and so is their complement", which is incoherent. The correction is
recorded here because it is a reasoning error the project must not repeat.

The 43% was already present in the frozen checkpoint before either arm trained.
Every arm kept the full pretrained weights, all 12 blocks, all 768 dimensions at
inference, and the ordinary gold-answer loss. Only the auxiliary residual term was
filtered, and that term had no demonstrated marginal utility, so removing most of it
changed almost nothing.

Two further reasons selected and complement behaved alike:

1. Orthogonality in the 768-dimensional residual space does not imply independent
   parameter updates. With gradients `g_sel = Jᵀ P r` and `g_comp = Jᵀ (I − P) r`,
   the model Jacobian can map orthogonal activation directions onto overlapping
   LoRA updates.
2. Gradient norms were matched, so a naturally weak six-direction signal was
   inflated to full strength. That tests directional quality at equal update
   magnitude, not how much information the directions naturally carry.

No projection bug was found: the selected loss used `P r`, the complement used
`r − P r`, synthetic reconstruction tests passed, and arms produced different
predictions.

Stated plainly:

> The experiment was a marginal auxiliary-training comparison and was briefly
> interpreted as a causal hidden-representation ablation. It answers "can six
> residual directions replace the full residual target without losing accuracy?"
> It does not answer "are these directions responsible for the model's accuracy?"

Corrective principle:

> To test whether directions contribute to accuracy, do not retrain each arm. Start
> from the identical frozen checkpoint and intervene during inference.

## 32. Frozen-checkpoint answer-colon ablation and 232-arm accuracy localization

Following §31, a frozen-checkpoint inference intervention was implemented at the
forced answer cue. No parameter is updated; the only difference between arms is a
temporary hidden-state edit at the colon. An early smoke run failed its cue-reach
assertion at 0% coverage and was fixed by forcing the teacher-forced cue and
tracking which questions actually reach it.

The 232-arm localization run completed as
`jonraza15/codi-answer-colon-accuracy-localization`: 948 files SHA-256 verified,
1,319 paired questions per arm in one order, 100% endpoint coverage, and the saved
report reproduced from raw JSONL.

| Arm | Accuracy | Loss | 95% CI | Matched-random p |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 43.290% | — | — | — |
| Energy joint | 43.063% | 0.227 pp | −0.682 to 1.137 | negative control passes |
| Answer-conditioned joint | 41.622% | 1.668 pp | 0.455 to 2.881 | 0.139 |
| Parameter-aware joint | 40.637% | 2.654 pp | 1.365 to 3.942 | 0.0495 raw, 0.099 Holm |

Both selected subspaces genuinely harm accuracy (McNemar Holm `p = 0.00517` and
`p = 0.0000388`). Neither passed the stronger activation-energy-matched random gate:
13 of 100 and 4 of 100 controls were at least as damaging.

Two findings carried forward:

- **Localization to state 12.** Parameter-aware state 12 lost 1.516 points
  (Holm `p = 0.0066`); state 11 lost 0.834.
- **Interaction, not individual necessity.** The joint effect of 2.654 points
  greatly exceeds the 0.379-point sum of its six single-direction losses. Removing
  state-12 PC10 alone cost 0.152 points, yet retaining it while removing the other
  five rescued 0.986 points.

Recorded caveat: calibration matching was numerically near-exact (relative energy
error about `3.3e-16`) but did not transport to GSM8K. Random evaluation RMS
exceeded selected RMS by about 14.2% and 9.8% (answer-conditioned, states 11/12)
and 3.1% and 18.9% (parameter-aware). The null was therefore conservative.

## 33. Completed parameter-aware state-12 confirmation

One preregistered primary hypothesis, calibrated on 2,048 disjoint GSM8K **train**
questions, 500 selected-orthogonal energy-matched controls, 502 paired full-test
arms, no selector multiplicity correction, and an added guard withholding the result
if median random/selected evaluation RMS differed by more than 10%.

Completed as `jonraza15/confirm-parameter-aware-state-12-at-codis`.

> Status: **`not_confirmed`**. `parameter_aware_state12_confirmed = false`.

Five of six conditions passed:

| Condition | Value | Outcome |
| --- | --- | --- |
| Positive in both deterministic halves | 1.5152 / 1.5175 pp | pass |
| Bootstrap 95% lower bound above zero | CI [0.531, 2.502] pp | pass |
| One-sided exact McNemar `p ≤ 0.05` | 0.00227 | pass |
| Empirical matched-random `p ≤ 0.05` | **0.1557** | **fail** |
| Calibration matching | rel. error 3.4e-16, overlap 1e-32 | pass |
| Evaluation RMS transport within 10% | ratio 1.0497 | pass |

Primary arm: 43.290% → 41.774%, a 1.5163-point loss, 33 correct-to-wrong and 13
wrong-to-correct, 100% cue coverage, state 11 untouched.

Matched-random null over 500 replicates: mean 0.576 pp, median 0.379 pp, 95th
percentile 2.047 pp, maximum 2.502 pp. **77 of 500 controls were at least as
damaging**, placing the selection at the 84.6th percentile.

Two points of interpretation:

1. **The transport confounder was eliminated and the result still failed.** The 10%
   RMS gate was added specifically because the discovery null was conservative. Here
   the ratio is 1.0497, comfortably inside the band. The failure cannot be blamed on
   an unmatched null.
2. **The 1.5163-point effect is not an independent replication.** It equals the
   discovery value because it is the same deterministic computation: same frozen
   checkpoint, same PCs 9/10/32, same forced cue, same 1,319 questions, greedy
   decoding. The confirmation's novelty is entirely in the null.

Bounded conclusion:

> The parameter-aware state-12 rank-three subspace is causally involved in CODI's
> answer prediction, but it is not distinguishable from energy-matched random
> subspaces at the same state. About one in six random rank-three subspaces does as
> much or more damage.

## 34. Diagnosis of what the state-12 design could detect

Before proposing another selector, a source audit asked what the confirmation was
capable of measuring. Three properties bound it independently of the hypothesis.

**The causal channel is nearly absent.** `state_module_map` records state 12 as
`transformer.ln_f output after transformer.h[11]`. GPT-2 builds every block's
key/value inside the block and `ln_f` runs after all twelve, so a state-12 edit
never enters the cache. The diagnostics confirm one intervened forward pass per
batch (`calls_by_state {12: 42}` at `eval_batch_size 32` over 1,319 questions) on
`hidden[:, -1, :]` only. The whole pathway is

```text
Δ logits = − W_U · U Uᵀ (h − μ)      at exactly one token
```

with no propagation. Arbitrary control of the state-12 vector would drive accuracy
to roughly zero, so about 43 points of headroom exist; the rank-three
mean-preserving removal realised 1.52, about 3.5% of it. State 11 does reach the
cache, which matches the observed 0.834 / 1.516 / 2.654-point pattern.

**The outcome discarded most of the measurement.** Binary exact match on 1,319
questions expresses the effect as ~20 flipped answers against a null spread of ~±9,
and the empirical gate required beating the top 25 of 500 controls, i.e. 2.05 points.

**The selection criterion did not match the test statistic.** Every selector used a
first-order gradient score and was then tested with a finite rank-three projection.

Two smaller defects: the mean-preserving edit `h − U Uᵀ(h − μ)` removes only
variance along `U` and is blind to the constant component; and every arm asked
necessity while "responsible for the majority of accuracy" is a sufficiency claim.

One quantity is worth recording because it is not nothing: the selected subspace
removed 4.97% **less** activation energy than the median control yet caused 2.6× the
mean damage (1.52 pp vs 0.58 pp). A real but weak directional effect exists; the
design simply required it to beat the extreme tail of a 500-draw null.

## 35. The six-selector pattern

Six independent selection criteria have now been tested against matched controls on
paper-accuracy checkpoints, and all six failed:

| Selector | Comparison it lost | Section |
| --- | --- | --- |
| R-KV token selection | uniform selection | §11 |
| Boundary-aware R-KV | structured controls | §12 |
| Learned rank-four KV spectral | energy-matched random | §13 |
| Pooled key/value KV targets | answer-only and shuffled | §22 |
| Sparse answer-aligned gradient mask | full, random, complement, answer-only | §24 |
| Parameter-aware state-12 endpoint | energy-matched random | §33 |

The consistent pattern is that structure which is stable, predictable, or even
causally involved has repeatedly failed to be *specific*: matched controls do about
as well. This is itself a substantive result and is consistent with the instructor's
original skepticism in §5.

Added to the decisions in §17:

- Do not propose a seventh heuristic selector scored by a first-order criterion and
  tested by a finite projection without first showing that the design can detect a
  known-present effect.
- Do not read a joint causal effect as evidence that its individual directions are
  necessary when the joint effect greatly exceeds the sum of the singles.
- Do not interpret a marginal auxiliary-training comparison as a causal ablation
  (§31).
- Report whether a matched control's energy target is attainable at all; an
  unattainable target makes the null conservative rather than matched.

## 36. Answer-colon margin geometry and effective dimensionality

The next experiment does not add a selector. It removes the five diagnosed defects
and asks whether the earlier negatives were underpowered or genuine.

The enabling observation is that GPT-2's `lm_head` is bias-free and consumes the
`ln_f` output, so a state-12 edit is exactly

```text
z' = W h' = z − (W U)(Uᵀ (h − centre))
```

Caching one colon state per question therefore turns every state-12 arm into a
matrix product rather than a full greedy decode, which is what makes continuous
outcomes, a full rank sweep, and hundreds of matched controls affordable. A parity
gate checks the analytic first token against the released decoder before any sweep
is allowed to run.

**States are captured from the generation path itself.** The first implementation
cached colon states with the released *training* encoder and compared them against
generation; the parity gate correctly blocked the run at 89.06% first-token
agreement. The two paths differ in at least three ways: the generator normalises the
question while the row formatter does not (336 of 1,319 test questions differ), the
answer cue is tokenised with a leading space in one path only, and left padding
shifts GPT-2's absolute position ids for every row in a chunk when the longest
sequence changes. Rather than patch each difference and remain one undiscovered
divergence away from silently invalid results,
`OfficialCODIEndpointStateCollector` observes the answer-cue forward pass during
real generation, so the cached state *is* the state the decoder consumed and parity
reduces to the `lm_head` claim alone.

A consequence worth recording: under `max_new_tokens=1` with the forced cue the
answer never enters the model, so collection needs no reasoning trace, no released
row formatting, and no answer-eligibility filter. That also removes an earlier
complication — GSM8K test rows 489 (`-10`) and 1113 (`-3`) are rejected by
`official_codi_answer_is_eligible`, which is a *training* filter; dropping them
would have broken pairing with every completed 1,319-question experiment.

Corrections, one per defect:

| Defect | Correction |
| --- | --- |
| Outcome too coarse | Primary outcome is per-example gold-answer NLL; margin and top-1 reported alongside |
| Selector ≠ test statistic | Primary subspace is the closed-form maximiser of the measured objective: top-`k` eigenvectors of `sym(E[c gᵀ])` |
| Wrong basis | Adds `readout` (numeric-token unembedding) and `answer_nll` families |
| Mean-preserving edit only | `mean`, `zero` and `resample` semantics separated |
| Necessity only | Retention arms sweep rank 1 → 512 for sufficiency |
| No propagation | State-11 and all-position generation arms |

Two preregistered gates. **Primary 1** fixes rank three in advance, because that is
the rank the failed confirmation tested, and asks whether the closed-form margin
subspace beats energy-matched selected-orthogonal random subspaces on held-out
gold-answer NLL. **Primary 2** reports the smallest rank whose retained subspace
preserves 90% of baseline first-token accuracy, per family, with the random curve
alongside. Primary 2 never gates Primary 1.

If an explicitly optimal subspace still fails Primary 1, the negative can no longer
be attributed to a weak heuristic or to a coarse outcome. That is the point of
running it.

Calibration is 2,048 GSM8K **train** questions at seed 89 with proven zero
normalized-question overlap with the 1,319-question test set. No test label or test
activation enters any fit.

Implementation:

- `docs/OFFICIAL_CODI_ENDPOINT_MARGIN_GEOMETRY.md`
- `configs/official_codi_gpt2.yaml` under `endpoint_margin_geometry`
- `src/mech/endpoint_margin_geometry.py`
- `scripts/collect_official_codi_endpoint_margin_states.py`
- `scripts/run_official_codi_endpoint_margin_sweep.py`
- `scripts/run_official_codi_endpoint_margin_generation.py`
- `src/eval/official_codi_endpoint_margin_geometry_analysis.py`
- `scripts/analyze_official_codi_endpoint_margin_geometry.py`
- `notebooks/kaggle_official_codi_endpoint_margin_geometry.ipynb`
- `tests/test_endpoint_margin_geometry.py`

`src/models/official_codi.py` gained one additive change: the generator honours an
`applies_to_all_positions` attribute on an endpoint intervention, defaulting to
absent/`False`, so completed experiments are unchanged.

Status:

> Implemented and locally validated: 273 tests pass, 1 skipped. Kaggle execution of
> the parity gate, collection, sweep, generation arms, and the two gates remains.

No training study is authorized by this experiment, and no inference speed claim is
made: a directional projection does not narrow GPT-2's width or skip a block.


## 37. Completed margin-geometry run and the accuracy-bearing PC band

The margin-geometry experiment executed on Kaggle
(`jonraza15/codi-answer-colon-margin-geometry`, 6,912 analytic arms, 14 generation
arms). The parity gate returned **1.0 on all 1,319 questions**, confirming that
capturing colon states from the generation path makes the closed-form state-12
evaluator exact.

### Primary 1 passed, and is a positive control rather than a discovery

Removing the closed-form rank-three margin subspace raised held-out gold-answer NLL
by +21.215 (95% CI [+19.764, +22.728]), beating all 200 energy-matched controls
(empirical `p = 0.00498`, random mean +9.65). Matching was clean: maximum relative
energy error 9.9e-4, maximum selected overlap 6.1e-16.

This is close to tautological — the subspace is fit with gold labels to maximise the
quantity then measured — and its value is as a **positive control**. It establishes
that the design can detect a real effect, so the six previous negatives were not the
product of a test too weak to find anything.

### The substantive result: variance rank and answer contribution are near-orthogonal

Retaining only principal components of the *student colon state's own* covariance
(fit on 2,048 GSM8K-train questions, applied to held-out GSM8K test), scored on
first-token accuracy against a 0.4208 baseline:

| subspace | dims | % of variance | retain → frac. | remove → Δ |
| --- | ---: | ---: | ---: | ---: |
| PC 0 | 1 | 66.25% | 0.063 | 0.15 pp |
| PC 0–3 | 4 | 82.31% | 0.067 | 1.74 pp |
| PC 4–15 | 12 | 7.49% | **0.506** | 13.57 pp |
| PC 4–31 | 28 | 11.31% | **0.859** | **32.22 pp** |
| PC 32–767 | 736 | 6.38% | 0.222 | 3.11 pp |

Twelve of 768 dimensions carry the majority of the accuracy; 28 carry 86% and their
removal collapses accuracy from 42.1% to 9.9%. The leading component holds two thirds
of all variance and 6% of the accuracy.

This plausibly explains the six-selector pattern of §35. Every prior selector searched
the teacher-minus-student **residual** basis or a gradient-alignment criterion, and any
variance-ranked method lands on PC 0–3, which do almost nothing for the answer. For
scale, the `answer_conditioned` and `parameter_aware` rank-three subspaces cost 1.7 and
2.4 points; this band costs 32.

Robustness: filling the complement from a different question rather than the
calibration mean still retains 0.728 of baseline; PCs fit on disjoint calibration
halves retain 0.834 and 0.852 with mean principal-angle cosine 0.979 and 0.968.

### Necessity and sufficiency dissociate

The margin subspace is devastating to remove (34.95 points) yet saturates at 0.41 of
baseline when retained at any rank. Energy PCs are the reverse. "Most damaging to
delete" and "carries the computation" are different objects, which is why both arm
types exist.

### Two defects found in this run

1. **The forced-cue baseline no longer reproduces, and precision is NOT the cause.**
   The completed run's generation arms defaulted to `--precision auto`, and the
   baseline came out at 40.41% against a historical 43.29%. That was initially
   attributed to `auto` resolving through `torch.cuda.is_bf16_supported()` to
   emulated bfloat16 on T4-class hardware. **That diagnosis was wrong.** Re-running
   with `--precision float32` explicitly resolved gives 40.56% - two answers away
   from the `auto` value. Every configuration input is identical between the two
   runs (checkpoint SHA, answer cue and its token ids, batch size, example count,
   max_new_tokens), so the remaining variable is the execution environment; the
   Kaggle image demonstrably changed, since it also broke peft/torchao. The
   reproduction gate cannot detect this because it reads a summary computed on the
   older image. The next run therefore re-decodes the native full-GSM8K gate on the
   current environment and references that fresh value. Note that 40.56% falls just
   below the preregistered 0.437 +/- 0.03 band, so the fresh gate may itself fail -
   which would be the finding. Precision is still pinned to float32 for
   reproducibility, but it is not the explanation.

   Superseded text kept for the record: the original entry read as follows.
   **Generation arms used the wrong precision.** Collection pinned float32 but the
   generation runner defaulted to `auto`, which on T4-class GPUs resolves through
   `torch.cuda.is_bf16_supported()` to emulated bfloat16 and moved the forced-cue
   baseline from 43.29% to 40.41%. Every configuration was otherwise identical. The
   analytic tier is unaffected. Precision is now pinned in the config, the runner
   default and the notebook, and the baseline arm asserts its own accuracy against
   the reproduction gate.
2. **The reference selectors reversed their earlier verdict, and this is not yet
   settled.** `parameter_aware` and `answer_conditioned` both beat their matched-random
   nulls here (`p = 0.00498` on first-token accuracy), against the `not_confirmed` of
   §33. But this run's null is 26 times weaker than the confirmation's (mean +0.022 pp
   versus +0.576 pp; maximum +0.758 versus +2.502) at comparable removed energy. Two
   explanations remain live: §33 fit its centering mean and matching covariance with
   the training encoder, whose colon states are now known to differ from the
   generation states; or this run's sampler is weak at low energy targets. **Do not
   cite the reversal until that is resolved.**

## 38. Exact-match confirmation of the PC band

The §37 band result is first-token accuracy. The confirmation re-tests it with full
greedy decoding and numeric exact match at pinned float32, with three gates frozen in
advance: sufficiency (retain PC 4–31 ≥ 0.70 of baseline), dissociation (retain PC 0–3
≤ 0.20, with a positive paired advantage for the primary band) and necessity (remove
PC 4–31 ≥ 20 points, positive bootstrap lower bound, exact McNemar `p ≤ 0.05`), plus a
baseline-drift guard.

Twelve arms: baseline, five retention bands, two removals, four descriptive random
rank-28 retention controls. Band targets are appended after every existing registry
target so the completed margin-geometry arms remain bit-identical.

Implementation is listed in `docs/OFFICIAL_CODI_ENDPOINT_BAND_CONFIRMATION.md`.

Status:

> Implemented and locally validated; the production band path reproduces the
> analytic numbers exactly. A first Kaggle attempt stopped at the baseline-drift
> guard (0.4056 versus the historical 0.4359), which is the guard working as
> intended. The notebook now re-establishes the reproduction gate on the current
> image before any arm runs, and every arm references that fresh value. Execution
> remains.

No training study is authorized and no inference-speed claim is made.


## 39. The execution environment stopped reproducing the checkpoint

The band-confirmation run re-decoded the native full-GSM8K reproduction gate on the
current Kaggle image rather than trusting the attached summary. It failed:

| quantity | recorded (2026-08-03) | current image |
| --- | ---: | ---: |
| native GSM8K accuracy | 0.43669 | **0.37225** (491 / 1,319) |
| forced-cue baseline | 0.43290 | 0.40561 |
| accuracy gate | passed | **failed** |

The eval manifest beside the original summary records the environment that
reproduced the published number:

```text
transformers 4.52.4   peft 0.15.2   datasets 3.6.0
huggingface_hub 0.32.4   torch 2.10.0+cu128
```

Torch is unchanged on the current image; transformers and peft are much newer, which
is also what broke peft/torchao. `src/models/official_codi.py` documents that its
cache handling is written against Transformers 4.52 legacy-tuple semantics, and the
CODI latent loop threads `past_key_values` through six hand-rolled forward passes, so
a change there degrades the model silently instead of raising.

Corrective principle:

> Pin transformers, peft, datasets and huggingface_hub when reproducing any
> official-checkpoint result, and re-decode the reproduction gate in the environment
> actually in use. A stored gate summary certifies the image it was computed on, not
> the one currently running.

### Consequence for the completed margin-geometry run

Section 37's results were produced on this non-reproducing environment. They remain
internally consistent — states captured from that image's own generation, parity 1.0
against its own decoder — but the model instance scores 0.3723 natively rather than
0.4367, so **its absolute numbers are not comparable to any earlier experiment**.

Partially resolved by §40: the PC-band bases fitted on that image's colon states were
applied on the pinned, reproducing environment and behaved as predicted, so the band
*geometry* transfers across the environment change. Only the §37 absolute figures
remain tied to the degraded instance. This also supplies a third candidate explanation for the
§37 reference-selector reversal, alongside the two already recorded.

The confirmation notebook now installs the pinned versions before anything runs,
asserts them in a fresh subprocess, and only then re-decodes the gate.


## 40. Confirmed: a 28-dimensional band carries CODI's answer accuracy

The exact-match confirmation completed on the pinned, reproducing environment
(`jonraza15/exact-match-confirmation-of-codis-pc-band`).

Environment and gate, both re-established in-run:

```text
transformers 4.52.4   peft 0.15.2   datasets 3.6.0
huggingface_hub 0.32.4   torch 2.10.0+cu128
native GSM8K 0.435936   accuracy gate: passed
forced-cue baseline 0.433662 (572 / 1,319), drift 0.0023
```

The pins restored the checkpoint from 0.3723 to 0.4359, confirming §39.

### Result: `band_confirmed`

All three preregistered gates passed on numeric exact match.

| Gate | Requirement | Observed |
| --- | --- | ---: |
| Sufficiency | retain PC 4–31 ≥ 0.70 of baseline | **0.878** |
| Dissociation | retain PC 0–3 ≤ 0.20; primary − control lower bound > 0 | **0.061**; +35.41 pp, CI [+32.75, +38.13] |
| Necessity | remove PC 4–31 ≥ 20 pts, lower bound > 0, McNemar ≤ 0.05 | **30.48 pp**, CI [+27.90, +33.13], p = 5.7e-101 |

Full arm set, against a 0.4337 baseline:

| arm | dims | % variance | accuracy | retained |
| --- | ---: | ---: | ---: | ---: |
| retain PC 0–3 | 4 | 82.31 | 0.0265 | 0.061 |
| retain PC 4–15 | 12 | 7.49 | 0.2191 | 0.505 |
| retain PC 4–31 | 28 | 11.31 | 0.3806 | **0.878** |
| retain PC 0–31 | 32 | 93.62 | 0.4094 | 0.944 |
| retain PC 32–767 | 736 | 6.38 | 0.1130 | 0.260 |
| remove PC 4–31 | 28 | 11.31 | 0.1289 | 0.297 |
| remove PC 0–3 | 4 | 82.31 | 0.4185 | 0.965 |
| random rank-28 retention x4 | 28 | matched | 0.0281–0.0379 | 0.065–0.087 |

### The analytic tier predicted exact match closely

| arm | analytic (first token) | exact match |
| --- | ---: | ---: |
| retain PC 0–3 | 0.067 | 0.061 |
| retain PC 4–15 | 0.506 | 0.505 |
| retain PC 4–31 | 0.859 | 0.878 |
| retain PC 0–31 | 0.926 | 0.944 |
| retain PC 32–767 | 0.222 | 0.260 |
| remove PC 4–31 | 32.22 pp | 30.48 pp |
| remove PC 0–3 | 1.74 pp | 1.52 pp |

Every arm agrees within about two points, which validates the closed-form state-12
evaluator as a cheap and faithful proxy: it made 6,912 arms affordable and its
predictions held under real greedy decoding.

### Bounded conclusion

> At the frozen official CODI GPT-2 checkpoint's forced answer cue, 28 of the 768
> principal components of the student colon state — carrying 11.3% of its variance —
> are sufficient to preserve 87.8% of numeric exact-match accuracy and necessary in
> the sense that removing them costs 30.5 points. Twelve components preserve the
> majority. The leading four components hold 82.3% of the variance and 6.1% of the
> accuracy, and removing them costs 1.5 points.

Verification: 12 arms, one checkpoint, identical question order, 100% cue coverage,
all float32, 54 of 54 SHA-256 checksums intact, and every reported figure recomputed
from the raw prediction files.

### Why this matters for §35

Variance rank and answer contribution are close to unrelated at this endpoint. Every
one of the six selectors in §35 searched the teacher-minus-student residual basis or a
gradient-alignment criterion; a variance-ranked method lands on PC 0–3, which are
nearly irrelevant. For scale, the `answer_conditioned` and `parameter_aware`
rank-three subspaces cost 1.7 and 2.4 points, while this band costs 30.5.

This is the project's first positive, preregistered, exact-match result.

### Remaining limits

- The band bases were fitted on colon states cached on the §39 non-reproducing image.
  They transfer, but a clean re-derivation on pinned-environment states would remove
  the last dependency on that run.
- Retention replaces the complement with the calibration mean, an off-manifold
  intervention. Under the stricter donor control the analytic tier retained 0.728
  rather than 0.859; that control has not been repeated on exact match.
- The random rank-28 arms are descriptive (four replicates). The specificity null
  rests on the analytic tier's 200 energy-matched replicates.
- Bounded to official CODI GPT-2, state 12, the forced answer cue, linear subspaces,
  and GSM8K. No distillation target is authorized and no inference-speed claim is
  made: a projection hook adds work and does not narrow the model.

## 41. Exploratory: the correctness split is nearly orthogonal to the accuracy band

§40 established which directions *determine* the answer. A natural follow-up asks a
different question of the same colon states: the covariance was built class-blind, so
it mixes questions the model got right with ones it got wrong. What happens if the 768
dimensions are split by correctness instead?

Everything in this section is **exploratory** — computed on cached states, with no
preregistration, and reported here so the preregistered version in §42 can be read as a
confirmation rather than as discovery. Figures are first-token accuracy at the forced
cue unless stated.

### The class geometry

Writing `d = mean(correct) − mean(incorrect)` over 2,048 calibration states:

| quantity | value |
|---|---:|
| `‖d‖` | 26.22 |
| between-class variance | 152.26 |
| total variance | 3,828 |
| **between-class share** | **3.98%** |
| share of `d` inside PCs 0–3 | **97.13%** (PC1 alone 57.12%) |
| share of `d` inside PCs 4–31 | ~2% |

96.02% of the variation at the answer cue is *within* class. Right and wrong answers
are not two separated clouds; they are one cloud with a slight offset.

Against a 200-replicate random-split null with the same class sizes: median leading-band
share 70.56%, **0/200 replicates reach 97.13%**, and `‖d‖` is **11.7×** the random
median. So the direction is real and it is genuinely concentrated — but concentrated in
PCs 0–3, which §40 showed carry 82.3% of the variance and 6.1% of the accuracy.

**The correctness signal lives almost entirely in the directions that cannot change an
answer.** PC1 shifts all 50,257 logits by roughly +27.1 with a spread of 0.46; a
near-uniform lift cannot move an argmax.

### Three uses of the resulting subspace, tested

**Detect.** Held-out AUC for predicting correctness from the state:

| detector | AUC |
|---|---:|
| projection on `d̂` | 0.700 |
| distance from a correct-only subspace, k = 4 / 28 / 64 | 0.237 / 0.293 / 0.306 |
| **the model's own margin** (top logit − runner-up) | **0.874** |

The projection carries real signal, and the model's own confidence carries more of it,
for free. The reconstruction-error rows are *inverted* — being further from the
"correct" subspace predicts being right. Flipped they reach 0.763, but that sign means
they are not measuring membership of a correct region; they should not be reported as a
detector without an account of what they are actually tracking.

**Steer.** `h + α·d̂`, evaluated on the 1,319-question test set:

| α | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|
| change (points) | +0.38 | +0.30 | −0.68 | −3.34 | −18.65 |

The best case is five questions out of 1,319. This outcome was **predicted in advance**
from the band mechanism and is the section's main evidential value: `d̂` is 97% inside
the uniform-lift subspace, so it can raise confidence but not change a decision.

**Project.** Retention using a subspace built only from correct examples:

| k | 4 | 12 | 28 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| fraction of baseline retained | 0.079 | 0.382 | 0.890 | 0.924 | 0.966 | 0.991 |

Indistinguishable from the class-blind band. Principal-angle cosines between the two
bases: k = 4 mean 0.9826 (min 0.9328); k = 28 mean 0.9921 (min 0.9066); k = 64 mean
0.9855 (min 0.7283). **They are nearly the same subspace**, which follows directly from
96% of the variance being within-class.

### The distinction this establishes

> A direction that **predicts** correctness is not a direction that **produces**
> correctness.

`d̂` reports that the model is in a confident regime. Fixing a wrong answer requires
knowing *which* answer is right, and that lives in the band, not in a class-mean
difference. Predictive and causal structure come apart here, and land in nearly
orthogonal parts of the space.

### What this does not settle

- A single global class-mean direction is the bluntest possible steering vector. It says
  nothing about a per-example or learned steering map.
- Steering was never tried *inside* the band. Steering in PCs 0–3 is provably wasted, so
  the informative experiment has not yet been run.
- All figures are first-token accuracy on cached states. None has been confirmed by
  decoding.
- Band boundaries (4, 32) still come from §37's test-set curves.

§42 addresses all four.

## 42. Preregistered three-track correctness experiment

Turns §41 into a design that can be run and read without an asterisk.

### Split discipline

The 2,048-question calibration pool is partitioned **fit (1,024) / select (1,024)**, and
GSM8K test (1,319) is read once per arm. Every direction, probe and steering vector is
estimated on fit; every hyperparameter — ridge strength, Fisher shrinkage, steering step
α, rank — is chosen on select. **Nothing is chosen on test.** This also removes the
standing §37/§40 caveat for every quantity this experiment reports.

### Preregistered gates

| track | primary arm | passes if |
|---|---|---|
| detect | `fisher_plus_margin` | ΔAUC over margin-only ≥ 0.01 with a positive paired-bootstrap lower bound |
| steer | `margin_band` | gain ≥ 1.0 point, positive lower bound, **and** above the best matched random direction drawn in the same band |
| project | `correct_only` at rank 28 | advantage over class-blind ≥ 1.0 point with a positive lower bound |

Each gate is framed against the thing that would otherwise explain the result — the
model's own margin, a random direction in the same band, and the class-blind subspace —
rather than against chance.

**The steer gate is expected to fail.** Stating that in advance is what makes either
outcome informative: a failure sharpens §40's claim from "the band is where the answer
is read" to "the band is a location, not a handle a constant offset can push", and a
pass would be the project's first accuracy-improving intervention.

### What is new relative to §41

- **`margin_band`**: the average margin-widening direction `E[w_gold − w_runner-up]`
  confined to PCs 4–31. This is the steering vector §41 never tried — the only one with
  somewhere to act. Its exact-gradient property is that with the runner-up fixed the
  margin is linear in the state, so this is the derivative, not an approximation.
- **Fisher direction**: `C_within⁻¹ d`, which discounts high-variance nuisance
  directions and so is not forced into PCs 0–3 the way the raw mean difference is. At
  768 dimensions against 1,024 examples the within-class covariance is near singular, so
  the shrinkage is *selected on the select split* rather than fixed.
- **Generation tier**: the steer track is confirmed on full GSM8K by real greedy decoding
  and numeric exact match, using `OfficialCODIEndpointSteerIntervention` — an additive
  edit at `ln_f`, deliberately a separate class from the frozen §40 projection hook.
  α is taken from the analytic export and never re-tuned at this tier.

### Implementation notes

- Reuses §37's colon-state cache, so every direction lives in the space §40 measured.
- Steering is a constant translation and the readout is linear, so
  `(h + αv)Wᵀ = hWᵀ + α(Wv)`. That turns ~150 [1319×768]×[768×50257] float64 products
  into two, and is exact. Retention uses the matching low-rank form
  `μWᵀ + ((h−μ)U)(UᵀWᵀ)`. Both are tested against the dense computation for bit-identical
  outcomes. Full-scale synthetic run: 26 s sweep, 14 s analysis.
- `roc_auc` uses midranks. A plain double `argsort` breaks ties by position and scores a
  fully tied probe at whatever the question order dictates rather than 0.5; the
  bootstrap resamples with replacement, so ties are the normal case there.
- Files: `src/mech/endpoint_correctness_geometry.py`,
  `src/eval/official_codi_correctness_tracks_analysis.py`,
  `scripts/run_official_codi_correctness_tracks.py`,
  `scripts/analyze_official_codi_correctness_tracks.py`,
  `scripts/run_official_codi_correctness_steer_generation.py`,
  `configs/official_codi_gpt2.yaml` (`endpoint_correctness_tracks`),
  `notebooks/kaggle_official_codi_correctness_tracks.ipynb`,
  `tests/test_endpoint_correctness_geometry.py`,
  `tests/test_correctness_tracks_integration.py`.

### Scope

Bounded exactly as §40: official CODI GPT-2, state 12, forced answer cue, linear
subspaces, GSM8K, frozen weights. No distillation target and no inference-speed claim.

## 43. Completed three-track correctness run

Kaggle export `codi-that-predicts-the-right-answer`, 376 files. Both tiers ran.
Every number below is from the preregistered §42 gates, applied once.

### The environment reproduces again

Forced-cue baseline exact match **0.43366** against a freshly re-decoded gate of
**0.43594** — drift 0.0023, inside the 0.015 allowance, 100% answer-cue coverage,
float32 pinned. §39's non-reproducing image is resolved: the transformers/peft pins
are what restore it, as §37 concluded.

### Confirmatory geometry

Fitted on the 1,024-question fit split alone, and it reproduces §40 exactly:

| band | variance share |
|---|---:|
| PC 0–3 | 82.18% |
| PC 4–31 | 11.54% |
| PC 32–767 | 6.28% |

The §41 exploratory class geometry replicates with slightly softer effect sizes, as
expected when the fitting pool halves from 2,048 to 1,024:

| quantity | §41 (exploratory) | §43 (preregistered) |
|---|---:|---:|
| between-class variance share | 3.98% | **3.46%** |
| mean difference inside PC 0–3 | 97.13% | **96.52%** |
| random-split null exceedances | 0/200 | **2/200** |
| ‖d‖ vs null median | 11.7× | **7.65×** |

The Fisher direction does exactly what it was introduced to do: **0.93%** of it lies
in PC 0–3 and **30.4%** in the accuracy band, against the raw mean difference's 96.5%
in PC 0–3. Whitening moves the correctness direction out of the lift band.

### Track 1 — detect: PASSED, at the edge

| probe | test AUC |
|---|---:|
| **fisher_plus_margin** (primary) | **0.8859** |
| margin only (baseline to beat) | 0.8736 |
| accuracy_band_plus_margin | 0.8744 |
| full_state_plus_margin | 0.8673 |
| fisher alone | 0.8668 |
| full_state | 0.8641 |
| lift_band | 0.7622 |
| accuracy_band | 0.7461 |
| mean_difference | 0.7086 |

ΔAUC = **+0.0123** (threshold 0.01) with CI **[0.0000325, 0.0249]**.

Both gate conditions hold, so this passes as preregistered — but the lower bound is
3×10⁻⁵. This is the weakest possible pass and should be reported as "the state adds a
small amount over the margin, right at the edge of detectability", never as a clean
positive. Two further cautions: `accuracy_band_plus_margin` adds +0.0008 and
`full_state_plus_margin` is *worse* than margin alone, so the increment is specific to
the Fisher direction rather than a general property of the state; and the preregistration
is the only thing separating that from a lucky pick among nine probes.

`mean_difference` at 0.7086 replicates §41's 0.700 closely.

### Track 2 — steer: FAILED, decisively, as predicted

Analytic (first token): `margin_band` 0.42305 vs baseline 0.42077, gain **+0.227
points**, CI [−0.076, +0.607], McNemar p = 0.969. The best matched **random** direction
inside the same band scored 0.42456 — **better than the principled arm** (margin
−0.152 points).

Exact match, full GSM8K greedy decoding:

| arm | α | exact match | vs baseline |
|---|---:|---:|---:|
| baseline | — | 0.43366 | — |
| **margin_band** | 0.25 | **0.43366** | **0.00** |
| random_band_r00 | 0.5 | 0.43442 | +0.08 (one question) |

The steering vector confined to the accuracy band changed **nothing at all** on the
outcome the project reports. Three of the five steering arms selected α = 0 on the
select split, i.e. the no-op won outright.

This is the informative failure §42 preregistered. The band is **where the answer is
read, not a handle a constant offset can push.** Confining the vector to the sensitive
directions was the strongest form of the idea available, and it did nothing.

### Track 3 — project: FAILED, as predicted

Correct-only rank-28 retention 0.37301 vs class-blind 0.37377 — advantage **−0.076
points**, CI [−1.14, +0.91]. Mean principal-angle cosine **0.9826** (§41: 0.9921). The
two subspaces are the same subspace; building from correct examples only buys nothing,
because 96.5% of the variance is within-class.

### A limitation this run exposed, not previously recorded

The split base rates are **fit 67.5% / select 66.3% / test 42.1% correct**.

Calibration is drawn from GSM8K *train*, which CODI was trained on, so the model is
**25 points more accurate there than on test**. Every direction, probe and steering
vector in this experiment was therefore fitted on a population in which the model
behaves measurably differently from the one it is evaluated on.

This does not invalidate the two nulls — a direction that does nothing on test did
nothing regardless of where it was fitted — but it is a live threat to the detect
result, which is the one that passed and passed narrowly. A detect replication should
fit on held-out *test-like* questions before the +0.0123 is treated as real.

### Standing

- detect: passed at the boundary; not yet a usable error detector, since the margin
  alone gives 0.8736 for free and the increment is 0.0123.
- steer: the project's cleanest preregistered null. Predicted in advance from the
  §40 mechanism, and confirmed on exact match.
- project: replicates §41's expected null.
