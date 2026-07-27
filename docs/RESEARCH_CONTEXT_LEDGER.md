# Research context ledger

Last updated: 2026-07-26

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

> Repair implemented and statically validated. The mandatory Kaggle GPU preflight
> and the scientific rerun remain to be executed.

The earlier failed Kaggle dataset must not be attached as a resume source. Only
outputs carrying the repaired dtype and schema contracts may be resumed.
