# Research context ledger

Last updated: 2026-09-21 (rev 4)

## Purpose

This is the durable source of truth for continuing the CODI–KaVa project when the
conversation becomes too long or is compacted. Read this file before proposing another
experiment. It records the original question, the instructor's criticism, the
TSV-inspired pivot, completed experimental gates, current evidence, and the next
decision.

Entries §60–§82 reconstruct the September 2026 work that ran without ledger
updates: the answer-readout compression track that grew out of §59's "route 2"
(§61–§68) and the KV-cache compression track (§69–§81), with §82 recording the
assessment. §83 is the first experiment run after that assessment, and it closes the
KV track with a mechanistic finding rather than another null. Read §82 and §83 first.

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

**Steer.** A historical constant shift along the class-mean direction, evaluated on
the 1,319-question test set:

| α | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---:|---:|---:|---:|---:|
| change (points) | +0.38 | +0.30 | −0.68 | −3.34 | −18.65 |

The best case is five questions out of 1,319. **Scale correction (2026-08-27):** the
exploratory export did not preserve the steering vector or normalization metadata. The
earlier text labelled this vector `d̂`, but the later preregistered implementation uses
explicitly unit-normalized vectors and does not reproduce the old alpha curve. The
alpha values in this table therefore cannot be compared numerically with §42/§43 and
must be treated only as a qualitative exploratory result. The raw `d` had norm 26.22,
so confusing `d` with `d̂` would change the physical intervention size by that factor.

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

§42 addresses test-set tuning, in-band steering, and exact-match decoding. Its steer
arm remains a **single constant vector**, so it does not test a question-conditioned or
learned correction map.

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
outcome informative: a failure would show that the fitted **constant offset** is not an
accuracy-improving intervention. It cannot establish that the band is unusable by a
question-conditioned correction map.

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

This is the informative failure §42 preregistered. Confining the vector to the
sensitive directions was the strongest form of a **single global translation** tested,
and it did nothing. It does not show that the band cannot be used by a
question-conditioned or learned intervention: averaging
`w_gold - w_runner-up` across questions with different gold answers can cancel the
answer-specific directions that such an intervention would need.

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

- detect: passed its preregistered gate at the boundary, but is provisional rather
  than a confirmed scientific positive because its fitting population is not
  test-like. The margin alone gives 0.8736 and the increment is 0.0123.
- steer: a clean preregistered null for a single constant translation, confirmed on
  exact match. It is not evidence against question-conditioned steering.
- project: replicates §41's expected null.

## 44. Post-run audit and corrections

The completed export and source were re-audited after the apparent contradiction
between §41's alpha curve and the unit-normalized §43 curve.

1. **The old steering scale is not recoverable from the repository.** The §41 record
   contains the outcomes but no saved vector or normalization metadata. Calling it
   `d̂` was unsupported and has been corrected above. The current implementation
   explicitly normalizes every steering arm in
   `build_steering_vectors`, tests unit norm, and interprets alpha in state units.
2. **The population shift is confirmed in source.** The colon-state collector labels
   the calibration source as GSM8K train and the evaluation source as GSM8K test.
   Because CODI trained on GSM8K train, 67.5/66.3% calibration correctness versus
   42.1% test correctness is a central validity threat to the narrow detect result,
   not a minor footnote.
3. **The steer conclusion was too broad.** `margin_band` is the normalized projection
   of the *average* per-question `w_gold - w_runner-up` vector. It tests whether one
   global offset helps all questions. Its exact-match null is valid for that arm, but
   cannot answer whether a per-question or learned map can use PCs 4-31.
4. **Probe under-convergence is possible but not established by the reported AUCs.**
   The completed probe used 700 Adam steps and did not export an objective gap or
   gradient norm. However, a 768-feature model scoring worse than a one-feature model
   on held-out AUC is not proof of failed optimization: nested models can generalize
   differently and AUC is not the ridge-logistic training objective. A detector rerun
   should use a convergence-checked solver and record optimization diagnostics, but
   the existing numbers alone do not license declaring the primary two-feature Fisher
   probe underfit.

Corrected standing: the project has a valid null for **constant global steering**, a
valid null for correct-only rank-28 projection, and a preregistered but scientifically
provisional detect pass. A defensible detect replication must use a held-out test-like
fitting population and a convergence-audited probe before the +0.0123 increment is
treated as real.

## 45. Test-like, convergence-audited detect replication

The corrective follow-up freezes the two repairs identified in §44 into a separate
contract. It partitions the 1,319 cached GSM8K test states into fit/select/test counts
440/440/439 with seed 20260827. Every direction and logistic weight is fitted on fit;
Fisher shrinkage and ridge are chosen on select; the remaining 439 questions are read
once. No GSM8K-train calibration state is eligible.

The primary gate is unchanged: `fisher_plus_margin` must improve over `margin` by at
least 0.01 AUC with a positive paired-bootstrap lower bound. A third requirement now
blocks the gate unless both selected fits have convergence certificates. The new
full-batch L-BFGS solver exports final objective, gradient norms, iteration counts and a
strong-convexity upper bound on the objective gap; every ridge candidate must converge.

This is a corrective replication on a previously inspected dataset, not a pristine
preregistration, and its 439-example test interval will be wider. The implementation,
tests, run commands and interpretation boundary are in
[`OFFICIAL_CODI_CORRECTNESS_DETECT_REPLICATION.md`](OFFICIAL_CODI_CORRECTNESS_DETECT_REPLICATION.md).

Status: complete — see §49. The gate did not pass.

## 46. Correct-versus-wrong contrastive covariance experiment

The new question is whether the 768-dimensional state-12 covariance can be separated
into directions specific to correct answers and directions specific to wrong answers,
and whether projecting a new answer-cue state through those directions can preserve or
repair the answer. This is not another name for §43's correct-only PCA arm: that arm
maximised variance within correct examples but did not penalise variance shared with
wrong examples.

The new contract solves the regularised generalized eigenproblem
`C_correct v = lambda C_wrong v` on 440 cached GSM8K-test states. The 28 largest-ratio
directions are the correct-specific candidate and the 28 smallest are the
wrong-specific candidate. Shrinkage is chosen on a disjoint 440 by held-out
projection-energy specificity; a final 439 are read once. The split and seed are the
same frozen test-like partition as §45, preventing a proliferation of differently
sampled final cohorts.

The primary intervention retains the correct-specific projection around the fitted
correct mean. It must beat correct-only PCA, the established class-blind accuracy band
(PCs 4–31), and the best of eight rank- and correct-class-energy-matched random controls by at least one accuracy point,
with positive paired bootstrap lower bounds. A separate secondary gate removes the
wrong-specific projection around the global fit mean and requires a one-point gain over
baseline plus superiority to its matched-random controls. State-12 first-token
accuracy, gold NLL, and gold margin are primary analytic outcomes. A GPU tier repeats
baseline and four essential arms with paired greedy exact-match generation on the same
439 indices.

Status: complete — see §47. The frozen design and interpretation boundary are in
[`OFFICIAL_CODI_CORRECTNESS_CONTRASTIVE_COVARIANCE.md`](OFFICIAL_CODI_CORRECTNESS_CONTRASTIVE_COVARIANCE.md).

## 47. Completed contrastive covariance run: `not_confirmed`

Kaggle export `jonraza15/can-codis-correct-and-wrong-be-separated` (2026-08-27),
pinned at commit `fa49050`. Verified locally: all 54 SHA-256 checksums intact, and
both the analytic gate report and the paired generation report recompute
**bit-identically** from the raw sweep export using the repository analyzers. The
generation manifests carry the same frozen partition SHA as the fitted geometry, so
every arm read the same 439 final questions once.

Two validity properties this run has that §43 lacked:

- **Test-like population.** All fitting used cached GSM8K-*test* states under the
  frozen §45 partition; realized base rates are fit 44.1% / select 42.0% / test 40.1%
  correct. The §44 population-shift threat does not apply here.
- **Shrinkage chosen off-test.** λ ridge 0.2 was selected on the select split by
  held-out log correct/wrong projection-energy ratio (correct-specificity 0.952), so
  the directions demonstrably *are* class-specific out of sample.

### Result

The primary gate failed decisively. Retaining the 28 largest-ratio directions of
`C_correct v = λ C_wrong v` around the correct mean scored **0.1207** first-token
accuracy against baseline 0.4009 — 20.05 points *below* both correct-only PCA and the
class-blind accuracy band (identical 0.3212; CI [−24.15, −15.95] for both contrasts).
The only sub-gate that passed is superiority over the eight energy-matched random
retains (best 0.0433; +7.74 pts, CI [+5.01, +10.71]).

The secondary gate also failed: removing the 28 wrong-specific directions produced
**exactly baseline accuracy** (0.4009, gain 0.00, CI [−1.59, +1.59]), and the best
matched-random removal tied it.

Correction (2026-08-28), recomputed from the artifact's per-question outcomes: that
null is a *cancellation*, not inertness. The removal moved **14 of 439 predictions —
7 gained, 7 lost** — and shifted the arm's mean gold margin (−0.218 versus −0.116)
and gold NLL (2.981 versus 3.052). The edit therefore changes the state and even the
answer on a few questions; what it lacks is any direction. Chat and report wording
that described it as changing "not one prediction" was stronger than the data
supports, and the accurate statement is that removal is directionless at the same
scale as matched-random removal.

The GPU exact-match tier confirms the ordering on the same 439 questions: baseline
0.4237, correct-retain 0.1298, wrong-remove 0.4123 (−1.14 pts, CI [−2.96, +0.68]),
correct-only PCA 0.3394, accuracy band 0.3622. Analytic and decoded tiers agree within
a few points on every arm, extending the §40 record of the analytic evaluator's
fidelity.

### What this settles

The §16/§46 question — can the state-12 covariance be split into correct-producing and
wrong-producing directions usable by a projection — is answered **no** for fixed
linear interventions. The contrastive directions are real (class-specific out of
sample, far above matched random) but they are variance that *accompanies* being
right, not variance that *makes* the answer right; and there is no wrong-specific
channel whose deletion helps. This replicates §41/§43's orthogonality conclusion on a
test-like population and closes the third of the three uses (detect / steer / project)
against the strongest class-conditioned linear estimator tried so far.

Standing additions to §17: do not design further *global, question-independent* linear
projections or translations at state 12 from class-conditioned second-order statistics;
the family is exhausted. The open direction remains a question-conditioned correction
map (§43, §44.3), which no completed experiment has yet tested.

## 48. Same-question paired, conditioned correction

The contrastive covariance design in §46 still compares correct states from some
questions with wrong states from other questions. Its generalized eigenvectors can
therefore encode question content, answer value, difficulty, or confidence rather than
a within-question transition from wrong to correct. Its retention arm also replaces
the 740 coordinates outside the selected rank-28 subspace, making destructive
compression inseparable from the intended projection.

The corrective experiment constructs actual within-question state pairs. Sampling the
answer token is insufficient because the state-12 vector at `The answer is:` exists
before that token is sampled, so all samples would share exactly the same vector.
Instead, eight frozen relative-RMS perturbations enter at state 11 and propagate through
the last block to state 12. A question is eligible only when its variants yield both a
correct and wrong greedy first answer token. It contributes one equal-weight target:
the mean correct state minus the mean wrong state, projected into fit-derived PCs 4–31.

A multi-output ridge map predicts that paired displacement from the incoming state's
28 band coordinates plus its top-two output margin. Deployment is additive, preserving
all off-band coordinates. Ridge strength, edit size, and a low-confidence margin gate
are chosen on a disjoint selection split. The final 439 questions are read once. A
global mean displacement and a target-shuffled map must both be beaten, in addition to
a one-point accuracy gain over no intervention with a positive paired-bootstrap lower
bound. An optional four-arm generation tier checks full greedy exact match.

The perturbations are controlled answer-selection counterfactuals, not evidence of
natural alternative reasoning traces. This also reuses a previously inspected GSM8K
test population, so a positive is exploratory rather than a pristine confirmation.

Status: complete — see §50. The gate did not pass. The contract is in
[`OFFICIAL_CODI_PAIRED_CORRECTION.md`](OFFICIAL_CODI_PAIRED_CORRECTION.md).

## 49. Completed detect replication: `test_like_detect_not_supported`

Kaggle export `jonraza15/replication-of-codis-correctness-detector` (2026-08-27).
Verified locally: all SHA-256 checksums intact and the gate report recomputes
**bit-identically** from the raw sweep export with
`scripts/analyze_official_codi_correctness_detect_replication.py`. The partition SHA
equals the frozen §45 split (`c8316e46…`), the same one used by §47, so fit/select/
test are 440/440/439 GSM8K-test questions with correct shares 44.1% / 42.0% / 40.1%.
Both §44 repairs held: no GSM8K-train state entered any fit, and every probe exports a
convergence certificate (L-BFGS strong-Wolfe, gradient norms ≤ 7e-8, objective-gap
bounds ≤ 8e-15, all ridge candidates converged).

### Result: the gate did not pass

| probe | test AUC |
|---|---:|
| **fisher_plus_margin** (primary) | **0.8942** |
| margin (baseline to beat) | 0.8795 |
| full_state_plus_margin | 0.8753 |
| full_state | 0.8709 |
| fisher alone | 0.8664 |
| mean_difference | 0.7216 |

ΔAUC = **+0.0147**, above the frozen 0.01 magnitude threshold and consistent with
§43's +0.0123 — but the paired-bootstrap CI is **[−0.0080, +0.0389]**, so the positive
lower-bound requirement fails. Under the frozen §45 decision rule the §43 detect pass
is **retired as not established**.

Two things this settles beyond the headline:

- **§44.4 is closed.** The 768-feature probe again scores below the two-feature probe
  (0.8709 vs 0.8942) with a machine-precision convergence certificate, so that gap is
  a genuine generalization property, not failed optimization.
- **The ordering replicated.** Fisher-direction structure (fisher 0.8664,
  mean_difference 0.7216, the primary above both full-state probes) reproduced on a
  disjoint fitting population, which is why the retirement is phrased as
  "not established at this sample size" rather than "shown to be absent": the point
  estimate replicated in sign and size, but 439 test examples give an interval about
  ±0.023 wide, and an increment of ~0.013 cannot clear it.

### Corrected standing of the three-track experiment

detect: **not supported** on a test-like population (this section). steer: null
(§43). project: null (§43, replicated by §47). Combined with §47, no fixed linear
state-12 quantity — direction, subspace, or probe increment — has survived its
preregistered gate beyond the model's own margin. The only open branch remains the
§48 same-question paired conditioned correction, whose GPU run is pending. No larger
detect replication is planned; establishing a ~0.013 AUC increment would need a
substantially larger test-like population than GSM8K test provides under this split.

## 50. Completed paired conditioned correction: `not_confirmed`

Kaggle export `jonraza15/same-question-correct-to-wrong-correction-for-codi`
(2026-08-27), pinned at commit `e67bb20`. Verified locally: all SHA-256 checksums
intact, and both the analytic gate report and the paired exact-match generation report
recompute **bit-identically** from the raw exports with the repository analyzers. The
frozen partition (`c8316e46…`) is the same one used by §47 and §49; the in-run
reproduction gate re-passed; no noisy final-test state was ever saved.

### The counterfactual collection worked, and measured two things

First-token correct share on the 880 fit/select questions stayed between 0.431 and
0.453 at **every** perturbation level — even 0.60 relative RMS at state 11 barely
moves greedy answer selection. Only 65 fit and 60 select questions produced both a
correct and a wrong variant. The §48 composition diagnostic shows both transition
types present (fit 25 baseline-correct / 40 baseline-wrong; select 30 / 30), so the
result is not an artifact of a denoiser-only training set.

The held-out target cosine was ≈ 0.016 at every ridge (falling to −0.004 at ridge
100), and MSE selection chose maximum shrinkage. **The paired wrong-to-correct delta
is essentially unpredictable from the incoming state's band coordinates and margin.**

### Result

Selection honestly chose α = 0.25 with a 10% margin gate for the conditioned arm and
α = 0 — the no-op — for the global-mean arm. On the 439 final questions the
conditioned map edited 12.8% of states (analytic) and 48 of 439 decoded states (RMS
edit norm 0.207) and changed **zero predictions on both tiers**: analytic 0.4009 =
baseline, exact match 0.4237 = baseline, gold NLL and margin moved at the fourth
decimal. The shuffled-target control moved one question. Primary and specificity
gates failed; every bootstrap interval crosses zero.

### What this closes

With §43 (global constant translation), §47 (class-conditioned projection), and now a
question-conditioned additive map, **every conditioning level of fixed linear
state-12 editing tested by this project has returned a null.** The two measured
causes — answer selection is robust to isotropic late-layer noise, and the flip
direction is unpredictable from state-12 observables — are each sufficient alone.

Bounded as frozen: this does not rule out corrections conditioned on richer inputs
(question text, earlier-layer states), non-linear maps, or counterfactuals from
genuinely different reasoning traces rather than local noise. It rules out the
strongest form the completed evidence had motivated. Standing addition to §17: do not
propose further fixed linear edits at the answer-cue endpoint, at any conditioning
level, without a qualitatively new source of counterfactual pairs or conditioning
information.

The endpoint-editing branch of the project is closed. The completed record now
supports write-up: one confirmed positive (§40 accuracy band), replicated geometry
(§43, §47), and preregistered nulls covering detection (§49), steering (§43),
projection (§43, §47), and conditioned correction (§50).

## 51. Latent-trajectory detection gate

§50 closed the endpoint-editing branch and named the two requirements any new
location must satisfy: edits must propagate, and correctness must still be separable
from question content there. The six latent thought states are the only location in
CODI meeting both — they enter the KV cache (the Phase-3 ablations showed they are
causally load-bearing), and they are computed while reasoning is in progress. No
completed experiment has tested a correct/wrong or answer-identity split there: every
prior latent-position experiment used variance-ranked or teacher-residual selectors,
the ranking §40 later proved wrong.

Per §19, a cheap read-only gate precedes any intervention design. One observational
GPU pass captures all 78 trajectory cells (6 latent positions × the standard 13
states) from the released generation path, with a zero-noise endpoint capture that
must match the validated colon-state cache within 0.001 relative deviation before the
export saves. On the frozen 440/440/439 partition shared with §47/§49/§50, two frozen
gates are read once:

- **correctness**: best select-chosen cell + margin must beat the margin alone by
  ≥ 0.02 AUC (not 0.01 — §49 measured that ~0.013 is unresolvable on 439 questions)
  with a positive bootstrap lower bound and convergence certificates on both fits.
- **answer identity**: on final-test *wrong* questions, an exact one-hot ridge probe
  on the chosen cell must recover the gold first answer token ≥ 5 points better than
  the same probe class reading the endpoint state and better than the majority class,
  with positive lower bounds against both.

The frozen decision rule: only a passed answer-identity gate justifies proposing a
latent-state editing experiment; a correctness-only pass is a detection finding; a
double failure closes the latent-trajectory question for linear probes and the
project proceeds to write-up with no further mechanistic runs.

Status: complete — see §52. Both gates failed. The contract is in
[`OFFICIAL_CODI_LATENT_TRAJECTORY_DETECT.md`](OFFICIAL_CODI_LATENT_TRAJECTORY_DETECT.md).

### §51 addendum: the pre-pin cache states are not reproducible, by measurement

The first collection attempt gated on exact vector agreement between the live
forced-cue state 12 and the §37 colon-state cache. It failed at **maximum relative
deviation 2.35 — identically at batch sizes 16 and 32**, so the mismatch is
deterministic and chunking-independent. Capture semantics were verified identical to
the original collector (same `ln_f` hook, same last-position slice, same activation
window). This measures directly what §40's remaining-limits paragraph recorded as a
caveat: the colon-state cache was collected on the §39 pre-pin image, and the pinned
environment reproduces the checkpoint's aggregate accuracy but not that cache's exact
state vectors.

Consequences applied:

- The collection now validates the live pass on its own terms — analytic parity
  (`argmax(W·h12)` reproduces the decoded token, ≥ 0.99) and an accuracy-reproduction
  gate against the cache (allowance 0.02) — and reports the cache-state deviation as
  a diagnostic. Labels, margins, and the endpoint-probe baseline all come from the
  live pass, so the §51 experiment is fully self-consistent on the pinned
  environment and, for itself, removes §40's last-dependency caveat.
- The collector reproduces the cache's recorded collection batch size (16), since
  GPT-2's absolute position ids depend on each chunk's left-padding width.
- **Audit note for §50:** its pairing pool mixed the cache-derived baseline variant
  with live-collected noise variants. Given the now-measured live-versus-cache state
  mismatch, some of its 125 "paired" questions may be environment-flip artifacts
  rather than noise flips, and the uniform ~2-point live-versus-cache correct-share
  offset in its collection table is explained. Neither §50 null changes: the fitted
  map altered zero final-test predictions regardless of how its training pairs arose.

## 52. Completed latent-trajectory gate: `latent_trajectory_not_supported`

Kaggle export `jonraza15/does-codis-trajectory-its-endpoint-forgets` (2026-08-27),
pinned at commit `40ddfc0`. Verified locally: all checksums intact and the gate
report recomputes **bit-identically** with the repository analyzer. Collection was
fully live-consistent on the pinned environment: analytic parity agreement 1.0 on all
1,319 questions, live first-token accuracy 0.4466 (+0.0106 over the reproduction's
exact match, inside the frozen band), the frozen partition (`c8316e46…`), and the
cache's recorded chunking. Five of 234 correctness fits stalled at ridge 100 and were
recorded as ineligible candidates; every used fit carries a convergence certificate.

### Both gates failed

**Correctness.** The best of 78 trajectory cells (position 4, state 11; select AUC
0.8844) scored test AUC **0.8571** against the margin baseline's **0.8819** —
ΔAUC **−0.025**, CI [−0.054, +0.005]. Trajectory states carry real correctness
signal, but adding 768 trajectory coordinates makes the probe generalize *worse*
than the margin alone. This extends §49's conclusion upstream: at no point in the
latent computation does a linear readout of the state predict correctness beyond
the model's own confidence.

**Answer identity — the decisive gate — failed inverted.** On the 249 wrong-answer
final-test questions, the select-chosen trajectory cell recovered the gold first
token on **1.6%**, against the same probe class reading the endpoint state at
**6.4%** and the majority class at **3.6%** (gain −4.8 points; CI against the
endpoint [−8.0, −1.6]). The hypothesis was that the trajectory knows answers the
endpoint discards; the measurement says the opposite — answer identity is *more*
linearly recoverable at the endpoint than anywhere in the trajectory, and even
there it barely beats the majority class. **When CODI answers incorrectly, the
correct answer is essentially absent from its latent states in any linearly
readable form. There is nothing there for an editor to restore.**

### Measured environment side-result

The collection's cache diagnostic quantified the §51-addendum finding: only 84 of
1,319 pre-pin cache states fall within 0.001 relative deviation of the live pinned
pass (median deviation 0.218, max 2.35), with 71.1% first-token agreement — while
aggregate accuracy differs by only ~2.6 points. Environment changes below the
accuracy-visibility threshold produce O(1) state-level differences through the
recurrent latent loop. Any future experiment mixing cached states with live passes
must treat them as different populations.

### The project's mechanistic program is closed

Per the frozen §51 decision rule, no latent-state editing experiment is justified
and no further mechanistic run is planned. The completed record now reads:

- **One confirmed positive**: PCs 4–31 of the endpoint state carry CODI's answer
  (§40), geometry replicated across four runs.
- **Preregistered nulls at every level asked of that finding**: detection beyond
  the margin (§49, extended upstream by §52), constant steering (§43),
  class-conditioned projection (§47), question-conditioned correction (§50), and
  latent-trajectory recoverability (§52).
- The unified reading: CODI's answer is *determined* by a small linear subspace at
  the endpoint, but neither its correctness nor its correction is linearly
  readable or writable anywhere the project could reach — predictive and causal
  structure dissociate at every conditioning level and every depth tested.

Remaining work is write-up and the original CODI-versus-KaVa scope (seeds and
controls). Standing addition to §17: do not propose further linear-probe or
linear-edit experiments on the frozen checkpoint at any state without a
qualitatively new information source.

## 53. Exploratory: the trajectory is a workspace of intermediates, and wrong answers have wrong workspaces

Everything in this section is **exploratory** — computed locally on the completed
§52 export (`jonraza15/does-codis-trajectory-its-endpoint-forgets`) plus the pinned
GSM8K test solutions (`3101c7d5…`), with no preregistration, and recorded so a frozen
confirmation can be read as confirmation rather than discovery. It was prompted by a
§52 post-mortem which found the answer-identity tier had asked the trajectory for the
wrong content: the CODI paper's own interpretability section decodes latent thoughts
into *intermediate calculation results*, never final answers, and §52's probe was a
data-starved 140-class classifier rather than the model's own vocabulary projection.

Reading each thought's state 12 through the frozen readout (top-5 logit lens, exact
string match against the `<<…=v>>` intermediate values of the gold solutions):

| quantity | value |
|---|---:|
| gold intermediates recovered in thoughts (all 1,319) | **32.7%** |
| shuffled-question null | 4.6% |
| recovery on questions answered correctly | **41.4%** |
| recovery on questions answered wrongly | **25.5%** |
| gold *final* answer in thoughts (correct / wrong) | 9.3% / 10.0% |
| questions with a hit at thought 0 / 1 / 2 / 3 / 4 / 5 | 3 / **837** / 2 / **749** / 5 / **754** |
| correctness AUC of the recovery fraction alone (fit+select) | 0.659 |
| margin + recovery + numeric-count, fit→select AUC | 0.897 vs margin's 0.893 |

Three exploratory conclusions:

1. **The CODI paper's decoding claim replicates quantitatively** on this checkpoint:
   the six latent thoughts carry gold intermediate values at seven times the matched
   null, in a strict alternating structure — values at odd thoughts, operator/syntax
   tokens at even thoughts. With M = 6 the workspace holds roughly three values.
2. **The correct/wrong separation the project has hunted since §41 exists at the
   trajectory, as content**: correct runs' workspaces contain 16 points more of the
   gold intermediates than wrong runs'. This is not a confidence direction — it is
   whether the model computed the right quantities.
3. **The §43–§52 null lattice now has a mechanistic explanation.** The trajectory
   holds intermediates, not the final answer (9–10% versus 33%), so no probe or edit
   aimed at final-answer identity could succeed; and a wrong run's workspace contains
   *wrong values*, actual computation errors rather than a removable overlay, so no
   fixed linear correction could repair it. The nulls were measuring the structure of
   the mechanism, not its absence.

The margin-relative detect increment remains small (+0.004 on select), so this does
not promise a detect-gate pass; its value is content, structure, localization, and
the explanation of the nulls. A frozen confirmation should preregister: recovery
versus matched null, the odd-slot structure, correct-versus-wrong recovery with a
paired interval, thought-to-step alignment, and — on wrong questions — whether the
model's *own* wrong answer is traceable to its decoded wrong intermediates. All
tiers are CPU-only on the existing export. The final 439 rows entered today's
aggregates, so the confirmation is corrective-lineage, §42-style, not pristine.

## 54. Preregistered latent-workspace confirmation

Freezes §53 into four gates before one read of the frozen 439-question final split.
The instrument stays the model's own vocabulary projection (top-5 logit lens at each
thought's state 12, exact match against the pinned solutions' `<<…=v>>` values), so
no probe fitting can overfit and multi-token values undercount conservatively.
Thresholds were frozen from the fit/select observations: content ≥ 10 points over a
seeded derangement null (observed ≈ 28), even-slot hit share ≤ 10% with every odd
slot ≥ 0.30 (observed 0.4% and ≈ 0.6), correct-versus-wrong recovery gap ≥ 5 points
(observed ≈ 16), and own-versus-gold answer-token tracing on wrong questions ≥ 4
points (observed 19.5% vs 11.0%). Thought-to-step alignment is preregistered as
descriptive only: fit/select shows step order is NOT preserved. Because §53's
aggregates touched the test rows, this is a §42-style corrective-lineage
confirmation, not a pristine preregistration. All tiers are CPU-only on the
completed §52 export. The contract is
[`OFFICIAL_CODI_LATENT_WORKSPACE.md`](OFFICIAL_CODI_LATENT_WORKSPACE.md).

## 55. Confirmed: the latent thoughts are an arithmetic workspace

The §54 gates were read once against the frozen 439-question final split
(preregistration commit `5c1e326`; CPU-deterministic run on the checksummed §52
export; report recomputed bit-identically; summary, artifact, and report pinned
in-repo under `artifacts/latent_workspace/`). **All four gates passed**, every one
with a margin over its frozen threshold:

| gate | threshold | observed |
|---|---|---|
| content | ≥ +10 pts | 31.0% vs 5.6% null → **+25.3** [+22.5, +28.3] |
| structure | even ≤ 10%, odd ≥ 0.30 | even **0.14%**; odd 0.62 / 0.54 / 0.54 |
| correct/wrong gap | ≥ +5 pts | 41.7% vs 22.6% → **+19.2** [+14.3, +24.1] |
| faithful readout | ≥ +4 pts | own 19.3% vs gold 10.4% → **+8.8** [+2.8, +14.9] |

Hits per thought: `[0, 263, 0, 228, 1, 227]`. The descriptive alignment table
confirms the workspace is an *unordered* value store: intermediate k does not
preferentially occupy value slot k.

### The confirmed statement

> On official CODI GPT-2 over GSM8K, the six latent thoughts are a measurable
> arithmetic workspace: the odd thoughts store the solution's intermediate values
> (31% recovered verbatim by top-5 decoding against a 5.6% matched null — a
> conservative lower bound, since multi-token values cannot match), wrongly
> answered questions carry 19 points less of the correct content, and the model's
> wrong answers appear among its own workspace numbers significantly more often
> than the gold answers do.

This is the project's **second confirmed positive** (after the §40 accuracy band),
and it closes the arc as an explanation rather than a mystery: the endpoint's
28-dimensional band *determines* the answer (§40) by faithfully reading out a
workspace whose contents were computed earlier; correctness is not an endpoint
overlay but the property of having computed the right intermediate values (§55);
therefore nothing at the endpoint could detect it beyond the margin (§49, §52) and
no linear edit at any conditioning level could repair it (§43, §47, §50, §52).
Every null in the lattice is now a corollary of the confirmed mechanism.

### Scope and limits

Bounded to the frozen checkpoint, GSM8K, top-5 exact-match single-token decoding at
each thought's state 12, and the live pinned-environment pass. No intervention was
run and no claim is made that the workspace can be edited; the §52 gate already
established that its contents are not linearly *writable*, and §55 explains why —
the workspace holds computed values, not a correctable code. The natural future
work (not planned this semester) is trajectory-level supervision: KaVa-style
distillation targets scored against teacher intermediates, which reconnects the
mechanistic finding to the project's original CODI-versus-KaVa question.

## 56. Preregistered latent value injection

§55 is correlational: it reads values out of the workspace. The causal tier writes
values in. During the latent loop, at the measured value slots (thoughts 1/3/5,
state 11), each arm adds a value token's unit-normalized readout direction scaled
by beta times the state RMS — `gold` writes the question's gold intermediates,
`offset` writes gold-plus-one, and `random` writes seeded numeric tokens with the
identical slot mask and scale, so the arms differ only in which value they write.
Unlike every §43–§52 endpoint edit, this enters inside the latent loop and
propagates through the KV cache; the gold-value information source is exactly the
"qualitatively new information" the §50 standing rule requires.

Frozen gates on the standard partition, one test read, thresholds set before any
run (pristine preregistration): corruption — on baseline-correct rows,
damage(offset) minus damage(random) ≥ 5 points with a positive paired lower bound
(values causally used); repair — on baseline-wrong rows, recovery(gold) minus
recovery(random) ≥ 3 points (values repairable). Beta is chosen on select by the
repair criterion only and shared by all arms. Stated expectation: corruption
likely passes, repair is the open question; "used but not repairable" would itself
be a sharp mechanism result. Contract:
[`OFFICIAL_CODI_VALUE_INJECTION.md`](OFFICIAL_CODI_VALUE_INJECTION.md).

## 57. Findings-derived efficiency measurement protocol

A companion protocol-frozen measurement study (no hypothesis gates) quantifies the
two efficiency routes the findings expose: the latent-budget sweep
(`latent_iterations ∈ {3,4,5,6}`, full-GSM8K numeric exact match and wall clock —
the model was trained at M = 6, so truncation cost is the open measurement) and
the rank-{28,32,64} answer-readout microbenchmark (the accuracy side is already
measured: §40's rank-32 retention kept 94.4% of exact match). Both run in the §56
notebook. Estimated combined latency saving if the sweep cooperates: ~20% at a
~2-point accuracy cost; the measurement replaces the estimate.

Status of §56–§57: implementation, synthetic tests, and the run-all Kaggle
notebook are complete; no run has been executed and no empirical claim is made.

## 58. Completed value injection: `value_injection_not_supported`

Kaggle export `jonraza15/writing-values-two-efficiency-routes` (2026-08-28), pinned
at commit `bb8631d`. Verified: all checksums intact, the gate report recomputes
bit-identically, the frozen partition and checkpoint hashes match. Beta selection
behaved honestly (select curve peaked at a mere +1.6 points at beta 2.0); every arm
applied 1,128 identically scaled edits at twice the state RMS.

Both frozen gates failed, and the corruption gate failed **at exactly zero**: on
the 183 baseline-correct test questions, writing plausible wrong values
(gold-intermediate-plus-one) into the value slots left 99.45% of answers intact —
identical to matched random tokens (difference 0.00, CI [−1.6, +1.6]). Repair:
gold recovered 3 of 248 wrong answers versus random's 1 (+0.8 points, CI
[0.0, +2.0]).

### What this establishes

The §55 workspace values are **readable but not writable through their own readout
directions**. This does not contradict the latent states' causal necessity (the
Phase-3 zero/shuffle ablations cost many points); it localizes it: the
vocabulary-aligned component that carries the decodable values is not the
component the computation runs on. The values behave as a readable shadow of a
redundant, distributed code. Together, §55 + §58 are a sharp instance of the
field's newest methodological warning — decodable patterns need not be causally
used — established here with preregistered gates on both sides: four passed
decoding gates, two failed injection gates, same checkpoint, same values, same
slots.

Bounded as frozen: the null covers additive readout-direction injection at state
11 of the value slots. Replacement edits, multi-token values, earlier layers, or
KV-entry edits remain untested and unplanned this semester. Standing addition to
§17: do not claim the workspace is editable, and do not build supervision
proposals that presuppose the readout-aligned component is load-bearing.

## 59. Completed efficiency measurements

Same export, protocol-frozen measurement study (no hypothesis gates), Tesla T4,
float32, batch 32, full 1,319-question GSM8K per condition:

| latent budget M | exact match | vs M=6 |
|---:|---:|---:|
| 6 (trained) | 0.4337 | — |
| **5** | **0.4359** | **+0.23 pts (free)** |
| 4 | 0.3836 | −5.00 pts |
| 3 | 0.3768 | −5.69 pts |

**One latent thought can be dropped at inference for free** — M=5 scored slightly
above the trained M=6 — and the cliff is at M=4. Batched T4 wall clock was flat
across M (fixed costs dominate at batch 32), so the latency saving of the dropped
pass (~1 of ~16 sequential steps in a latency-bound deployment) is architectural
rather than demonstrated on this hardware configuration.

The rank-k answer-readout microbenchmark measured the full lm_head projection at
1,327 µs against **117 µs at rank 32 (11.3×)**, 120 µs at rank 28 (11.0×), and
154 µs at rank 64 (8.6×), with the accuracy side already measured by §40's
retention arms (rank 32 keeps 94.4% of exact match). The two findings-derived
efficiency corollaries are therefore: a free M=5 inference budget, and an
11×-faster answer readout at a 2.4-point accuracy cost.

## 60. September 2026: two tracks ran without ledger entries

Between 2026-08-29 and 2026-09-21 the project ran roughly twenty Kaggle experiments
in two tracks and recorded none of them here. Entries §61–§81 were reconstructed on
2026-09-21 from the protocol docs, the notebook preambles (each restates its
predecessor's outcome), the completed global-head report
(`scripts/build_completed_global_low_rank_head_report.py`, which hard-codes the
immutable run values), and the run transcripts of the last three experiments. Where a
Kaggle `summary.json` was never transcribed into the repository the outcome is marked
**not recorded**; those cells must be filled from the Kaggle datasets before any of
them is cited.

- **Track 1 (§61–§68)** pursued §59's second efficiency route: replace the full
  50,257-way vocabulary head with a low-rank head at every visible answer position.
- **Track 2 (§69–§81)** pursued KV-cache compression: find answer-causal K/V
  directions, protect them inside xKV-style cross-layer factorization, then spend a
  fixed cache budget unevenly across layers, components, and requests.

Neither track advanced the CODI-versus-KaVa question (§1). §82 assesses why both
stalled and states the decision point.

## 61. Eigenspace-distilled readout: eigen-init works, the 98% gate fails

`kaggle_codi_eigenspace_distilled_readout.ipynb` (2026-09-01), full 1,319-question
GSM8K test, frozen backbone, head-only training on question-disjoint train states.
Fixed forced-cue baseline 43.366% (572/1,319).

| head | exact match | of dense | isolated head | end-to-end |
|---|---:|---:|---:|---:|
| full | 43.366% | 100% | 637 µs | 1.000× |
| fixed eigen rank 32 (§40 basis reused at every position) | 5.080% | 11.7% | — | — |
| learned random-init rank 32 | 22.517% | 51.9% | — | — |
| learned eigen-init rank 32 | 40.334% | 93.0% | 128 µs | 1.136× |
| learned eigen-init rank 64 | 42.002% | 96.85% | 139 µs | 1.122× |

The preregistered 98% retention gate failed at both ranks. Two facts survive: the
§40 answer subspace is a strong *initialization* (eigen-init beats random-init by 18
points at equal rank and equal training), and the fixed colon basis cannot be reused
at later answer positions (5.08%). A later notebook preamble quotes the end-to-end
gain as "about 6%"; the report records 1.122×. The two measurements were not
reconciled.

## 62. Qwen generalization of the answer eigenspace: run, outcome not recorded

`kaggle_eigenspace_readout_generalization_qwen.ipynb` (v1 2026-09-01, corrected v2
2026-09-02) tested whether the readout-aware eigenspace selection transfers to
`Qwen/Qwen2.5-Math-1.5B-Instruct`. v1 measured prompt-endpoint states and was
declared invalid because CODI's measured state is post-reasoning, pre-answer; v2
measured the state before Qwen's final answer token, with rank 192 endpoint locality
and a distilled full-generation head as separate gates. A Kaggle dataset exists
(`does-the-answer-eigenspace-generalize-beyond-codi`). **Outcome not recorded** in
the repository.

## 63. Position-conditioned readout: v1 failed safely, v2 outcome not recorded

`OFFICIAL_CODI_POSITION_CONDITIONED_READOUT.md` (2026-09-04). Version 1 stopped at
state collection: the 1,024-question fit split held 1,024 `p0`, 1,024 `p1`, 169 `p2`,
92 position-3–5 and 2 position-6+ states, so a tail expert was unidentifiable. Version
2 pooled to `p0 / p1 / p2_plus` and ran. The only trace of its result is the fast-path
doc's sentence that "a much cheaper vocabulary head produces only a small
complete-model speedup." Whether the locality gate (on-policy local head beats
`same_pc4_31_everywhere`, `learned_global_r32`, and the permuted-expert control)
passed is **not recorded**.

## 64. Systems fast path: lossless arms adopted, numbers not recorded

`OFFICIAL_CODI_SYSTEMS_FASTPATH.md` (2026-09-05). Nine cumulative arms from the
released eager path (B0) through body-only decoding without discarded logits (B1),
merged LoRA (B2), fast tokenizer (B3), length bucketing (B4), FP16 (B5), M=5 (B6),
numeric vocabulary (B7), and `torch.compile` (B8). Every later notebook in both tracks
runs on the merged-LoRA, body-only decoder with a decoded-string parity gate, so B1–B2
were adopted as lossless infrastructure. The B3 (≥1.20× batch-one) and B6 (≥1.50× at
≥98% accuracy) gate outcomes are **not recorded**. This is the only experiment in the
September set that targeted the transformer rather than the head or the cache.

## 65. Trajectory-whitened global head: rank 96 retains 98.43%

`kaggle_global_low_rank_lm_head.ipynb` (2026-09-05), Kaggle dataset
`trajectory-whitened-global-low-rank-lm-head`. One head shared across every visible
answer position; initialization by randomized SVD of `W S` (activation-whitened);
KL + top-token CE + ranking-margin distillation; nested rank 32/64/96 prefixes; one
on-policy recovery round; 1,024 / 256 / 256 question-disjoint train splits; test
opened once. Validation top-token agreement went 91.44% → 93.29% (clean) → 93.46%
(recovery).

| arm | correct | exact match | of dense | paired 95% CI vs dense |
|---|---:|---:|---:|---|
| dense | 572 | 43.366% | 100% | — |
| rank 32 | 539 | 40.864% | 94.23% | [−4.018, −0.910] pp |
| rank 64 | 559 | 42.381% | 97.73% | [−2.123, +0.076] pp |
| **rank 96** | **563** | **42.684%** | **98.43%** | [−1.744, +0.379] pp |
| adaptive 32→64 | 560 | 42.456% | 97.90% | [−1.971, +0.227] pp |

Rank 96 passed the 98% retention gate. This is the quality ceiling of route 2 on this
checkpoint: a 7.9× arithmetic reduction of the head at a nine-question cost.

## 66. Deployment benchmark: no end-to-end speedup; route 2 closed as a systems route

`kaggle_global_head_deployment_benchmark.ipynb` (2026-09-05), frozen rank-96
artifact, merged-LoRA FP16 body-only decoder, dense vs eager vs compiled vs Triton
projection-plus-blockwise-argmax.

| batch | dense head | rank-96 head | head speedup | dense end-to-end | rank-96 end-to-end | e2e speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 355.15 µs | 87.58 µs | 4.06× | 92,685.59 µs | 92,317.57 µs | 1.004× |
| 8 | 416.95 µs | 133.43 µs | 3.12× | 12,417.29 µs | 12,364.39 µs | 1.004× |
| 32 | 535.97 µs | 136.03 µs | 3.94× | 3,301.34 µs | 3,336.08 µs | 0.990× |

Compiled rank 96: 0.999×. Triton rank 96: 1.001× at 42.305% (97.72%, below the
gate). Deployed storage grew from 263.83 MB to 273.73 MB because GPT-2 ties the output
matrix to the input embeddings, which stay dense. The ≥1.10× batch-one gate failed in
every arm. The head is a small fraction of CODI's latency once the body is optimized,
so no head-only change can move the end-to-end number; §59's "11× faster readout"
was a component figure and does not compose into a deployment result.

## 67. U28-to-rank-96 bridge: the learned head relies on the causal band

`kaggle_codi_28_to_global96_bridge.ipynb` (2026-09-06, commit `0d04b20`). No
training; connects §40's U28 (PCs 4–31 at the answer colon) with §65's rank-96 head.

- Row-space capture of U28 by the learned rank-96 down-projection: **58.58%**,
  against 12.5% isotropic expectation and above the 200-control null.
- Rank 96 with only U28 retained at answer position 0: 39.121% (91.65% of rank 96).
- Rank 96 with U28 removed at position 0: 8.491% (a 34.19-point loss).
- The fixed U28 head used at every position "failed completely at later answer
  positions", confirming §61's 5.08%.

All three preregistered conditions for a shared mechanism held. This is the one
positive mechanistic result of track 1: a head trained only on trajectory logit
fidelity rediscovers and depends on the band that §40 found by intervention. It is
also the last point at which track 1 produced new knowledge.

## 68. Track 1 remainder: report written, two experiments built and not run

- `output/documents/CODI_Global_Low_Rank_LM_Head_Report.docx` and its PDF
  (2026-09-08) report §65–§67 with the systems and quality results separated.
- `CODI_CONSTRAINED_U28_GLOBAL_HEAD.md` (2026-09-06): freeze U28 as the first 28
  bottleneck columns and learn the residual 36 or 68 orthogonal directions, against an
  energy-matched random-28 constraint and an equal-budget unconstrained continuation.
  Implemented (`src/mech/constrained_global_head.py`, notebook, tests), **uncommitted,
  never run**.
- `kaggle_cross_model_global_head_benchmark.ipynb` (2026-09-12): the §65 method
  against weight-SVD, ASVD-style, and SVD-LLM-style baselines on WikiText-fitted heads
  evaluated across seven corpora, per-model shards. Built; **outcome not recorded**.

Track 1 status: closed as a systems route (§66); the constrained-U28 question is open
but low value, since §66 already shows that no rank-64 head changes latency.

## 69. Layerwise U28 transport and causal xKV: flat, superseded

`CODI_LAYERWISE_U28_CAUSAL_XKV.md` (2026-09-12). Ridge-transported U28 into every
block, then protected it as a core inside cross-layer SVD at matched total rank. The
first-token retain/remove screen produced a flat binary-only result that the next
notebook calls "the insensitive binary-only test that produced the previous flat
plot." Superseded before any compression claim.

## 70. Direct layerwise KV: invalid, disconnected gradients

`CODI_DIRECT_LAYERWISE_KV_EXPERIMENT.md` (2026-09-13). Hooked `ln_1` outputs were
differentiated against the answer loss, but under the active Transformers cache path
those tensors were disconnected from the loss. The reported rank-28 lists were forced
candidates, not validated directions. Recorded as an implementation error, not a
null.

## 71. Pre-answer KV subspaces: connected, but no single eigenvector survives

`CODI_PREANSWER_KV_SUBSPACE_EXPERIMENT.md` (2026-09-14). Exact gradients on the
real pre-answer K/V tensors (100% connectivity gate passed), per-block covariance
eigenvectors mapped through LoRA-aware `W_K`/`W_V`, variable-rank selection with
split stability and Benjamini–Hochberg correction across 9,216 layer-direction
hypotheses. **No individual covariance eigenvector passed.** Compression stayed
blocked.

## 72. Task-sensitive hidden-space subspaces: projector mismatch, superseded

`CODI_TASK_SENSITIVE_KV_SUBSPACE.md` (2026-09-14). Built the symmetric task matrix
`M = −½ E[x gᵀ + g xᵀ]` in the 768-dimensional `ln_1` space so linear combinations of
covariance directions could carry the signal. The hidden-to-cache mapping and separate
orthonormalization changed the projector the causal intervention actually tested, so
the run was set aside rather than interpreted.

## 73. Direct native-cache task subspaces: candidates found

`CODI_DIRECT_CACHE_TASK_SUBSPACES.md` (2026-09-14). Same task matrix, fitted
directly in each block's cached key and value vectors with independent K and V ranks;
four disjoint train splits (1,024 / 1,024 / 256 / 128); family-wise corrected paired
bootstrap across 12 layers. Produced the chain's frozen candidates: **layer 11 value
rank 1** as primary and layers 2, 3, 5 as secondary.

## 74. Native-KV confirmation: layer 11 imported downstream as confirmed

`CODI_NATIVE_KV_CONFIRM_AND_COMPRESS.md` (2026-09-14). The next 512 questions in the
deterministic sampling order confirmed layer 11 / value rank 1 against four
energy-matched random bases with fold consistency, ≥95% retain-only first-token
agreement, and ≤0.10 retain-only NLL rise. Every later runner imports "the completed
layer-11 confirmation" and its frozen basis, so the primary gate is treated as passed;
the interval itself is **not recorded** here. The same run produced exploratory
layer-2 `(K8, V48)` and layer-3 `(K16, V64)` candidates.

## 75. Task-aware protected xKV: a rank-16 discovery on the first 256 test rows

`CODI_TASK_AWARE_PROTECTED_XKV.md` (2026-09-14, fixed through 2026-09-20 for
frozen-model gradient calibration and evaluation-slice alignment). Arms at ranks 16,
32, 48: per-layer SVD, grouped xKV, answer-Fisher-weighted xKV, sparse causal residual
on the six latent rows, and the full weighted + layer-adaptive + protected method;
20 random protected controls at rank 32; INT8/INT4 proxies. The confirmatory
layer-2/3 outcome is **not recorded**. The run's carried-forward result: on GSM8K
**test rows 0:256** the full method beat ordinary xKV at rank 16, which became the
frozen hypothesis for §76. Note that this is the first experiment in the project to
use GSM8K test rows for discovery; the chain then partitioned the test set into
discovery 0:256, confirmation 256:768, and final 768:1319.

## 76. Rank-16 confirmation: better NLL and accuracy, fidelity gate failed

`kaggle_codi_rank16_xkv_mechanism_confirmation.ipynb` (2026-09-20), rows 256:768,
everything imported unchanged. The full method again beat ordinary xKV in answer NLL
and GSM8K accuracy, but dense first-token top-1 agreement fell below the frozen 95%
requirement, so the compound gate failed, the mechanism ablations did not run, and
rows 768:1319 stayed locked. The chain's only positive compression signal therefore
lives at a rank where the compressed model no longer makes the dense model's
decisions.

## 77. Fidelity frontier: no candidate reaches 95% fidelity

`kaggle_codi_xkv_fidelity_frontier.ipynb` (2026-09-20). Fifteen frozen candidates,
ranks {16, 24, 32, 40, 48} × answer-Fisher weight {0, 0.5, 1}, each with the layer-11
protected basis, selected on the already-open rows 256:768. **None passed**; even rank
48 reached only about 85% dense first-token agreement in this grouped cross-layer
factorization. Execution stopped; final rows untouched. Kaggle dataset
`can-xkv-without-giving-up-its-storage-advantage`.

## 78. Fidelity residual: comparator switched to per-layer xKV, no candidate passed

`kaggle_codi_fidelity_residual_xkv.ipynb` (2026-09-20). Changed mechanism: ordinary
**per-layer** xKV at ranks {48, 64, 80, 96} plus a label-free residual of rank
{1, 2, 4} aligned with the dense top-1-versus-runner-up margin gradient, stored only
for the six latent rows; screened on a fresh 512-question train slice. No candidate
passed the five-check fresh screen; final rows untouched. Note the comparator change:
§75–§77 factorized grouped cross-layer caches, §78–§81 factorize per layer. Per-layer
rank 64 sits near dense (§79), whereas grouped rank 48 sat at 85% fidelity (§77). The
two "ordinary xKV" baselines are not the same object.

## 79. Adaptive per-component allocation: screen passed, final interval crossed zero

`kaggle_codi_adaptive_kv_allocation.ipynb` (2026-09-20/21), Kaggle dataset
`can-adaptive-beat-uniform-per-layer-xkv`. Rank-utility curves per layer × K/V from
exact answer-NLL gradients on 512 fresh train questions; nine candidates (budget rank
{48, 64, 80} × Fisher blend {0, 0.5, 1}) screened on another 512; padding charged so
modeled bits match ordinary per-layer xKV exactly. The screen selected
`adaptive_kv_r64_w1` (rank-64 budget, pure answer-Fisher) at **1.48× modeled
compression** with NLL advantage +0.0108, CI [+0.0034, +0.0191], first-token fidelity
97.46%. This opened the final 551 test rows for the first time.

| final slice (551) | value |
|---|---:|
| NLL advantage over ordinary xKV | +0.00387 |
| paired 95% CI | [−0.00728, +0.01495] |
| first-token fidelity | 98.19% |
| adaptive accuracy | 43.38% |
| dense accuracy | 43.38% |
| ordinary per-layer xKV accuracy | 43.01% |

Every gate passed except the NLL lower bound, so the decision is
`STOP: the adaptive allocation did not replicate on the locked final slice`.
Descriptive mechanism from the utility curves: answer sensitivity concentrates in
layers 6–11, layer 11 is important but not uniquely so, keys carry more sensitivity
than values, and rank 16 captures over 90% of the Fisher utility for most components.
The full GSM8K test set is now opened for this chain.

## 80. Request-adaptive profile router on SVAMP: nothing left to route

`CODI_REQUEST_ADAPTIVE_XKV_ROUTER.md` (2026-09-21). Multi-output ridge router over
twelve pre-answer lexical features, choosing per request among the three frozen rank-64
profiles (reconstruction, hybrid, answer-Fisher); SVAMP 1,000 rows (700 train + 300
test, after a fix that had loaded only the 300-row split) hash-split 400 fit / 300
screen / 300 final. Screen result: answer-Fisher KL from dense ≈ 1×10⁻⁶, hybrid ≈
3.9×10⁻⁵, reconstruction ≈ 1.7×10⁻⁴; every arm at exactly 45% accuracy and 100%
first-token fidelity; router matched the per-question oracle 50% of the time; routed
minus global-Fisher KL improvement −1.51×10⁻⁷, CI [−4.24×10⁻⁷, +4.35×10⁻⁹]. Screen
failed; final 300 locked. The fixed answer-Fisher rank-64 profile is already
indistinguishable from dense, so routing among rank-64 profiles measured noise.

## 81. Cache-risk rank router: built, uncommitted, not run

`CODI_CACHE_RISK_RANK_ROUTER.md` (2026-09-21). Keeps the answer-Fisher profile and
routes **rank** {48, 64, 80} per request under a fixed rank-64 aggregate bit budget,
using 27 spectral tail-energy features of the dense pre-answer cache plus question
features; GSM-Hard 1,319 rows split 600 / 400 / 319; requires the failed §79 artifact
as predecessor. Implemented with tests and a notebook builder; nothing committed and
no run executed. §82 recommends not running it as designed.

## 82. Assessment and decision point (2026-09-21)

### What the September work established

1. **Route 2 is closed as a systems route.** A rank-96 trajectory-trained head keeps
   98.43% of accuracy (§65) and relies on the §40 band (§67), but delivers 1.004×
   end-to-end (§66) and increases stored bytes. The §59 microbenchmark did not
   compose into a deployment gain because the head is a small share of latency.
2. **The cache holds a readable answer-sensitivity structure** (layer 11 value
   direction, layers 6–11 concentration, keys over values, low-rank Fisher utility;
   §73, §74, §79), and protecting it improves NLL at rank 16 (§75, §76). But every
   compression scheme that reaches dense-level fidelity does so at a rank where
   ordinary per-layer xKV is already dense-equivalent (§79, §80).
3. **Neither track touched the CODI-versus-KaVa question.** §55 named
   trajectory-level supervision as the natural continuation; it was not pursued.

### Why the KV track circled

- **No headroom at the operating point.** Per-layer rank-64 answer-Fisher xKV is
  indistinguishable from dense CODI (§79, §80). Allocation and routing experiments at
  that budget can only measure noise, and §81 repeats the choice.
- **Gates on quantities nobody would cite.** Decision rules were paired intervals on
  NLL or KL differences of a few thousandths or less; the accuracy comparisons that
  matter tied exactly. With the §79 final-slice interval half-width ≈ 0.011 at n = 551
  the per-question SD is ≈ 0.13, and detecting the screen's +0.0108 at 80% power
  needs roughly 1,300 paired questions. The screen-to-final shrinkage (+0.0108 →
  +0.0039) is ordinary winner's curse from selecting on the screen, not a near miss.
- **1.48× modeled compression of a 12-layer, 768-wide, ~100-token cache is not a
  deployment result**, and every protocol doc already disclaims latency. The track
  used systems vocabulary while producing mechanistic observations, then judged itself
  by systems criteria it could not meet.
- **Benchmarks were spent one per experiment.** GSM8K test is fully opened (§75–§79);
  700 of 1,000 SVAMP rows are opened (§80); §81 would open GSM-Hard, where dense CODI
  is weak enough that accuracy non-inferiority passes trivially.
- **Hard predecessor-artifact chaining** (§78–§81 each require the previous failed
  summary) pinned every follow-up to the same operating point and made most of the
  last day's commits plumbing fixes.
- **The ledger lapsed on 2026-08-28.** The step that forces "what did we learn, what
  is the decision" was skipped for the entire period in which the circling occurred.

### Standing additions to §17

- Do not run allocation, routing, or residual experiments at a budget where the
  ordinary comparator already matches dense first-token decisions within a point.
- Gate compression claims on paired exact-match accuracy at matched storage, or on
  storage at matched accuracy. Do not preregister an NLL or KL interval as the primary
  gate without a pilot-based minimum-detectable-effect check for the planned n.
- Do not open a new external benchmark for a method-development follow-up. One
  reserved benchmark, opened once, after an accuracy-gated screen.
- Do not describe reconstructed-cache, modeled-bit results as compression or latency
  results outside the mechanism framing.
- Do not build a follow-up whose runner hard-requires a failed predecessor artifact
  unless the follow-up changes the operating point.
- Append a ledger entry before building the next experiment, not after.

### Decision point

> Do not run §81 as designed. Choose one of:
>
> (a) **One decisive allocation test at a degraded operating point.** Take the rank at
> which ordinary per-layer xKV loses at least five accuracy points on the §77/§79
> curves, test §79's Fisher allocation there with a paired exact-match gate at matched
> storage, on the reserved benchmark, once. Close the KV line either way.
>
> (b) **Write the KV track up as mechanism**, not systems: where CODI's
> answer-sensitive K/V information lives (§73, §74, §79) and the rank-16 protection
> effect with its fidelity limit (§75–§77).
>
> (c) **Return to the thesis question with trajectory-level supervision.** §55 shows
> the odd thoughts' value slots hold the intermediates; a KaVa-style distillation loss
> that scores those slots against teacher intermediates is the first training
> experiment that reconnects the mechanistic findings to §1. This is the only option
> that can produce a chapter rather than another null on the frozen checkpoint.
>
> Recommendation: (c), with (b) as the write-up of what already exists. (a) only if a
> compression claim is required for the semester deliverable.

Housekeeping before any of the above: commit or delete the uncommitted §68 and §81
files; fill the **not recorded** cells in §62, §63, §64, §74, §75 from their Kaggle
summaries; add a README pointer to §60–§82.

## 83. Completed fixed per-head K/V basis experiment: `selection_not_passed`, test unread

Kaggle run 2026-09-21, code pinned at `9d44081`, notebook `84cb90a`, protocol in
[`CODI_FIXED_BASIS_KV.md`](CODI_FIXED_BASIS_KV.md). This is the experiment §82
should have replaced §69 with: it asks whether the cache has a *fixed*,
question-independent low-dimensional subspace per attention head, in the sense that
§40 and §67 established for the final hidden state. One orthonormal basis per (layer,
kind, head) was fitted from uncentred second moments on 1,024 GSM8K-train questions
(seed 20260921) over every cached row type; every key and value was then projected
onto its head's leading `r` directions. GPT-2 has no rotary embedding on keys, so the
projection is algebraically an `r`-dimensional per-head attention, not a proxy.
Storage is `head_dim / r` per component and does not decay at short context, which
was the whole reason to ask the question at CODI's scale. Only the reproduction
summary was attached; no xKV-chain artifact was required.

The frozen selection rule required, on the disjoint 256-question selection split,
at least 98% of dense exact match and at least 95% dense first-token agreement. No
rank in the grid `{8, 16, 24, 32, 40, 48}` passed, so the runner stopped and the
GSM8K test set was **not read**. Dense selection-split accuracy was 66.8% (train
questions; the checkpoint was trained on a GSM8K-derived set).

| rank / head | storage | uniform retained | first-token agreement | energy-allocated retained | random retained (2 seeds) |
|---:|---:|---:|---:|---:|---:|
| 48 | 1.33× | 93.0% | 77.7% | 35.1% | 32.7 / 35.7% |
| 40 | 1.60× | 86.5% | 69.5% | 24.0% | 9.4 / 9.4% |
| 32 | 2.00× | 75.4% | 57.8% | 11.7% | 2.9 / 2.3% |
| 24 | 2.67× | 58.5% | 41.4% | 5.3% | 4.1 / 1.8% |
| 16 | 4.00× | 17.5% | 14.1% | 4.1% | 0.6 / 2.9% |
| 8 | 8.00× | 5.3% | 3.5% | 2.9% | 2.3 / 2.3% |

Head-averaged rank needed to retain a given fraction of second-moment energy
(head width 64):

| target | keys, range over layers | values, range over layers |
|---|---|---|
| 90% | 14.3 (layer 2) to 39.6 (layer 11) | 36.9 to 46.6 |
| 95% | 26.9 to 49.7 | 47.1 to 54.6 |
| 99% | 50.3 to 60.6 | 59.3 to 62.1 |

Row-type subspace agreement (fraction of the question-row subspace captured, mean
over heads; isotropic expectation `r/64` in parentheses): latent rows 0.42 (0.25) at
rank 16 and 0.62 (0.50) at rank 32; answer rows 0.38 and 0.59; cue rows 0.46 and
0.63. The workspace rows share the question-row geometry only partially.

### What this establishes

1. **The per-head basis is real but the cache is not low-rank in it.** The fitted
   basis beats random bases by 40 to 70 points at ranks 24 to 40, so heads do write
   into question-independent directions. But 95% of energy needs roughly 34 to 50
   key directions and 47 to 55 value directions out of 64, there is no knee, and
   keeping 48 of 64 still loses 7 points of accuracy and 22% of first-token
   decisions. The fixed 28-dimensional structure of the output (§40) has no
   analogue inside the cache at head level.
2. **The cache's compressibility is per-request, not fixed-basis.** Per-request
   xKV at rank 64 per layer was dense-equivalent at 1.48× (§79); the fixed basis at
   1.33× is not. The low-rank structure xKV exploits lives across the tokens of one
   request, in the token-side factor, not in a direction set shared across requests.
   This is the empirical answer to "why not SVD each layer's space once": it loses.
3. **Values are less compressible than keys** at every layer and every energy
   target, and early-layer keys are the most compressible of all. Together with
   §79 (keys carry more *answer-specific* sensitivity) the picture is consistent:
   keys need only the directions queries probe, values must carry the content that
   is summed into the residual stream.
4. **Allocating rank across heads by raw eigenvalue is wrong.** The energy-greedy
   arm was exactly optimal for retained energy and landed near random (35% versus
   uniform's 93% at rank 48). Raw eigenvalues across heads are dominated by
   activation scale, so the rule strips rank from small-norm heads the computation
   depends on. Any cross-component allocation must use per-component energy
   fraction or a loss-based marginal utility.

### Caveats

The selection split is GSM8K train (dense 66.8%, not the 43.4% of test), so
retention fractions on test could differ; the energy tables make the direction of
the conclusion robust to that. The moments are uncentred and include the
first-position token, whose keys carry very large norm in GPT-2; that can inflate one
leading direction per head, costing one rank unit, not the twenty separating this
result from the gate. PCA keeps the directions a head *writes*; a basis fitted under
the query metric (keys) or the output-projection metric (values) would keep what
attention *reads*, and could be sharper. That is the one remaining diagnostic in this
line (§83 decision point); it is not expected to reach the gate.

### Standing additions to §17

- Do not allocate rank across attention heads or layers by raw eigenvalue or raw
  reconstruction energy; scales differ by component. Use energy fraction or a
  loss-based marginal utility.
- Do not propose further fixed-basis KV compression of this checkpoint at the
  accuracy gate; the per-head variance spectrum has no knee below rank ~50.
- Treat the cache's low-rank structure as per-request (token-side) unless a
  read-side (query- or output-weighted) basis shows otherwise.

### Decision point

> The KV track is closed as a compression line and complete as a mechanism line.
> Write it up as one section: answer sensitivity in layers 6–11 and in keys (§79),
> the causally specific layer-11 value direction (§74), rank-16 protection that
> helps NLL but breaks fidelity (§76–§77), per-request rather than fixed-basis
> compressibility (§83), near-full-rank per-head geometry (§83).
>
> Optional single diagnostic, uniform ranks 24 and 32 only, no new gates: refit the
> bases under the query metric for keys and the output-projection metric for values,
> to ask whether the cache is full-rank in what heads write or also in what queries
> read. It must not delay the next item.
>
> Next experiment: trajectory-level supervision (§55, §82 option c). Append its
> ledger entry before building it.

## 84. Preregistered: trajectory-level supervision with mechanism-selected targets

Written before any code, per the §83 standing rule. Protocol in
[`CODI_TRAJECTORY_SUPERVISION.md`](CODI_TRAJECTORY_SUPERVISION.md); contract
`official_codi_trajectory_supervision_v1`.

### Question

§1 asked whether KaVa's KV-trajectory supervision improves on CODI's endpoint
distillation under matched conditions. §55 established what CODI's trajectory
*holds*: the odd latent slots (0-based 1, 3, 5) store the solution's intermediate
values, unordered. This experiment asks the §1 question with that finding as the
selector: does supervising the value-holding slots toward the teacher positions
that emit intermediate values improve on (a) endpoint-only CODI, (b) KaVa's
redundancy-selected targets, and (c) matched controls that differ only in which
positions or which slots are supervised?

### Arms, all warm-started from the frozen official checkpoint

Identical data, order, optimizer, steps, and base objective (student gold-answer
NLL plus the official endpoint hidden loss at the answer cue, smooth-L1 over all 13
states with teacher-std normalisation). Every auxiliary term has its gradient norm
matched to the endpoint term's gradient norm each step, so arms differ in *what*
they supervise, not how hard.

| arm | auxiliary target | teacher positions | student slots |
|---|---|---|---|
| `codi` | none | — | — |
| `kava` | K and V, L1 | R-KV (λ = 0.1), per layer and head | all six |
| `value_odd` | K and V, L1 | first token of each `<<…=v>>` result in the truncated trace | 1, 3, 5 |
| `random_odd` | K and V, L1 | seeded random trace positions, count matched per example | 1, 3, 5 |
| `value_even` | K and V, L1 | as `value_odd` | 0, 2, 4 |
| `recon_odd` | cross-entropy of the slot's own readout toward the value's first token | as `value_odd` | 1, 3, 5 |

Slot-to-value assignment is a per-example minimum-cost injective matching on the
detached loss, because §55 showed the store is unordered. Examples with fewer values
than slots leave the surplus slots unsupervised; examples with more let the matching
choose. `recon_odd` is the generative-objective comparison and is the arm §58 warns
about: it trains the readable component directly.

### Frozen budget

GSM8k-Aug (`eq_only`, the released training distribution), unique questions,
digit-leading answers, at least two equations so the truncated trace has one
value: 8,192 training and 256 selection rows, data seed 20260922. Batch 8, one epoch
(1,024 steps), AdamW, learning rate 2e-5 constant, weight decay 0, gradient clip 1.0,
float32, training seeds 1, 2, 3. Trainable parameters are the released LoRA
adapters and the projector, as in §30. The teacher path is frozen (no teacher CE),
identically for every arm.

### Screen, then one test read

After `codi` seed 1 trains, its selection-split accuracy must be within 3 points of
the frozen checkpoint's on the same split. Failure means the learning rate wrecked
the model; the run stops and the test set is not read. On pass, every arm and seed
is evaluated once on the full 1,319-question GSM8K test with the official native
decoding protocol, plus the frozen checkpoint as reference.

### Gates (paired bootstrap over questions on per-question seed-mean correctness)

1. **H1, KaVa versus CODI:** `kava − codi` lower bound > 0.
2. **H2, selector:** `value_odd − kava` lower bound > 0 **and** `value_odd −
   random_odd` lower bound > 0.
3. **H3, slot specificity:** `value_odd − value_even` lower bound > 0.
4. **H4, objective:** `recon_odd − value_odd`, reported two-sided, no gate.

The headline claim "mechanism-selected trajectory supervision helps" requires H2
and `value_odd − codi` lower bound > 0. Per-seed accuracies, the selector-overlap
diagnostic (fraction of R-KV picks that land on value tokens), assignment statistics,
and gradient-scale records are reported regardless.

### Stated expectations

Modest effects at best. §17 records that R-KV was not better than uniform selection
in the pilot, and on equation-only traces the value tokens are plausibly among the
least redundant tokens, so the selectors may largely coincide; the overlap
diagnostic measures exactly that. Nulls are informative here: they would say the
§1 question is not decided by which trajectory positions are supervised at this
budget. The 1,024-step warm start is a cheap held-out gate (§17 rule 6), not the
final word on longer training.

## 85. Completed trajectory-level supervision: `STOP`, a null bounded by its own weighting

Kaggle run 2026-09-22, code `12f5eac`, notebook `bbf3f87`. Screen passed (continued
CODI 84.8% versus frozen 82.4% on the selection split). Full test, three seeds per
arm, native decoding:

| arm | test mean | vs frozen 43.44% |
|---|---:|---:|
| codi | 43.54% | +0.10 |
| recon_odd | 43.62% | +0.18 |
| random_odd | 43.14% | −0.30 |
| value_odd | 43.14% | −0.30 |
| value_even | 43.11% | −0.33 |
| kava | 42.96% | −0.48 |

All fifteen paired intervals cross zero; every gate failed. The training curves are
the finding: answer NLL and endpoint loss show no trend over 1,024 steps and
continued CODI ended 0.1 points from the frozen model. Auxiliary gradients were
matched to the endpoint term's gradient norm, and that term is at its optimum for
the released checkpoint, so every auxiliary was neutered equally. The null bounds
the weighting and the budget, not the selectors. Two facts survive: only 18.8% of
R-KV picks land on value tokens, so the selectors do not coincide (§84's stated
prior was wrong), and direct readout cross-entropy on the value slots neither
helped nor hurt. Standing addition to §17: never norm-match an auxiliary loss to a
reference term that is already converged; a warm start of a converged checkpoint
cannot discriminate training targets.

## 86. Preregistered: causal versus variance selection of the distillation subspace

Written before any code. Protocol in
[`CODI_CAUSAL_SUBSPACE_DISTILLATION.md`](CODI_CAUSAL_SUBSPACE_DISTILLATION.md);
contract `official_codi_causal_subspace_distillation_v1`.

### Question

Subspace distillation methods (LoRi, June 2026; Flex-KD; SubDistill; SPREAD) choose
the subspace by teacher variance, second moments, or gradient relevance. §40 showed
that at CODI's decision state the top-variance directions are inert and accuracy
lives in a low-variance band; "Function Lives Where Variance Doesn't" (September
2026) makes the same point generally; and §43–§52 showed gradient-selected
directions repeatedly failed matched causal tests. Nobody has asked whether a
variance- or relevance-selected distillation target wastes its rank on directions
the answer never uses.

> When a latent student is distilled toward a rank-r subspace of the teacher's
> decision state, does selecting that subspace by causal retain-and-remove on the
> teacher transfer more accuracy than selecting it by variance, by gradient
> relevance, or at random, at matched rank?

### Design

**Teacher.** The frozen official CODI checkpoint in its explicit-CoT teacher mode.
Its post-`ln_f` colon state (state 12) is precomputed for every row used. All
subspaces are index sets over the teacher's own principal components at state 12,
fitted on 2,048 train rows, so arms differ only in *which* PCs are kept:

- `variance`: the first r PCs (LoRi-style).
- `relevance`: the r PCs with the largest Fisher score E[(g·v)²], g the gradient of
  the gold first-token NLL at state 12 through the readout (Flex-KD-style).
- `causal`: greedy forward selection over the top-128 PCs maximising retain-only
  first-token accuracy, ties broken by mean answer margin, on a disjoint 2,048-row
  split, the §36 analytic tier. (Amended before the first run: the draft optimised
  mean gold log-probability, which a synthetic test showed is dominated by
  confident errors under sharp logits, so partially restored states scored below
  the constant mean state and greedy drifted to inert directions. Accuracy and
  margin do not saturate.)
- `random`: r seeded PCs.
- `full`: all 768 dimensions (single-state CODI).
- `none`: answer cross-entropy only.

Rank is chosen on the teacher, on a third disjoint 2,048-row split, from {8, 12, 16}:
the smallest r whose causal set retains at least 75% of the teacher's dense
first-token accuracy under retain-only and beats the variance set by at least five
points. If no rank satisfies both, the selectors are not distinguishable on the
teacher and the run stops before any training.

**Student.** Base GPT-2 with the checkpoint's embedding table (so the special tokens
and the shared readout are meaningful), fresh LoRA r = 128 adapters (A random, B
zero) and a fresh projector: the latent task is learned from scratch, the readout is
shared with the teacher. Loss: answer cross-entropy plus the state-12 distillation
term, smooth-L1 over the selected coordinates scaled by the teacher's coordinate
standard deviation, with the distillation gradient norm-matched to the
cross-entropy gradient each step (the reference term is being learned, not
converged, so §85's failure mode does not apply). Six arms, two training seeds split
across two Kaggle accounts, 10,000 steps at batch 16 (160,000 GSM8k-Aug rows, about
0.4 epoch), AdamW at 1e-4 cosine with 500 warm-up steps, weight decay 0.1, gradient
clip 2.0, float32. Runs checkpoint every 500 steps and resume across sessions.

### Outcomes and gates

Selection-split curve every 1,000 steps: teacher-forced answer NLL, first-token
accuracy, exact match on 256 rows. Final: full GSM8K test once per run, native
decoding, exact match primary, teacher-forced NLL secondary. Paired bootstrap over
questions on per-question seed-mean correctness:

1. **S1, sanity:** `full − none` lower bound > 0. If distillation itself is not
   detectable at this budget the subspace comparisons are uninformative and the
   claim is `STOP: distillation signal not detectable`.
2. **H1:** `causal − variance` lower bound > 0.
3. **H2:** `causal − relevance` lower bound > 0.
4. **H3:** `causal − random` lower bound > 0; `variance − random` reported.
5. `causal − full` two-sided: whether a rank-r causal target matches the 768-d one.

Headline requires S1, H1 and H3. Teacher-side retention of each set, set overlaps,
gradient scales and curves are reported regardless.

### Amendment 2, before training: the rank rule

The first Kaggle session (2026-09-22, seed-1 account, code `dc281ac`) ran the
smoke pass end to end at about 1.2 steps per second, precomputed the teacher cache
in 34 minutes, and stopped at the rank rule. On the 2,048-row validation split the
teacher's dense first-token accuracy is 89.4% and retain-only gives:

| rank | variance | relevance | causal | random | variance share of `variance` |
|---:|---:|---:|---:|---:|---:|
| 8 | 10.5% | 25.5% | 33.1% | 3.9% | 87.9% |
| 12 | 32.3% | 38.6% | 48.5% | 2.8% | 89.6% |
| 16 | 51.3% | 48.3% | 59.1% | 2.3% | 91.0% |

Three teacher-side facts stand regardless of what follows: the top eight PCs hold
88% of the variance and 12% of the accuracy (§40 replicates on the explicit-CoT
teacher path); causal beats variance by 23, 16 and 8 points; causal beats gradient
relevance by 8, 10 and 11 points (the §43–§52 lesson again). The gap clause passed
at every rank and the 75% retention clause failed at every rank, because this
teacher needs more than 16 PCs to keep 75% of its accuracy and at such ranks the
sets converge. The two clauses conflict on this teacher; the 75% figure was a
judgment call made before any teacher number was seen. Amended rule, applied before
any student is trained and before the test set is read for this experiment: among
ranks with a gap of at least five points and causal retention of at least 50%,
choose the rank maximising gap times retention. On the observed table that is rank
12 (54% retained, 16-point gap). The `full − none` sanity gate remains the guard
against a target too weak to matter.

### Stated expectations and risks

From-scratch adapters at 0.4 epoch may reach low absolute accuracy (the §2 pilot
reached 13% after one epoch of joint training), so the primary comparison may be
underpowered even though n = 1,319; the selection-split NLL curve is the sensitive
secondary. The teacher-side rank rule protects against the §84 mistake: if the
variance and causal sets barely differ, nothing is trained. A positive result is a
targeted correction to a June 2026 method; a null with S1 passing says the inert
directions cost nothing during training, which is also worth knowing.

## 87. Interim, seed 1 of §86: primary STOP, secondary ordering favours causal selection

Kaggle run 2026-09-22, seed-1 account, code `dfc4836`, six arms, 10,000 steps each,
rank 12 chosen by the amended rule (causal set PCs 5–19 skipping 8, 16, 18; variance
set PCs 0–11; Jaccard 0.33). Two out-of-memory failures preceded it: batch 16 in
float32 did not fit a T4 (fixed by exact micro-batch accumulation) and the training
loop retained per-step gradient tuples in its logging window (fixed; standing
lesson: never keep step results that hold GPU tensors).

**Primary (exact match, 1,319 test questions).** Every arm sits at 3.8–4.2%; all
intervals cross zero; `full − none` fails, so the preregistered claim is
`STOP: distillation signal not detectable at this budget`.

**Secondary (teacher-forced answer NLL, lower is better).**

| arm | test NLL |
|---|---:|
| none | 2.348 |
| causal | 2.358 |
| relevance | 2.366 |
| random | 2.373 |
| variance | 2.385 |
| full | 2.390 |

Paired intervals: `variance − causal` +0.027 [+0.014, +0.040]; `full − causal` +0.032
[+0.017, +0.048]; `random − causal` +0.015 [+0.003, +0.028]; `relevance − causal`
+0.008 [−0.003, +0.019]; `none − full` −0.042 [−0.058, −0.027]; `none − causal`
−0.010 [−0.022, +0.000]. The selection-split curves show the same ordering from
step 4,000, and final selection exact match splits causal and none at 7.4%,
relevance and random at 6.6%, full and variance at 5%.

**Reading.** At 0.4 epoch under norm-matched pressure, distillation is a tax on
answer learning: matching the full state or the variance subspace raises NLL
relative to no distillation, matching the causal subspace costs nothing detectable,
and causal beats variance, full and random with intervals excluding zero. H1 and H3
hold on the secondary while S1 inverts. This extends §40: the inert high-variance
directions are not merely useless to the readout; pulling a learning student toward
them conflicts with learning the answer, and intervention-selected directions avoid
the conflict. Relevance selection, half-overlapping the causal set, sits in between
and is not separable from it.

**Bounds.** One training seed; the bootstrap covers question sampling only; effects
are about 1.3% of NLL; the claim concerns early training dynamics under equal
gradient pressure, not final accuracy, since no student approached CODI's 43%; part
of the tax is the equal-pressure design itself. Seed 2 (second account) runs the
identical protocol; the finding stands only if the ordering replicates.

## 88. Completed §86, two seeds: the cost pattern replicates, the novel contrast is a tie

Seed 2 (second account, code `dfc4836`, identical protocol, rank 12 re-selected
from the same teacher table) completed 2026-09-23. Exact match is noise in both
seeds (3.1–4.3%); S1 fails twice; the preregistered primary is `STOP`. The
secondary, teacher-forced test NLL, across seeds (positive favours the second arm):

| comparison | seed 1 | seed 2 | verdict |
|---|---|---|---|
| none − full | −0.042 [−.058, −.027] | −0.040 [−.055, −.026] | replicated |
| none − variance | −0.038 [−.050, −.025] | −0.023 [−.036, −.009] | replicated |
| none − causal | −0.010 [−.022, +.000] | −0.015 [−.025, −.004] | replicated, smaller |
| random − causal | +0.015 [+.003, +.028] | +0.017 [+.006, +.028] | replicated |
| full − causal | +0.032 [+.017, +.048] | +0.026 [+.012, +.040] | replicated |
| variance − causal | +0.027 [+.014, +.040] | +0.008 [−.006, +.022] | **not replicated** |
| relevance − causal | +0.008 [−.003, +.019] | −0.001 [−.012, +.009] | tie, twice |
| random − variance | −0.012 [−.024, +.000] | +0.009 [−.004, +.022] | sign flipped |

### What stands

1. **Negative transfer from variance and full-state targets.** Under norm-matched
   pressure at 0.4 epoch, distilling toward all 768 dimensions or toward the twelve
   highest-variance PCs of the teacher's decision state raises the student's answer
   NLL relative to no distillation, in both seeds with intervals excluding zero.
   Extends §40 from "the high-variance directions are inert" to "pulling a learning
   student toward them conflicts with learning the answer."
2. **Intervention- and relevance-selected targets avoid most of that cost** and beat
   random and full in both seeds.
3. **Causal versus relevance is a tie** in both seeds. H2, the one comparison not
   already in the literature (Circuit Distillation selects components by ablation;
   Flex-KD and SubDistill select directions by score; no one had put ablation-selected
   directions against score-selected ones), returns a null at this budget: for this
   purpose the gradient proxy is as good as the intervention it approximates.
4. **Causal versus variance** is directionally consistent but the effect fell from
   0.027 to 0.008 between seeds; with two seeds the seed-level variance is not
   estimable and the honest description is "probably small."

### Bounds

Two training seeds; effects of 0.5–1.7% of NLL; early-training regime only, no
student approached CODI's accuracy; the equal-pressure design is itself part of the
tax. Pooled per-question aggregate over both seeds pending (attach both outputs and
rerun the aggregation).

### Decision point

> The KV, head, warm-start and distillation-subspace lines are all closed. Remaining
> quota, if spent at all, goes to one thirty-minute gradient-alignment diagnostic
> (angle between the answer gradient and each target's distillation gradient along
> an early `none` trajectory) to give finding 1 a mechanism. No more seeds, no longer
> training, no new arms. Then write: the mechanistic audit (§40, §55, §58, §83) as the
> body, §85 and §86–§88 as the training-side coda, with finding 1 as the one new
> training-side fact and finding 3 as the honest close of the tested-versus-scored
> question.

## 89. Preregistered: is the variance-target tax transient or permanent? (templated task)

Written before any code. Protocol in
[`CODI_TEMPLATED_SUBSPACE_DISTILLATION.md`](CODI_TEMPLATED_SUBSPACE_DISTILLATION.md);
contract `official_codi_templated_subspace_distillation_v1`.

### Question

§88 established, in two seeds, that distilling an early-stage student toward the
teacher's variance-selected decision-state directions raises its answer NLL above
no distillation, while intervention-selected directions cost far less. On GSM8K no
student reached usable accuracy, so whether that cost persists once the student can
solve the task, or washes out, is unknown. This experiment keeps the official teacher,
GPT-2 and its pretrained geometry, and replaces GSM8K with templated two- and
three-step arithmetic word problems in the equation-only format the teacher was
trained on, so fresh-adapter students reach real accuracy within a few thousand
steps and exact match becomes the primary again.

> Once a latent student can solve the task, does variance-selected distillation
> leave a final-accuracy deficit relative to no distillation, and does
> intervention-selected distillation avoid it?

### Go/no-go on the teacher, before training

1. The official teacher's explicit-CoT generation must reach at least 80% exact
   match on 500 held-out templated problems, otherwise its decision states on this
   task are not trustworthy targets.
2. The teacher-side selector table is recomputed on the templated task (2,048 fit,
   2,048 select, 2,048 validate rows) and the §86 amended rank rule must select a rank
   (causal − variance ≥ 5 points, causal retention ≥ 50%, maximise the product).
3. Dense first-token accuracy at the colon must be at least 80%.

Any failure stops the run with no training.

### Design

Three arms, `none`, `variance`, `causal` (`relevance` dropped: §88 showed it ties
causal; `full` and `random` dropped for budget), five training seeds, 3,000 steps at
batch 16 (two micro-batches of eight), AdamW 1e-4 cosine with 150 warm-up steps,
weight decay 0.1, clip 2.0, float32, same student construction and norm-matched
distillation as §86. Data: 48,000 unique generated training problems, 256 selection,
2,000 held-out generated test problems; no GSM8K row is touched. Curve every 300
steps on the selection split. Runs checkpoint and resume; seeds split across accounts;
`--aggregate-from` pools.

### Gates (paired bootstrap over test questions on seed-mean correctness)

- **T1, persistence:** `none − variance` lower bound > 0 means the tax is permanent
  at this budget; upper bound < 0 means variance distillation has become a benefit;
  an interval covering zero means it washed out.
- **H1:** `causal − variance` lower bound > 0.
- **H1b:** `causal − none`, two-sided, reported.
- **Secondary:** steps to first reach 30% selection-split exact match, per run.

Claims: `PERMANENT` if T1 lower bound > 0 and H1 lower bound > 0; `TRANSIENT` if T1
and H1 both cover zero; otherwise `PARTIAL` with the surviving comparisons named.

### Expectations

Minimum detectable difference is roughly 2.5 points per seed and about 1.5 with
five seeds. The §88 NLL effects were 0.5–1.7% of the loss; if they translate to
under a point of accuracy this returns `TRANSIENT` or a null, cleanly. The result is
a statement about training dynamics on a task the student can finish, not about
GSM8K difficulty. This is the last training experiment the quota allows.

## 90. Completed §89, three seeds: the go/no-go passed, the student never left the floor

Run 2026-09-24 on one account (seeds 1–3, code `0d5150a`, notebook pin `3e50d65`),
≈55 min per (arm, seed). Output published as a Kaggle dataset; the other account's
seeds 4–5 are not needed for the conclusion below.

### Go/no-go (all passed)

- Teacher explicit-CoT generation on 500 held-out templated problems: **82.6%**. The
  failures shown are arithmetic slips inside a correct plan (`63*7=471`), so the task
  transfers to the GPT-2 teacher as intended.
- Dense colon first-token accuracy on the validate split: **88.1%** (GSM8k-Aug: ≈71%).
- Rank rule chose **r = 12** (causal 0.509 vs variance 0.458, gap 0.050, exactly at
  the floor). Top-4 eigenvalue share 0.696.

Selector table (validate split, first-token accuracy / variance share):

| r | variance | relevance | causal | random |
|---|---|---|---|---|
| 8 | 0.160 / 0.77 | 0.038 / 0.04 | 0.269 / 0.15 | 0.000 |
| 12 | 0.458 / 0.83 | 0.100 / 0.06 | 0.509 / 0.21 | 0.000 |
| 16 | 0.689 / 0.87 | 0.221 / 0.10 | 0.689 / 0.87 | 0.000 |

Sets at r = 12: variance `[0..11]`, causal `[2..12, 15]`. The two share ten of twelve
PCs; the causal set drops PCs 0 and 1, which carry ≈62% of the variance and nothing
the readout uses, and adds PCs 12 and 15. At r = 16 the two sets are identical. On
this task the selectors are barely distinguishable; the "inert top PCs" of §40 are
here exactly two directions.

### Training (3,000 steps × batch 16, three seeds per arm)

| arm | test EM seeds 1/2/3 | mean EM | test NLL |
|---|---|---|---|
| none | 1.9 / 2.0 / 0.0% | 1.3% | 2.150 |
| variance | 0.5 / 2.4 / 0.2% | 1.0% | 2.259 |
| causal | 1.1 / 1.3 / 1.0% | 1.1% | 2.270 |

No run reached the 30% selection-split threshold; the best selection-split exact
match at any checkpoint was 2.3%. Selection-split NLL was still falling at step 3,000
(2.1–2.4 nats per answer token). The distillation gradient scale rose from ≈1 to ≈6
over training in both distilled arms, i.e. the norm-matched term kept pace with CE.

Exact-match comparisons (paired bootstrap, seed-mean): `none − variance` +0.003
[−0.001, +0.007]; `causal − variance` +0.001 [−0.002, +0.004]; `causal − none`
−0.002 [−0.005, +0.001]. The runner's decision logic printed **TRANSIENT**.

### Reading: the preregistered claim is vacuous, the secondary replicates

The §89 design assumed a task the student would finish; it did not, so every
exact-match interval is a comparison of floor against floor and the `TRANSIENT`
label does not mean the tax washed out. The design error is mine: §89 gated on the
teacher and not on the student (§86 had S1 for that). The claim is withdrawn; the
experiment is a **STOP: student did not learn the task at this budget**.

The teacher-forced NLL secondary replicates §87–§88 for the third time, now on a
different task and with three seeds: `none < causal < variance`.

| comparison | mean NLL difference | 95% CI |
|---|---|---|
| variance − none | +0.109 | [+0.097, +0.120] |
| causal − none | +0.120 | [+0.110, +0.131] |
| variance − causal | −0.012 | [−0.020, −0.004] |

Here the sign of `variance − causal` is reversed relative to §88: causal is very
slightly *worse* than variance in NLL, by 0.012 nats, with sets that share ten of
twelve directions. The stable fact across all three runs is not "causal beats
variance"; it is **norm-matched distillation toward any low-rank decision-state
target slows a from-scratch latent student's NLL learning, by 0.02–0.12 nats at
fixed steps**, and the selector moves that by an order of magnitude less.

### Why the student did not learn

48,000 examples (<1 epoch of a generated task) with fresh LoRA and projector from
the official embeddings and readout, at the §86 learning rate, is far below the
released recipe (385k GSM8k-Aug examples, many epochs). Answers are 2–3 digit
numbers, so exact match demands the whole number from one latent pass. The latent
task is genuinely slow to acquire from scratch; this was under-estimated in §89.

### Bounds and what would change the reading

- Not measured: the official checkpoint's own *latent-path* accuracy on the
  templated test. If it is high, the from-scratch failure is budget; if it is low,
  the task is hard for the latent path itself. Two minutes of GPU.
- Not tested: a warm-start (official CODI weights, no reinitialisation) fine-tuned on
  the templated task under the same three arms. That regime has non-trivial accuracy
  from step 0 and would measure the tax where it matters; §85 warns that
  norm-matching to a converged student can be inert, but the templated task is new
  to the student, so the term is not near zero here.
- The equal-pressure design is part of the finding: a weaker distillation weight may
  remove the tax entirely, which would make it a tuning artefact rather than a
  property of the targets.

Three training experiments (§83, §86–88, §89–90) have now returned the same shape:
distillation toward a low-rank subspace of the teacher's decision state does not
help a from-scratch CODI student at the budgets available, and variance-selected
targets are never better than intervention-selected ones. That negative result, with
the §40/§55/§58 mechanistic findings it rests on, is the write-up.

## 91. Preregistered diagnostic: official latent-path accuracy on the §89 test split

Contract `official_codi_templated_latent_diagnostic_v1`. No training. The unmodified
official checkpoint is scored on the same 2,000 generated test problems as §89/§90
(same generator, seed 20260924, same split order; tested byte-identical) by the
released latent path (question + BOT + 6 thoughts + forced cue) and by explicit-CoT
generation from the question alone, plus the latent path on the 256-row selection
split. Per-template and per-step-count breakdowns and latent/CoT agreement are
reported. Reading, fixed in advance on latent test exact match: ≥ 50% **BUDGET** (the
§90 floor was training budget; a warm-start design is the informative follow-up);
≤ 20% **TASK** (the latent path itself finds the task hard; no from-scratch design at
this budget could have reached it); otherwise **MIXED**. Cost ≈ 5 GPU minutes.

## 92. Completed §91: the official latent path solves 72.5% of the templated test — the §90 floor was budget

Run 2026-09-24 (code `e44ef82`, pin `bce8a2e`), ≈5 GPU minutes, no training.

| path | test (n = 2,000) | selection (n = 256) |
|---|---|---|
| official latent (BOT + 6 thoughts + forced cue) | **72.5%** | 67.6% |
| official explicit CoT (question alone) | 85.4% | — |

Agreement on the test split: both correct 1,397; CoT only 310; latent only 53.

By step count and template (latent / explicit CoT):

| group | latent | CoT | note |
|---|---|---|---|
| 2-step | 0.852 | 0.972 | |
| 3-step | 0.598 | 0.735 | |
| gain_lose (a+b−c) | 0.940 | 0.984 | |
| share_left (a·b+c) | 0.996 | 0.992 | latent ≥ CoT |
| hours_pay (a·(b+c)) | 0.920 | 0.944 | |
| savings (a·b+c−d) | 0.768 | 0.832 | |
| trips (a·b·c−d) | 0.588 | 0.656 | |
| boxes_broken (a·b−c) | 0.552 | 0.968 | largest latent gap, 2-step |
| classes ((a+b)·c+d) | 0.524 | 0.804 | |
| profit ((a+b)·c−d) | 0.512 | 0.648 | |

### Reading: BUDGET (preregistered threshold ≥ 50%)

A converged CODI latent path does this task at 72.5%, against 1% for the §90
students after 3,000 from-scratch steps. The §90 STOP was a training-budget failure,
not a task failure, and the §89 exact-match gates were never reachable at that budget.
The latent path's failures are structured and familiar: multiplication-heavy and
three-step templates; the sample latent errors copy a question number (`7*5−4` → 7)
or land within ±10 of the answer on three-step problems. `boxes_broken` (a·b−c) is the
odd one out, a two-step template where the latent path loses 42 points to CoT; it is
the only template whose first operation is a multiplication of two small numbers
followed by a subtraction, which fits the §55 picture of intermediate values being held
in odd latent slots less precisely for products.

### Consequence for the design

The from-scratch student regime is closed at this quota. The one informative design
left is a **warm start**: the official weights unchanged, fine-tuned on the templated
task under `none` / `variance` / `causal` for ≈1,000 steps, with final test exact match
as primary. It starts at 72.5% and the question becomes whether either norm-matched
subspace target helps or hurts a competent student's fine-tuning. §85 is the caveat
(norm-matched supervision of a converged student was inert on GSM8K); here the
distillation target is the explicit-CoT decision state, which differs from the latent
state even at the checkpoint, so the term is not zero at step 0. Cost ≈ 3 h for three
seeds; ceiling effects on the easy templates are likely, so the per-template breakdown
should be preregistered as the secondary outcome. Whether to spend the quota on it is
a separate decision; the negative result of §83/§86–§90 stands without it.

## 93. Standing correction: the selector comparison is inconclusive and must not be claimed

Across §87, §88 and §90 the causal-versus-variance NLL difference was +0.027
(CI excludes 0), +0.008 (covers 0) and −0.012 (excludes 0, opposite sign). Relevance
tied causal in both GSM8K seeds. Exact match never left the floor in any run. No
statement of the form "intervention- or relevance-selected targets beat
variance-selected targets" is supported by this repository's data, and any write-up
must present the comparison as inconclusive with that table.

What is supported (3/3 runs, tight intervals): norm-matched distillation of a
from-scratch CODI student toward any low-rank subspace of the teacher's decision
state raises answer NLL at fixed steps relative to answer cross-entropy alone.

A decisive selector test needs, simultaneously, a student trained to non-trivial
accuracy and a teacher on which the selectors diverge. The GSM8K runs had divergent
selectors but students at the floor; the templated run had a learnable task (§92:
official latent path 72.5%) but selectors sharing 10 of 12 directions. A warm start
from the official weights does not satisfy the first requirement in a useful way:
the target is the decision state CODI's own loss already matched, so the term starts
near converged (§85). The remaining path is a from-scratch student trained to
competence on a task with divergent selectors, a multi-day GPU budget outside this
project's quota. The negative result stands without it.

## 94. Preregistered: recovery test of distillation targets on the real checkpoint (60 GPU-hours, two accounts)

### Question

§93 states what a decisive selector test needs: a student in the measurable-accuracy
regime whose distillation term is far from converged, on a teacher where the
selectors diverge. From-scratch students cannot reach that regime at this quota
(§90, §92). This experiment gets there by **removing one piece of the official
checkpoint's competence and measuring which target restores it**: the projector, the
two-layer module that turns each latent thought into the next input embedding, is
re-initialised; the LoRA adapters, embeddings and readout are untouched. The student
keeps the arithmetic skill but cannot write its thoughts, so GSM8K accuracy falls from
≈43% toward the no-thought baseline, and the distance between its latent decision
state and the teacher's explicit-CoT decision state is large again. Fine-tuning then
recovers accuracy in the hundreds of steps, and the question becomes which target,
if any, recovers it faster and further. This is the continued-training regime of a
latent reasoner, not learning from scratch; the claim is scoped to it.

Contract `official_codi_recovery_subspace_distillation_v1`. Code path: the §86 module
(teacher decision-state PCA, selectors, norm-matched smooth-L1 distillation) with
three additions: `reset_projector` (projector only), `scale_lora_b` (second damage
mode), and a per-step record of the cosine between the answer-CE gradient and the
distillation gradient, plus an `auxiliary_multiplier` on the norm-matched term.

### Data and teacher side

GSM8k-Aug equation-only, same `DATA_SEED` 20260923 and the same `sample_splits` order
as §86, so fit / select / validate (2,048 each) are the §86 splits and the selectors
should reproduce §86's rank-12 sets; train = steps × 16 rows drawn next; selection =
256 held-out GSM8k-Aug rows for the learning curve. Test = the 1,319 GSM8K test
questions, read once per trained model. Rank rule as amended in §86 (gap ≥ 0.05,
causal retention ≥ 0.5, maximise gap × retention over {8, 12, 16}).

### Go/no-go, before any seed is spent (≈1 h)

1. **Headroom.** Projector-reset model's selection-split exact match ≤ 30%.
2. **Selector divergence.** At the chosen rank, causal and variance share at most 8 of
   12 PCs (Jaccard ≤ 0.5). The templated teacher failed this (§90: 10 of 12); if GSM8K
   fails it, the selector question is moot on this teacher and the run stops.
3. **Term not converged.** Mean distillation loss of the variance and causal targets on
   256 fit rows at the reset checkpoint ≥ 2× its value at the official checkpoint.
4. **Recovery pilot.** One `none` run, seed 0 (not counted), 2,000 steps, curve every
   100 on the selection split. Step budget for all arms = 1,000 if the pilot reaches
   30% by step 1,000, else 2,000 if by 2,000, else STOP. The pilot's curve is reported.

### Training

Projector re-initialised per seed (same generator as §86's `reinitialize_student`,
projector part only). AdamW, LoRA parameters 1e-4, projector 5e-4 (fresh module),
warmup 50, cosine to 0 over the step budget, weight decay 0.1, clip 2.0, batch 16 as
2 × 8 exact accumulation. Distillation: smooth-L1 on the selected teacher-PCA
coordinates of the student's latent decision state against the teacher's explicit-CoT
decision state, scaled by teacher coordinate std, gradient norm-matched to the CE
gradient. Data order and reset are functions of the seed, so arms are paired by seed.

### Stages and budget (≈30 min per run at 1.1 s/step for 1,000 steps + 1,319 generations)

| stage | damage | arms | seeds | runs | hours | account |
|---|---|---|---|---|---|---|
| 1 primary | projector reset | none, full, variance, relevance, causal, random | 1–5 | 30 | ≈15 | A: 1–3, B: 4–5 |
| 2 weight robustness | projector reset | variance ×0.3, variance ×3, causal ×0.3, causal ×3 | 1–3 | 12 | ≈6 | A |
| 3 second damage | LoRA-B × 0.5 | none, variance, causal, random | 1–3 | 12 | ≈6 | B |

≈27 h of the 60 available; the remainder is margin for failures and a 2,000-step
budget if the pilot requires it (which doubles stage costs to ≈50 h, still inside).

### Outcomes and gates (paired bootstrap over the 1,319 test questions, seed-mean correctness)

Primary: GSM8K test exact match at the final step.

- **R0** sanity: `none` final test EM ≥ 30% (recovery happened).
- **S1** `full − none` > 0 (does any distillation help recovery?).
- **H1** `causal − variance` lower bound > 0. **H2** `causal − relevance`. **H3** `causal − random`.
- `variance − none`, `causal − none` two-sided (does a target hurt recovery, as it hurt learning in §86–§90?).
- **Headline** = H1 ∧ H3 → `CONFIRMED`. If H1's interval covers 0 with width ≤ 3 points → `NULL: selector does not matter in the recovery regime`. Otherwise `PARTIAL` naming the survivors.
- Stage 2: the sign of `causal − variance` at ×0.3 and ×3; a headline that flips sign with the weight is reported as weight-dependent, not confirmed.
- Stage 3: the sign of `causal − variance` and of `variance − none` under LoRA-B damage.
- Secondary: steps to 30% selection EM; test NLL; mean gradient cosine (CE vs distillation) per arm, reported as the mechanism behind any tax.

### What this can and cannot say

It can say whether, for a latent reasoner that must recover a lost component under
continued training, an intervention-selected decision-state target transfers better,
worse or the same as a variance-selected one, with five seeds on the real task and
1,319 paired questions (minimum detectable difference ≈ 2 points). It cannot say
anything about learning from scratch; §93 stands for that regime. If go/no-go 2 fails
the selector claim is undecidable on this teacher and the write-up says so.

## 95. §94 go/no-go: three of four passed; the headroom threshold was mis-scaled (amendment)

First preliminary run 2026-09-24 (code `87f4e70`, pin `1d39f6a`), projector damage:

| check | measured | rule | result |
|---|---|---|---|
| selectors distinguishable | rank 12 chosen (gap 0.163, retention 0.543) | rank rule | pass |
| selector divergence | causal `[5,6,7,9,10,11,12,13,14,15,17,19]` vs variance `[0..11]`: 6 shared | ≤ 8 shared | pass |
| term not converged | causal 0.212 → 0.508 (2.4×), variance 0.160 → 0.330 (2.1×), full 0.062 → 0.157 | ≥ 2× | pass |
| headroom | official 0.820 → damaged **0.504** on the selection split | damaged ≤ 0.30 | **fail** |

On GSM8K the selectors diverge properly (unlike the templated teacher, §90): the causal
set skips PCs 0–4 entirely and reaches into PCs 12–19; relevance overlaps causal on 8
of 12 and adds two tail PCs (760, 767).

**Why the headroom check failed.** The 30% ceiling in §94 was written on the GSM8K-test
scale (official ≈ 43%) but is evaluated on the selection split, which is 256 held-out
GSM8k-Aug *training-distribution* rows where the official model scores 82%. A reset
projector removes 32 points there, which is ample room; the absolute ceiling was the
wrong instrument. Recorded as a scaling error in the preregistration, not a property
of the damage.

**Incidental finding worth keeping.** With the projector re-initialised, i.e. six
latent slots fed by a random LayerNormed MLP of the previous state, the LoRA-adapted
GPT-2 still answers 50% of training-distribution problems. Roughly 32 of the official
model's 82 points on this split depend on the thoughts being written correctly.

### Amendment (before any counted seed; nothing trained)

All recovery quantities are now relative to the measured gap `official − damaged` on
the selection split, computed in the go/no-go and stored in `preliminary.json`:

- **Headroom:** `official − damaged ≥ 0.20` (measured: 0.316).
- **Recovery threshold** `= damaged + 0.5 × (official − damaged)` (measured: ≈ 0.66).
  The pilot's step rule (1,000 / 2,000 / STOP) and the per-run steps-to-threshold
  secondary use this threshold instead of the absolute 30%.
- **R0:** the `none` arm's final selection-split exact match, averaged over seeds,
  reaches the recovery threshold.

The checks are recomputed from the stored measurements on every run, so the first
run's teacher cache and selectors are reused when its output is attached. Gates on the
1,319-question test set, arms, seeds, budgets and claims are unchanged.

### §95 addendum: go/no-go passed under the amended rule; pilot fixes the budget at 2,000 steps

Second preliminary run 2026-09-24 (code `60068d6`). All four checks pass (gap 0.316,
recovery threshold 0.662). Pilot (`none`, seed 0, 2,000 steps, projector reset):
selection-split exact match 0.27 at step 100, 0.43 at step 1,000, crosses the
threshold at **step 1,300** (0.685), plateaus at ≈0.72 from step 1,600 (official
0.82). GSM8K test exact match after 2,000 steps: **32.4%** (428/1,319; official ≈43%),
test NLL 1.61. Recovery is a phase transition between steps 1,000 and 1,300 rather
than a gradual climb. Step budget for all counted runs = **2,000** (≈50 min per run;
stages ≈25 h on account A and ≈21 h on B). Training begins from this output's teacher
cache, selectors and pilot.

## 96. §94 pilot is bimodal across accounts: projector reset replaced by calibrated projector noise (amendment)

The second account ran the identical preliminary (runner unchanged between `60068d6`
and `ae8ed41`; same seed 0, same reset, same data; go/no-go numbers identical to the
digit: 0.820 / 0.504, term ratios equal). Its `none` pilot did **not** recover:
selection-split exact match hovered at 0.30–0.37 for all 2,000 steps (0.36 at the
end) and GSM8K test exact match was **13.9%** (184/1,319), against 0.72 / 32.4% on the
first account.

| pilot (none, seed 0, 2,000 steps) | selection EM at 1,000 / 2,000 | test EM |
|---|---|---|
| account A | 0.43 / 0.72 | 32.4% |
| account B | 0.33 / 0.36 | 13.9% |

Same code, same seed, opposite outcome: the only difference is GPU numerics (kernel
nondeterminism, possibly a different GPU model). Recovery from a fully re-initialised
projector is a phase transition (§95 addendum) that can fail to fire within the
budget, and whether it fires is decided by noise below the seed. That makes the
projector-reset regime unusable as a primary: with five seeds, arm differences would
be swamped by which runs happened to transition. Recorded as an incidental finding
in its own right: a CODI student whose thought-writer is destroyed sits on a plateau
at ≈0.35 (train-distribution) / ≈0.14 (GSM8K) and escapes it stochastically.

### Amendment (before any counted seed; nothing trained)

- **Primary damage mode becomes `projector_noise`:** each projector `Linear` weight
  receives `W ← W + σ · std(W) · N(0, 1)` (seeded per run; biases and LayerNorm
  untouched). The model stays in its basin and recovery should be gradual. **σ is
  calibrated in the go/no-go**, not chosen by hand: over σ ∈ {0.25, 0.5, 1.0, 2.0}
  the smallest σ whose selection-split gap is ≥ 0.20 is taken (sweep recorded); if
  none qualifies the headroom check fails and the run stops.
- **Two pilot seeds** (0 and 100) instead of one. The step budget is 1,000 if *both*
  pilots cross the recovery threshold by step 1,000, 2,000 if both cross by 2,000,
  otherwise STOP. A pair that disagrees is itself a STOP, which is the bimodality
  check this section shows is necessary.
- Stage 2 (weights) uses the same `projector_noise` damage and σ; stage 3 keeps
  `lora_half`. `projector` (full reset) remains available as a non-default mode.
- Everything else (arms, seeds, learning rates, gates, claims, 1,319-question test)
  is unchanged.

Cost of the new preliminary: σ sweep ≈ 5 min plus two pilots ≈ 1.7 h, once, on one
account; the second account reuses the published output.

### §96 addendum: third reset pilot pair confirms the bimodality; noise mode not yet run

A further preliminary on 2026-09-25 ran the `034cd69` runner but with the previous
notebook's stage table, i.e. still `--damage projector`. Two pilots: seed 0 stalled
again (selection EM ≈ 0.34 throughout, test **13.8%**); seed 100 partly recovered
(test **28.7%**) without crossing the selection threshold. Across three seed-0 runs of
the reset regime the outcomes are 32.4%, 13.9%, 13.8%; the two-pilot rule stopped the
run as designed. The `projector_noise` mode has not yet been exercised; it requires
the notebook file at `034cd69`, whose stage table maps `primary` to `projector_noise`.

## 97. §96 go/no-go for `projector_noise`: σ calibrated at 2.0; term-ratio floor lowered to 1.5 (amendment)

First `projector_noise` preliminary, 2026-09-25 (code `034cd69`). Sigma sweep on the
selection split (official 0.820):

| σ | selection EM | gap |
|---|---|---|
| 0.25 | 0.816 | 0.004 |
| 0.5 | 0.813 | 0.008 |
| 1.0 | 0.793 | 0.027 |
| 2.0 | 0.531 | 0.289 |

The projector is robust to weight noise up to σ ≈ 1 and gives way sharply after; σ = 2.0
is the calibrated value (gap 0.289, recovery threshold 0.676). Selectors unchanged
(rank 12, 6 shared PCs). Term losses official → damaged: causal 0.212 → 0.439 (2.07×),
variance 0.160 → 0.295 (**1.84×**), relevance 0.183 → 0.376 (2.05×), random 0.113 →
0.241 (2.13×), full 0.062 → 0.135 (2.18×). Headroom, distinguishability and
divergence pass; `term_not_converged` fails on the variance ratio alone, 0.16 short of
the 2.0 floor.

**Amendment, made after seeing the value and recorded as such.** The 2.0 floor in §94
was set with no reference measurement; its purpose is to exclude the §85 situation
where the term is already at its floor (ratio ≈ 1). An 84% rise in the variance term
is not that situation. The floor becomes **1.5×** for the variance and causal targets.
No other rule changes. The two pilots (seeds 0 and 100) still have to agree before any
seed counts, which is the check that matters for this regime.

## 98. §97 pilots: GO at 1,000 steps; repair is fast, so a dense early secondary is added

Preliminary 2026-09-25 (code `9e78741`), `projector_noise` at σ = 2.0. All four checks
pass (term ratios: causal 2.07×, variance 1.84×, both above the 1.5 floor). Pilots:

| pilot (none) | selection EM at step 100 | at step 2,000 | GSM8K test EM | test NLL |
|---|---|---|---|---|
| seed 0 | ≈0.68 | ≈0.77 | 40.2% (530) | 1.545 |
| seed 100 | ≈0.74 | ≈0.78 | 39.2% (517) | 1.538 |

Both pilots crossed the recovery threshold (0.676) at step 100, the first measurement,
so the preregistered rule sets the budget at **1,000 steps**. The two pilots agree
closely, unlike the reset (§96): noise damage is not bimodal.

### Reading

- **Repair is fast.** Most of the 29 lost selection points return within 100 steps; the
  curve then creeps from ≈0.74 to ≈0.78 and plateaus below the official 0.82. On GSM8K
  test the repaired `none` student sits ≈3–4 points under the official 43.4%.
- **What the primary therefore measures.** Final GSM8K exact match after 1,000 steps
  compares arms on the plateau: whether a copying target lifts a lightly damaged
  student toward the teacher-consistent official solution, or pulls it further away.
  There is room in both directions (the `none` arm is 3–4 points under official and
  the MDE is ≈2 points), but a large positive effect is capped near the official
  accuracy. A NULL is therefore a plausible outcome and would be informative.
- **The `steps_to_threshold` secondary is uninformative**: every run will cross at the
  first curve point. The repair phase sits inside the first 100 steps and the curve
  (every 100 steps) cannot resolve it.

### Addition (secondary only; primary, arms, seeds, budget and gates unchanged)

Every 10 steps for the first 200, and once at step 0 immediately after damage, each run
records the teacher-forced answer NLL on the 256-row selection split (no generation) and
the student's distance from the teacher in **every** arm's subspace (causal, variance,
relevance, random, full) on 256 held-out fit rows. This makes the repair phase visible,
and makes distance-in-the-causal-directions comparable across arms, including arms not
trained on it. Cost ≈ 2 minutes per run. Reported as seed-mean curves, not gated.

Budget at 1,000 steps: ≈ 28–30 minutes per run; primary 30 runs ≈ 15 h (account A
seeds 1–3 ≈ 9 h, account B seeds 4–5 ≈ 6 h), weights 12 runs ≈ 6 h, `lora_half` 12 runs
plus its own preliminary ≈ 8 h.

## 99. Interim, §94 primary seeds 1–3 (account A): copying helps repair; the full state is best; causal beats variance only

Run 2026-09-25/26, code `b44e288`, `projector_noise` σ = 2.0, 1,000 steps, 18 runs of
≈26 min. Seeds 4–5 (account B) outstanding; this is **not** the preregistered read,
which requires all five seeds.

### GSM8K test exact match (1,319 questions)

| arm | seed 1 | seed 2 | seed 3 | mean | test NLL | mean grad cosine |
|---|---|---|---|---|---|---|
| none | 38.44 | 39.42 | 37.83 | 38.56 | 1.571 | — |
| full | 40.56 | 40.71 | 39.95 | **40.41** | 1.571 | 0.402 |
| causal | 40.56 | 40.18 | 39.04 | 39.93 | 1.600 | 0.309 |
| relevance | 39.95 | 40.33 | 38.67 | 39.65 | 1.622 | 0.325 |
| random | 38.82 | 40.11 | 38.67 | 39.20 | 1.476 | 0.162 |
| variance | 38.51 | 39.58 | 38.82 | 38.97 | 1.542 | 0.132 |

Official checkpoint 43.4%. Paired bootstrap over questions on seed-mean correctness:

| comparison | mean | 95% CI | per-seed differences |
|---|---|---|---|
| full − none (S1) | +1.84 | [+0.91, +2.75] | +2.12, +1.29, +2.12 |
| causal − none | +1.36 | [+0.53, +2.20] | +2.12, +0.76, +1.21 |
| causal − variance (H1) | +0.96 | [+0.13, +1.82] | +2.05, +0.61, +0.23 |
| causal − random (H3) | +0.73 | [−0.15, +1.62] | +1.74, +0.08, +0.38 |
| causal − relevance (H2) | +0.28 | [−0.51, +1.09] | +0.61, −0.15, +0.38 |
| variance − none | +0.40 | [−0.45, +1.29] | +0.08, +0.15, +0.99 |
| full − causal (not preregistered) | +0.48 | not computed | 0.00, +0.53, +0.91 |

Interim runner verdict: **PARTIAL** (H1 passes, H3 does not). R0 passed (none final
selection 0.772 against threshold 0.676).

### Reading (interim)

1. **The from-scratch sign reverses in repair.** In §86–§90 every copying target slowed
   learning. Here every copying target beats `none` in every one of the three seeds,
   by +0.4 (variance) to +1.8 points (full). Copying the teacher's decision state helps
   a competent student recover; it hurt a student that could not yet do the task.
2. **The full state is the strongest target**, ≥ causal in all three seeds. The strong
   form of the hypothesis, that copying only the answer-deciding directions beats
   copying everything, is not supported at this n.
3. **Among 12-direction targets, answer-directed beats loud.** Causal beats variance
   (H1, all seeds positive). Causal and relevance are tied, as in §88. Causal versus
   random is positive in every seed but its interval crosses zero.
4. **The benefit tracks gradient alignment.** Across the five copying arms, the mean
   cosine between the copying gradient and the answer gradient orders the accuracy
   gains almost exactly (Pearson 0.96, Spearman 0.90, n = 5 arms): full 0.40, relevance
   0.32, causal 0.31, random 0.16, variance 0.13. Variance-target gradients are the
   least aligned with the answer, as §94 predicted. Descriptive, five points.
5. **Teacher-forced NLL dissociates from exact match.** Random and variance have the
   lowest test NLL and near-lowest accuracy; causal and relevance raise NLL while
   raising accuracy. The primary is exact match; NLL is not a proxy for it here.
6. **Early curves.** The full target reaches the smallest distance from the teacher in
   both the causal and the variance directions. Training on the causal target *raises*
   the distance in the variance directions above the `none` arm; the random target
   raises both distances. Repair of selection NLL is similar across arms in the first
   200 steps, full slightly fastest.

### Caveats

- The bootstrap resamples questions, not seeds. Between-seed spread of `none` (1.6
  points) is comparable to the arm effects, so per-seed sign consistency is reported
  alongside and is the more conservative read with three seeds.
- Effects are about 1 point, below the ≈2-point planning MDE; seeds 4–5 decide H3.
- `full − causal` was not a preregistered comparison and is descriptive only.

## 100. Completed §94 primary, five seeds: PARTIAL. Copying helps repair; the full state is best; the benefit tracks gradient alignment

Seeds 4–5 (account B, code `b44e288`) completed 2026-09-26 and were pooled with seeds
1–3 (§99). Thirty runs, `projector_noise` σ = 2.0, 1,000 steps, one read of the 1,319
GSM8K test questions per run. This is the preregistered read.

### GSM8K test exact match

| arm | mean (5 seeds) | seed SD | gain over none | seeds with gain > 0 | test NLL | grad cosine |
|---|---|---|---|---|---|---|
| none | 38.48 | 0.64 | — | — | 1.578 | — |
| **full** | **40.23** | 0.57 | +1.74 | 5/5 | 1.578 | 0.400 |
| causal | 39.85 | 0.71 | +1.36 | 5/5 | 1.597 | 0.311 |
| relevance | 39.74 | 0.67 | +1.26 | 5/5 | 1.625 | 0.330 |
| random | 39.41 | 0.79 | +0.92 | 5/5 | 1.481 | 0.163 |
| variance | 39.15 | 0.59 | +0.67 | 5/5 | 1.542 | 0.128 |

Official checkpoint 43.4%. Paired bootstrap over questions on seed-mean correctness:

| gate | comparison | mean | 95% CI | seeds > 0 | result |
|---|---|---|---|---|---|
| R0 | none final selection EM ≥ 0.676 | 0.771 | — | — | pass |
| S1 | full − none | +1.74 | [+0.99, +2.52] | 5/5 | **pass** |
| H1 | causal − variance | +0.70 | [+0.05, +1.38] | 5/5 | **pass** |
| H2 | causal − relevance | +0.11 | [−0.52, +0.74] | 3/5 | fail |
| H3 | causal − random | +0.44 | [−0.26, +1.12] | 4/5 | fail |
| — | causal − none | +1.36 | [+0.64, +2.08] | 5/5 | positive |
| — | variance − none | +0.67 | [−0.09, +1.43] | 5/5 | covers 0 |
| — | variance − random | −0.26 | [−0.89, +0.39] | 1/5 | covers 0 |

**Verdict: PARTIAL** (H1 ∧ ¬H3), exactly as preregistered. Descriptive, not
preregistered: full − causal +0.38 (4/5 seeds), full − relevance +0.49 (4/5).

### What is established (five seeds, one model, recovery regime)

1. **Distillation toward the teacher's decision state helps a damaged competent
   student.** Every target beats plain fine-tuning in every seed. This is the sign
   reversal from §86–§90, where the same targets slowed a from-scratch student.
2. **The full decision state is the best target.** CODI's own choice. The strong
   hypothesis, that copying only the answer-deciding directions beats copying
   everything, is rejected at this budget.
3. **Variance selection is the worst target** and is not distinguishable from twelve
   random directions (variance − random −0.26, 1/5 seeds). The LoRi-style rule has no
   support here; §40's "inert loud directions" carries into distillation.
4. **Answer-directed selection beats variance selection** (H1, 5/5 seeds), and causal
   and relevance are tied (H2), for the reason given in §93's follow-up discussion: at
   the readout layer, intervention and gradient attribution select the same object.
   Causal does not clear random (H3), so the claim is "not variance", not "causal".
5. **The benefit of a target is predicted by the alignment of its gradient with the
   answer gradient.** Across the five copying arms, Pearson 0.97 / Spearman 0.90
   between mean cosine and accuracy gain (n = 5 arms, descriptive). Ordering: full
   0.40 > relevance 0.33 > causal 0.31 > random 0.16 > variance 0.13.
6. **Teacher-forced NLL dissociates from exact match** (random and variance have the
   lowest NLL and the smallest gains). NLL is not a usable proxy for this outcome.
7. **Early repair curves (10-step resolution, 5 seeds).** The full target reduces the
   student's distance from the teacher fastest in both the causal and the variance
   subspaces. Training on the causal target *raises* the distance in the variance
   directions above `none`; random raises both. Low-rank copying is local.

### Bounds

Effects are 0.7–1.7 points with a seed SD of 0.6–0.8, so the question-level intervals
understate seed-level uncertainty; the per-seed sign counts are the conservative read
and agree with the intervals on every gate. One model (CODI GPT-2), one task (GSM8K),
one damage mode, one distillation weight; stages 2 (weights) and 3 (LoRA-half) are the
robustness checks and remain to run. Final weights were not saved, so no out-of-
distribution read is possible for these thirty runs.

### Decision

> The decision-state selector question is answered for this checkpoint: full > answer-
> directed (causal ≈ relevance) > random ≈ variance, with a gradient-alignment
> mechanism. Run stage 2 (weights) to test whether the ordering survives ×0.3 and ×3
> pressure. For the second account, replace stage 3 (LoRA-half) with the trajectory-
> level test if a genuinely causal selector is wanted (§93 discussion); otherwise run
> stage 3 as planned. Add final-weight saving before either, so the remaining runs can
> be read on SVAMP / GSM-Hard / MultiArith later.

### §100 addendum: per-question analysis of the 30 runs — full wins by breaking less, not by repairing more

No GPU. `scripts/analyze_recovery_per_question.py` re-scores every run's stored test
outputs with `official_answers_match` (arm means reproduce the summary exactly) and
pairs each arm with the `none` run of the same seed (same damage, same data order).

**Questions by plain-training outcome across the five seeds:** always right 392
(29.7%), unstable 254 (19.3%), always wrong 673 (51.0%).

**Where each target's gain over `none` comes from** (points of test accuracy):

| arm | total | always right | unstable | always wrong |
|---|---|---|---|---|
| full | +1.74 | −0.30 | **+1.06** | +0.99 |
| causal | +1.36 | −0.36 | +0.64 | **+1.09** |
| relevance | +1.26 | −0.39 | +0.62 | +1.03 |
| random | +0.92 | −0.50 | +0.50 | +0.92 |
| variance | +0.67 | −0.59 | +0.42 | +0.83 |

**Paired flips per seed** (none wrong → arm right = repair; none right → arm wrong = break):

| arm | repairs | breaks | net |
|---|---|---|---|
| full | 59.6 | **36.6** | +23.0 |
| causal | 59.0 | 41.0 | +18.0 |
| relevance | 57.0 | 40.4 | +16.6 |
| random | 53.8 | 41.6 | +12.2 |
| variance | 50.8 | 42.0 | +8.8 |

### Reading

1. **Full and causal repair the same number of questions (59.6 vs 59.0 per seed).** On the
   673 questions plain training never solves, causal's gain (+1.09) matches full's
   (+0.99). The answer-directed 12 directions carry the *teaching* signal as well as
   the whole state does.
2. **Full's whole advantage is fewer breaks** (36.6 vs 41.0 per seed) and it lands on
   the 254 *unstable* questions (+1.06 vs +0.64). Copying the remaining 756 directions
   does not teach; it anchors, keeping marginal questions from drifting during repair.
   This matches the early curves (§100 item 7): training on the causal target lets the
   student drift away from the teacher in the directions it is not trained on, and the
   drift costs borderline questions.
3. **The repair sets are not nested.** Only 59–63% of any low-rank target's repairs are
   also repaired by full in the same seed; each arm repairs 19–24 questions per seed
   that full does not, and full repairs 24–27 that the arm does not. Overlaps are far
   above independence (Jaccard 0.35–0.46 vs 0.03–0.04 expected) but no target is a
   subset of another. There is no distinct set that "only full can fix": full's
   seed-mean gain exceeds every low-rank target's by ≥ 0.4 on just 5 of 1,319 questions.
4. **Variance and random are interchangeable.** Their repair sets overlap no more with
   each other (Jaccard 0.37) than with the other arms, they repair the fewest questions
   and break the most, and their gains on always-wrong questions are the smallest.
5. **Most flips are seed-specific.** Only 14–22 questions per arm are repaired in ≥ 3 of
   5 seeds; ~78% of per-seed repairs are one-off. Two `none` runs with different seeds
   disagree on 120 questions; an arm and its same-seed `none` disagree on 93–100. The
   damage-and-data seed moves more questions than the target does.

### Consequence for the claim and for stage 2

The mechanism splits into two components: an **answer-directed component** (the 12
causal/relevance directions), which repairs the never-solved questions as well as the
full state, and an **anchoring component** (everything else), which prevents breaks on
marginal questions. Variance directions provide neither. This predicts that the ×3
copying-pressure arms in stage 2 will reduce breaks for causal (stronger anchoring in
its 12 directions) without adding repairs, and that ×0.3 will lose anchoring first.
It also motivates a two-term target (answer-directed at full pressure plus a weak
full-state anchor) as the natural follow-up if stage 2 behaves as predicted; that
would be preregistered separately.

### §100 addendum 2: relevance comparisons not gated by the runner (paired bootstrap, five seeds)

| comparison | mean | 95% CI | seeds > 0 |
|---|---|---|---|
| relevance − variance | +0.59 | [−0.11, +1.30] | 4/5 |
| relevance − random | +0.33 | [−0.30, +0.99] | 3/5 |
| full − relevance | +0.49 | [−0.14, +1.14] | 4/5 |
| full − causal | +0.38 | [−0.26, +1.02] | 4/5 |

Only `causal − variance` (+0.70, [+0.05, +1.38], 5/5) clears zero among the
subspace-selector contrasts. The gradient-scored (relevance) selector does not
separately clear variance or random at five seeds. Any statement that "relevance
beats variance" or "relevance beats causal" is unsupported; the supported statement
is that the two answer-directed selectors are indistinguishable from each other and
that the intervention-selected one, and only it, clears variance.
