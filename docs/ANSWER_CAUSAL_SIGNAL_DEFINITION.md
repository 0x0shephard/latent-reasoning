# Operational definition of signal in teacher and student KV trajectories

Status: preregistration draft for the next mechanistic gate  
Checkpoint scope: author-released CODI GPT-2 at revision `fd641b3d3edc`  
Primary task: GSM8K mathematical reasoning  
Last updated: 2026-07-26

## 1. Decision

The project will not use the unqualified word **signal** for any direction that is
merely high-variance, low-rank, split-stable, or teacher-predictive.

For this project:

> **Answer-causal KV signal is information in a teacher or student KV component whose
> matched intervention produces a reproducible change in gold-answer probability on
> held-out examples, beyond an intervention-matched null.**

For a teacher component to become a **candidate distillation target**, it must satisfy
three separate requirements:

1. **Answer relevance**  
   Changing the information must change the model's gold-answer score under a valid
   causal intervention.

2. **Student accessibility**  
   A student latent state must be able to recover or align with that information on
   examples not used to fit the mapping.

3. **Positive marginal distillation utility**  
   Adding the target's matched supervision update must lower held-out answer loss more
   than omitting it or replacing it with a matched control target.

The final and strongest label is reserved for a downstream experiment:

> **Transferable supervision signal is an answer-causal, student-accessible teacher
> target that improves a student under a fixed-data, fixed-compute, fixed-target-budget
> training comparison.**

Answer causality is therefore a screening gate for training. It is not itself proof of
transfer.

## 2. Why this definition is necessary

The completed spectral experiments established that position-conditioned rank-four KV
structure can be stable and can predict aligned teacher representations on held-out
examples. The official-checkpoint causal intervention then showed that these learned
directions were not more necessary or sufficient for correct GSM8K answers than
energy-matched random directions.

The failed inference was:

```text
stable + low-rank + teacher-predictive
    therefore
task-relevant and worth distilling
```

The corrected logic is:

```text
stable representation structure
    is evidence of structural signal only

answer-relevant under held-out intervention
    is evidence of answer-causal signal

answer-causal + student-accessible
    defines a mechanistically plausible target

positive marginal distillation utility
    defines an optimization-aligned candidate target

student improvement under matched training
    establishes transferable supervision signal
```

## 3. Vocabulary that must remain separate

| Term | Operational meaning | What establishes it |
| --- | --- | --- |
| Structural signal | Reproducible variation in KV space | Split stability beyond isotropic noise |
| Compressible signal | Variation captured by a low-rank representation | Held-out rank/full retention |
| Paired predictive signal | Student state predicts the paired teacher target | Held-out prediction beyond shuffled pairing |
| Answer-associated signal | Component score correlates with answer score or correctness | Held-out association only |
| Answer-causal signal | Intervention changes gold-answer score beyond matched nulls | Held-out exact intervention |
| Student-accessible signal | Student can recover the teacher component on untouched examples | Cross-validated teacher–student mapping |
| Optimization-aligned target | Its matched update lowers held-out answer loss | With-versus-without target update |
| Candidate supervision target | Answer-causal, student-accessible, and optimization-aligned | Conjunction of the preceding gates |
| Transferable supervision signal | Target improves a newly trained student | Compute-matched downstream training |

The word **reasoning signal** should be avoided unless the experiment additionally
links the component to a defined intermediate computation, such as an operand,
operation, or intermediate result. Answer causality alone does not prove that a
human-interpretable reasoning algorithm is represented.

## 4. What counts as noise

Noise is defined relative to the intended use.

### 4.1 Representational noise

Variation that is unstable across data splits and no better than an energy-matched
random or shuffled-pairing null.

### 4.2 Causal noise

Variation that may be stable or predictive but does not affect the gold-answer score
more than a matched control.

This is the category into which the previously tested rank-four predictive directions
currently fall.

### 4.3 Supervision noise

Teacher information that is answer-relevant but inaccessible to the student at the
available latent capacity, or information whose supervision does not improve the
student under matched training.

Distillation loss magnitude alone does not identify supervision noise:

- high target loss can be a useful learnable gap or an inaccessible noisy target
- low target loss can mean the information is already matched, trivial, or redundant
- deleting a non-negative loss term always lowers the reported total objective
  mechanically

The relevant quantity is whether the target's **optimization update lowers held-out
answer loss**, not whether the target contributes a large or small number to the
distillation objective.

### 4.4 Redundant signal is not automatically noise

A component may have a small removal effect because the model stores the same
information in several heads, layers, positions, or directions. Therefore:

- a failed single-component necessity test does not prove absence of signal
- joint and patching interventions are needed when redundancy is plausible
- retaining a small subspace tests sufficiency but may create an off-manifold state

Every negative conclusion must name the tested intervention and scope.

## 5. Unit of analysis

A candidate component is indexed by:

```text
model path × KV kind × layer × head × latent position × direction or token target
```

The primary analysis keeps the following separate:

- teacher and student paths
- keys and values
- layers and attention heads
- latent positions 0 through 5
- token-position selection and within-vector direction selection

Pooling is allowed only as a prespecified summary after position-resolved effects have
been retained. Earlier experiments showed that pooling latent positions can hide
distinct relationships.

## 6. Outcome definitions

### 6.1 Screening outcome

For question `q_i`, canonical numeric answer `a_i`, and latent KV state `z_i`, use the
official answer sequence:

```text
The answer is: <canonical a_i><EOS>
```

`The answer is:` is teacher-forced context, but its fixed cue tokens are masked from the
loss. The differentiable screening score is the mean log-probability of only the
example-specific numeric answer tokens and EOS:

```text
s_i(z_i) =
    (1 / number_of_target_tokens)
    × sum_j log p(numeric_answer_token_j | q_i, z_i, answer_cue,
                  preceding_answer_tokens)
```

The target is scored after the official `<EOT>` transition and six latent iterations,
using the same tokenizer, prompt normalization, checkpoint, and cache path as the
passed official CODI evaluation.

Mean rather than summed log-probability avoids systematically favoring shorter numeric
strings. Masking the answer cue prevents generic formatting confidence from dominating
the target ranking. This score is a sensitive screening instrument. It is not the final
task metric and must not be reported as accuracy.

### 6.2 Specificity outcome

The same score is recomputed after shuffling gold answers across questions while
preserving answer-token-length bins. A genuine answer-conditioned method should lose
its advantage when the question–answer relationship is broken.

This control distinguishes answer-specific attribution from generic confidence,
formatting, or output-entropy effects.

### 6.3 Confirmatory task outcome

The final inference-time endpoint remains:

```text
numeric exact match under the released CODI greedy-generation protocol
```

Full GSM8K exact match is the confirmatory endpoint before any new distillation
training is justified.

## 7. Causal estimands

Let `I_c` be a prespecified intervention on candidate component `c`, and let `I_r` be
an intervention-matched random control with the same rank, location, expected energy,
and intervention type.

### 7.1 Necessity

```text
necessity_i(c) =
    s_i(unchanged) - s_i(remove c)
```

The specificity contrast is:

```text
necessity_i(c) - necessity_i(random matched c)
```

A positive value means removing the candidate is more harmful to the gold answer than
removing matched random information.

### 7.2 Sufficiency

```text
sufficiency_i(c) =
    s_i(retain c) - s_i(retain random matched c)
```

A positive value means the candidate preserves more answer-relevant function than the
matched random component at the same information budget.

### 7.3 Patching efficacy

When a semantically controlled source example is available:

```text
patch_effect_i(c) =
    s_i(counterfactual patch at c) - s_i(unchanged)
```

The expected effect must follow the prespecified counterfactual answer, not merely
decrease confidence. Activation patching is preferred for the final causal validation
because it can keep the intervention closer to the model's activation distribution.

### 7.4 Transfer estimand

After a causal gate passes:

```text
transfer_effect =
    accuracy(student trained with candidate targets)
    - accuracy(student trained with matched control targets)
```

The comparison must fix initialization, data order, optimizer steps, target count,
rank, loss scale, wall-clock class, decoding, and evaluation.

## 8. Marginal distillation utility

Answer causality and distillation usefulness are related but different.

- A current student component can be causal even when pulling it toward a particular
  teacher target is harmful.
- A teacher target can provide a useful learning scaffold even when its corresponding
  feature is not yet used by the current student.
- A perfectly predictable target may add no marginal learning value because the
  student already matches it.

Therefore, the project will categorize candidate targets by the effect of their
training update on answer loss before applying another spectral denoising method.

### 8.1 What must not be measured

The following comparison is invalid:

```text
total loss with target - total loss after deleting target
```

Deleting a non-negative term necessarily makes the reported objective smaller. This
does not show that the model has become better.

At the frozen checkpoint, merely disabling a distillation term also cannot change the
forward answer loss until an optimization update is taken.

### 8.2 First-order utility screen

Let:

```text
L_answer = gold-answer negative log-likelihood
L_c      = distillation loss for candidate target group c

g_answer = gradient of L_answer
g_c      = gradient of L_c
```

For a small update using the candidate target:

```text
theta_new = theta - learning_rate × g_c
```

the first-order change in answer loss is:

```text
change in L_answer ≈
    -learning_rate × dot(g_answer, g_c)
```

Consequently:

- positive gradient alignment predicts that distilling `c` lowers answer loss
- negative alignment predicts interference with the answer objective
- near-zero alignment predicts little immediate marginal utility

Cosine alignment is reported alongside the dot product so target groups with larger
raw gradients are not automatically ranked as more useful.

Gradients are restricted to the parameters that the intended student training would
actually update. For the official CODI GPT-2 setting, the trainable parameter contract
must be frozen before screening.

### 8.3 Stronger one-step counterfactual

Gradient alignment is a screening statistic. The stronger test performs a functional
one-step update without overwriting the checkpoint:

```text
theta_with_c =
    theta - learning_rate × normalized_gradient(base_loss + weight × L_c)

theta_without_c =
    theta - learning_rate × normalized_gradient(base_loss + matched_control_loss)

utility(c) =
    L_answer(theta_without_c; heldout_batch)
    - L_answer(theta_with_c; heldout_batch)
```

Positive utility means the candidate target produces a better held-out answer loss
after the matched update. Update norm, loss weight, batch, parameter set, and learning
rate must be identical. The target gradient is computed on a training batch and answer
loss is evaluated on a disjoint validation batch.

This with-versus-without comparison can be implemented with functional parameter
updates, so the official checkpoint remains unchanged.

### 8.4 Hierarchical target inventory

The first screen should not test every scalar KV dimension independently. Candidate
targets are categorized hierarchically:

1. key versus value supervision
2. latent position 0 through 5
3. early, middle, and late layer bands
4. individual layers within useful bands
5. heads within useful layers
6. directions within useful key/value, position, layer, and head groups

This answers **where useful supervision enters the optimization** before asking a TSV,
SAE, or other decomposition to decide **which directions to retain**.

Every refinement is selected on discovery data and frozen before the next held-out
level. Unselected branches are not silently revisited after seeing confirmation
results.

### 8.5 Target-utility labels

| Held-out answer-loss effect | Gradient alignment | Label |
| --- | --- | --- |
| Improves | Positive | Helpful target family |
| No reproducible change | Near zero | Redundant or neutral target |
| Worsens | Negative | Interfering target family |
| Improves briefly but not after a matched short run | Positive initially | Transient optimization scaffold |
| Cannot be matched by the student | Any | Inaccessible target |

These are optimization labels. They do not replace the answer-causality or
student-accessibility classifications.

## 9. Estimation is not the definition

The definition of signal is independent of the method used to find it.

Candidate estimators may include:

- gradient-times-activation screening
- an answer-conditioned or Fisher-style spectral decomposition
- AtP-style approximations to activation patching
- distributed alignment methods
- sparse dictionary or autoencoder features

None of these methods establishes signal by itself. Each only proposes candidates that
must pass the same held-out intervention contract.

This avoids defining signal circularly as whatever a preferred algorithm happens to
return.

## 10. Proposed discovery and confirmation boundary

### 10.1 Discovery data

Use augmented training examples for candidate discovery. Group examples by normalized
base question before splitting so augmented variants of the same problem cannot appear
on both sides of a validation boundary.

Candidate ranks, positions, layers, heads, and thresholds are selected only here.

### 10.2 Internal held-out validation

Use a disjoint group-held-out subset to:

- compare answer-conditioned candidates with variance-based TSV directions
- compare with energy-matched random directions
- test shuffled-answer specificity
- verify effect direction across independent splits

### 10.3 Final confirmation

Freeze the candidate-generation procedure before using the 1,319-example GSM8K test
set. The final set must not be used to choose rank, layer, head, position, or
intervention strength.

The unchanged official checkpoint must continue to reproduce 576 of 1,319 correct
answers before intervention results are accepted.

## 11. Required controls

Every proposed signal must be compared with:

1. **Energy-matched random subspace**  
   Tests whether any direction of the same rank and scale has the same effect.

2. **Variance-based spectral subspace**  
   Tests whether answer conditioning adds value beyond the completed TSV-style method.

3. **Shuffled-answer estimator**  
   Preserves activation statistics while destroying answer alignment.

4. **Location-matched control**  
   Uses the same layer, head, KV kind, and latent position.

5. **Unchanged baseline**  
   Revalidates official CODI accuracy and detects global intervention damage.

6. **Key-only and value-only arms**  
   Prevents a joint K/V intervention from hiding different causal roles.

R-KV and uniform token selectors are included when the candidate is a teacher-token
selector rather than a direction selector.

## 12. Gate for calling a component answer-causal

The exact numerical margins and primary component family must be frozen before running
the confirmatory set. At minimum, a component or fixed-budget component set is called
answer-causal only when all of the following hold:

1. Candidate construction uses no confirmatory examples.
2. The exact held-out intervention has the predicted direction.
3. Its candidate-minus-matched-random paired 95 percent interval excludes zero.
4. The result survives the prespecified multiple-testing correction.
5. The effect direction replicates across two disjoint discovery or validation splits.
6. The shuffled-answer version does not pass the same gate.
7. The unchanged checkpoint remains inside the official GSM8K reproduction gate.

Passing this gate supports **answer-causal signal under the tested intervention**.
It does not yet support the label **transferable supervision signal**.

## 13. Gate for beginning distillation training

Expensive training begins only if:

1. an answer-conditioned component passes the answer-causal gate
2. it outperforms the completed variance-based rank-four method
3. it is student-accessible beyond shuffled pairing on held-out examples
4. its target update has positive held-out marginal distillation utility
5. its construction can be frozen into a deterministic teacher-target procedure
6. the complete compute-matched training comparison is specified before training

If answer-causal structure exists but is not student-accessible, it is scientifically
interesting but is not a suitable distillation target for the current student.

## 14. Classification of future findings

| Answer-causal | Student-accessible | Marginal utility | Matched training benefit | Classification |
| --- | --- | --- | --- | --- |
| No | No or yes | Neutral | Not tested | Structural or predictive nuisance |
| Yes | No | Any | Not tested | Task-relevant but currently inaccessible |
| No | Yes | Positive | Not tested | Optimization scaffold without established causal content |
| Yes | Yes | Negative | Not tested | Causal target that interferes with learning |
| Yes | Yes | Positive | Not tested | Candidate supervision target |
| Yes | Yes | Positive | No | Causal but non-transferable under tested training |
| Yes | Yes | Positive | Yes | Transferable supervision signal |

This table is the project's meaning of “categorize what signal is.” It is an empirical
classification, not a survey taxonomy.

## 15. Immediate next research questions

The first question is now:

> Under matched update magnitude and loss weight, which teacher KV target families
> produce a reproducible reduction in held-out mathematical answer loss when their
> distillation gradient is included?

Only within target families that show positive marginal utility:

> Under a fixed rank-four and location-matched budget, do answer-conditioned student KV
> directions have greater held-out causal effects on mathematical answer probability
> than variance-derived TSV directions and energy-matched random directions?

If that gate passes, the teacher-target question becomes:

> Among answer-causal components, which teacher KV targets are accessible to the
> student, and do they improve distillation under a fixed six-target training budget?

The questions are deliberately sequential. The first does not require model training.
The second is not attempted unless the first produces a valid target.

## 16. Interpretation limits

- Signal is task-relative and checkpoint-relative.
- GSM8K answer causality does not imply general reasoning causality.
- Gold-answer likelihood is a screening score, not accuracy.
- A negative linear rank-four result does not exclude higher-rank, nonlinear, sparse,
  or redundantly distributed signal.
- A positive causal result does not imply human interpretability.
- Lower raw distillation loss does not imply a more useful target.
- Positive one-step utility does not guarantee a long-run training improvement.
- Only matched downstream training can establish transfer benefit.

## 17. Method references

- Task Singular Vectors: <https://arxiv.org/abs/2412.00081>
- AtP and AtP*: <https://arxiv.org/abs/2403.00745>
- Distributed Alignment Search: <https://arxiv.org/abs/2303.02536>
