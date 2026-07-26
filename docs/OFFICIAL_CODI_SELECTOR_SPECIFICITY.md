# Official CODI teacher-trace selector specificity

## Research question

The official CODI checkpoint contains reproducible position-conditioned low-rank
teacher–student KV correspondence. The next question is whether this signal depends on
which explicit-CoT teacher tokens are selected.

This experiment asks whether R-KV selects teacher trace positions whose key/value
information is more predictably aligned with the six latent student states than:

- six chronologically uniform trace positions
- four independently seeded random selections per layer and head

All selectors are observational analysis-time interventions. CODI was not trained with
R-KV, uniform, random, or KV-distillation targets.

## Fixed matched contract

The primary run uses:

- the accuracy-gated public `zen-E/CODI-gpt2` checkpoint
- 5,000 official-eligible training examples sampled with data seed 1
- the same two split halves for every selector
- six teacher targets and six student latent positions
- rank-four cross-validated student-to-teacher prediction
- four within-batch teacher-example derangements
- random selector seeds 101, 211, 307, and 401

The collector makes one teacher and one student forward pass per batch. R-KV, uniform,
and all random arms consume those same tensors. The split assignment and shuffled
permutations are also shared. Selector differences therefore cannot come from different
examples, data order, model execution, or null construction.

## Primary estimand

For every layer, head, latent position, selector, and KV kind, compute:

```text
selector signal R² =
    held-out rank-4 R² with correct teacher–student pairing
    minus
    held-out rank-4 R² with within-selector shuffled pairing
```

Raw held-out R-squared is secondary. Subtracting each selector's own shuffled null
controls for predictable marginal structure that does not depend on example pairing.

R-KV is compared with uniform selection and with the per-group median across the four
random selectors.

## Predefined gate

R-KV selector specificity is supported separately for keys and values only if:

1. R-KV passes the existing Stage 1c actual-versus-shuffle gate for that KV kind
2. the median R-KV signal advantage over uniform selection is at least 0.01 R-squared
3. R-KV beats uniform selection in at least 60 percent of matched groups
4. the median R-KV signal advantage over the per-group random median is at least 0.01
   R-squared
5. R-KV beats the random median in at least 60 percent of matched groups

The overall result states whether the gate passes for keys and values, one KV kind, or
neither.

## Run on Colab A100

Use
[`notebooks/colab_official_codi_selector_specificity.ipynb`](../notebooks/colab_official_codi_selector_specificity.ipynb).
The official full-GSM8K reproduction gate must already be present in Drive.

The notebook:

1. mounts Drive and checks out a pinned repository commit
2. runs all relevant regression tests
3. verifies the passed official CODI accuracy gate
4. audits selector indices, shapes, masks, and agreement on one batch
5. collects all selectors together over 5,000 examples
6. runs the CPU reduced-rank and selector-specificity analyses
7. displays and persists the compact Markdown result

Collection state is saved atomically every 1,000 examples. If Colab disconnects, rerun
the collection cell with the same parameters to resume.

## Durable outputs

```text
MyDrive/CODI_KAVA/
  outputs/official_codi_selector_specificity/
    audit_seed1/
    n5000_seed1/
      selection_audit.json
      selector_statistics.pt
      collection_manifest.json
  reports/official_codi_selector_specificity/
    official_codi_n5000_seed1_selector_specificity.json
    official_codi_n5000_seed1_selector_specificity.md
    official_codi_n5000_seed1_selector_specificity_details/
      rkv_reduced_rank.json
      uniform_reduced_rank.json
      random_seed*_reduced_rank.json
  logs/official_codi_selector_specificity/
```

## Interpretation and decision

A positive gate means R-KV identifies teacher trace positions with more transferable
linear KV signal than the matched controls. It still does not establish answer causality
or improved task accuracy. The next experiment would then be a compute-matched downstream
comparison of full targets, learned rank-four targets, and random rank-four targets.

A negative gate means the stable low-rank correspondence is not specific to R-KV token
selection under this protocol. In that case, the evidence would point toward the
spectral information directions, position conditioning, or generic trace coverage rather
than R-KV selection itself.
