# Official CODI boundary-aware selector confirmation

## Motivation

The matched selector-specificity experiment found that R-KV was much stronger than
random selection but did not beat uniform selection overall. Its exploratory
position-resolved results showed:

- uniform selection was substantially stronger at the first aligned position
- R-KV was often stronger at middle positions
- uniform selection regained an advantage at the final aligned position

This follow-up tests that pattern on new data rather than tuning and evaluating on the
same 5,000 examples.

## Candidate selector

The boundary-aware R-KV selector:

1. always retains the first valid explicit-CoT teacher trace token
2. always retains the last valid teacher trace token
3. uses the unchanged R-KV score to fill the remaining four slots
4. sorts all six selected indices chronologically before aligning them with the six
   official CODI latent positions

If a trace contains fewer than six valid tokens, every valid token is retained and
missing targets use the existing mask contract.

## Independent confirmation sample

The completed seed-1 selector manifest is a required input:

```text
MyDrive/CODI_KAVA/outputs/official_codi_selector_specificity/
  n5000_seed1/collection_manifest.json
```

All 5,000 indices in that manifest are excluded first. The remaining eligible examples
are permuted with data seed 2, and the first 5,000 are selected. The collector records:

- the prior manifest SHA-256
- the excluded-index SHA-256
- the new sample-index SHA-256
- the excluded count
- a verified overlap count of zero

The audit and complete collection both stop if the prior manifest is incomplete, uses a
different checkpoint or dataset, contains duplicate indices, or overlaps the new sample.

## Matched controls

The same forward pass accumulates:

- `boundary_rkv`, the candidate
- unchanged `rkv`
- unchanged `uniform`
- random selectors with seeds 101, 211, 307, and 401

Every selector shares the same disjoint examples, two split halves, student latent
states, and four within-batch teacher-example derangements.

## Primary estimand and gate

For every layer, head, latent position, and KV kind:

```text
selector signal R² =
    held-out rank-4 R²(actual teacher–student pairing)
    minus
    held-out rank-4 R²(within-selector shuffled pairing)
```

The boundary-aware selector passes separately for keys and values only if:

1. its existing Stage 1c actual-versus-shuffle gate passes
2. its median signal advantage over unchanged R-KV is at least 0.01 R-squared
3. it beats unchanged R-KV in at least 60 percent of matched groups
4. its median signal advantage over uniform is at least 0.01 R-squared
5. it beats uniform in at least 60 percent of matched groups
6. its median signal advantage over the per-group random median is at least 0.01
   R-squared
7. it beats the random median in at least 60 percent of matched groups

The overall status reports support for both KV kinds, one kind, or neither. This gate is
stricter than simply beating random selection because the candidate was motivated by
the R-KV-versus-uniform crossover.

## Run on Colab A100

Use
[`notebooks/colab_official_codi_boundary_selector.ipynb`](../notebooks/colab_official_codi_boundary_selector.ipynb).

The notebook:

1. mounts Drive and checks out a pinned repository commit
2. runs the selector, compression, and official-CODI regression tests
3. verifies the official full-GSM8K accuracy gate
4. verifies the completed prior selector collection
5. audits boundary retention and zero sample overlap
6. collects the disjoint 5,000-example matched moments
7. runs the CPU reduced-rank and candidate-specificity analyses
8. displays and persists the compact result

The approximately 2.7 GB seven-selector collection is atomically checkpointed every
1,000 examples. Rerun the collection cell with unchanged parameters after a disconnect.

## Durable outputs

```text
MyDrive/CODI_KAVA/
  outputs/official_codi_boundary_selector/
    audit_seed2/
    n5000_seed2/
      selection_audit.json
      selector_statistics.pt
      collection_manifest.json
  reports/official_codi_boundary_selector/
    official_codi_n5000_seed2_boundary_selector.json
    official_codi_n5000_seed2_boundary_selector.md
    official_codi_n5000_seed2_boundary_selector_details/
  logs/official_codi_boundary_selector/
```

## Decision

A positive gate supports a compute-matched downstream distillation experiment with the
boundary-aware selector. A negative gate means the exploratory position crossover did
not generalize into a globally stronger selector. In that case, do not start expensive
projection training based on this selector.
