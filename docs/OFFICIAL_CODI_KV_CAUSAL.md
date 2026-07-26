# Official CODI KV spectral causality

## Question

The earlier official-checkpoint analysis established that position-conditioned
rank-four student KV directions predict aligned teacher KV targets on held-out
calibration examples. It did not establish that CODI uses those directions to produce
correct answers.

This experiment asks a narrower causal question.

> Do the learned rank-four student KV directions have more causal value for official
> CODI answer accuracy than energy-matched random rank-four directions?

No new model is trained. The author-released CODI GPT-2 checkpoint remains frozen.

## Scientific inputs

The experiment requires:

1. the passed 1,319-example official CODI GSM8K reproduction summary
2. the completed 5,000-example seed-1 official CODI cross-moment collection
3. the pinned checkpoint and source revisions in
   `configs/official_codi_gpt2.yaml`

The compact intervention artifact is fitted from:

```text
MyDrive/CODI_KAVA/outputs/official_codi_kv_subspaces/
  n5000_seed1/statistics.pt
```

For every layer, attention head, latent position, and KV kind, the exporter fits the
rank-four student-to-teacher reduced-rank map and takes its leading four left singular
vectors. These are directions in raw student KV feature space. Calibration student
means are saved for centered interventions.

## Interventions

At each selected latent position, the new key and value entry is edited immediately
after it is appended to the cache. Later latent steps and answer decoding therefore
consume the intervened cache.

Let `x` be a student key or value vector, `mu` its calibration mean, and `P` the learned
rank-four projector.

```text
retain learned = mu + P(x - mu)
remove learned = mu + (I - P)(x - mu)
```

The random control uses deterministic groupwise random orthonormal rank-four bases.
Its projected component is scaled so that its expected squared energy on the
calibration covariance equals the learned projection energy. The same random seed and
artifact are used for every evaluation arm.

The complete design contains:

- one unchanged baseline
- learned retain and remove at each position 0 through 5
- energy-matched random retain and remove at each position 0 through 5
- the same four interventions at all six positions simultaneously

This is 29 full-GSM8K conditions. Position 4 and position 5 run first and form the
primary hypothesis family. Other positions and the all-position conditions are
secondary diagnostics.

## Primary estimands and gate

Retain is a sufficiency test:

```text
accuracy(retain learned) - accuracy(retain random)
```

Remove is a necessity test:

```text
accuracy(remove learned) - accuracy(remove random)
```

Useful learned directions predict a positive retain contrast and a negative remove
contrast. Every contrast is paired by exact GSM8K question and gold answer.

The four primary comparisons are retain and remove at positions 4 and 5. A primary
test is supported only when:

1. the learned-minus-random effect has the predicted direction
2. its 95 percent paired-bootstrap interval excludes zero
3. its exact McNemar p-value remains below 0.05 after Holm correction across the four
   primary tests

The overall gate is positive when at least one primary test passes this complete
contract. Every baseline effect and secondary position is reported regardless of the
gate.

## Run on Colab A100

Use
[`notebooks/colab_official_codi_kv_causal.ipynb`](../notebooks/colab_official_codi_kv_causal.ipynb).

The notebook:

1. mounts Drive and checks out a pinned repository commit
2. runs the intervention and analysis regression tests
3. verifies the official full-GSM8K reproduction gate
4. verifies and exports the 5,000-example rank-four intervention artifact
5. runs a small position-4 and position-5 smoke evaluation
6. runs the resumable 29-condition full-GSM8K evaluation
7. applies the preregistered paired analysis and displays its Markdown report

Each condition is written atomically and independently. If Colab disconnects, rerun
the full-evaluation cell with the same parameters. Completed conditions are verified
and skipped. Colab still requires an active runtime. Closing the browser or losing the
runtime is not guaranteed background execution.

## Durable outputs

```text
MyDrive/CODI_KAVA/
  outputs/official_codi_kv_causal/
    subspaces/
      student_rank4.pt
      student_rank4.json
    smoke_gsm8k/
    full_gsm8k/
      run_manifest.json
      summary.json
      baseline/
      retain_learned_p0/ ... retain_learned_all/
      retain_random_p0/  ... retain_random_all/
      remove_learned_p0/ ... remove_learned_all/
      remove_random_p0/  ... remove_random_all/
  reports/official_codi_kv_causal/
    official_codi_rank4_full_gsm8k.json
    official_codi_rank4_full_gsm8k.md
  logs/official_codi_kv_causal/
```

## Interpretation boundary

A positive result shows that learned rank-four student KV directions have
position-specific causal value beyond an energy-matched random rank-four subspace in
the official CODI checkpoint. It does not show that those directions implement a
human-interpretable reasoning algorithm, nor that using them as training targets will
improve a newly trained student.

A negative result means the earlier predictive subspaces should not be treated as
causal answer-relevant directions under this intervention contract. It does not erase
the held-out prediction result.
