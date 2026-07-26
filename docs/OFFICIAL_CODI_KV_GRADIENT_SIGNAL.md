# Official CODI sparse answer-aligned KV gradient signal

## Research question

> Does useful KV supervision exist only as a sparse, consistently answer-aligned
> gradient component that is obscured when complete KV targets are distilled together?

This follows the completed kind-level utility screen, in which neither globally pooled
key nor value targets passed the predefined held-out answer-loss gate.

## Fixed scope

- author-released CODI GPT-2 checkpoint at revision `fd641b3d3edc`
- official checkpoint must pass complete 1,319-example GSM8K reproduction
- completed `kind_seed3` target-utility artifact is required
- all 256 normalized question groups from that screen are excluded
- six student latent positions and R-KV-aligned explicit-teacher targets
- keys and values are fitted and tested separately
- five percent coordinate budget
- primary kind is key; value is a secondary diagnostic
- L1 KV target loss
- official numeric gold-answer NLL is the differentiable outcome

## Three fresh split roles

One seeded sample creates three equal, normalized-question-disjoint splits.

1. **Calibration** learns a frozen coordinate mask.
2. **Update** produces answer and KV gradients.
3. **Validation** measures the stateless parameter update on untouched answers.

No validation gradient, loss, or label is used to fit the mask.

## Signal definition

For every trainable official-CODI LoRA/projection parameter coordinate `i` and
calibration batch `b`:

```text
contribution[b, i] = answer_gradient[b, i] * KV_gradient[b, i]
```

A coordinate is eligible when:

- its mean contribution is positive
- its contribution is positive on at least 60 percent of calibration batches

Eligible coordinates are ranked by:

```text
positive mean contribution * (positive batch fraction - 0.5)
```

The top five percent of all trainable coordinates form the frozen learned mask.
A seeded random mask has exactly the same cardinality.

This defines signal in **optimization space**, not by reconstructing KV activations.

## Held-out conditions

For each KV kind, the update split produces:

- `full` — complete paired KV gradient
- `sparse_aligned` — paired KV gradient inside the frozen learned mask
- `random_sparse` — paired KV gradient inside the cardinality-matched random mask
- `shuffled_sparse` — shuffled-pairing KV gradient inside the learned mask
- `complement` — paired KV gradient outside the learned mask
- `no_target` — answer gradient only

Every auxiliary condition is rescaled to the full paired KV-gradient L2 norm for the
same batch. Every combined answer-plus-auxiliary parameter update is then normalized to
the same total parameter L2 norm.

## Primary gate

The primary key component is supported when paired batch-bootstrap 95 percent intervals
are positive for:

- sparse versus no target
- sparse versus full KV
- sparse versus random sparse
- sparse versus shuffled sparse
- sparse versus the complement

The median update-batch gradient cosine must also be positive.

The stronger word **only** is supported only when the complement-versus-no-target
interval also has a non-positive upper bound.

Value results are secondary and do not alter the primary key gate.

## Run

```bash
python -u scripts/run_official_codi_kv_gradient_signal.py \
  --config configs/official_codi_gpt2.yaml \
  --reproduction-summary /path/to/full_gsm8k/summary.json \
  --prior-target-utility-dir /path/to/kind_seed3 \
  --output-dir /path/to/official_codi_kv_gradient_signal/full_seed5 \
  --examples-per-split 128 \
  --batch-size 4 \
  --sparsity 0.05 \
  --minimum-positive-fraction 0.60 \
  --precision float32 \
  --device cuda
```

Use the Kaggle notebook
`notebooks/kaggle_official_codi_kv_gradient_signal.ipynb` for the complete workflow.

## Resume and outputs

```text
output_dir/
  run_manifest.json
  mask_artifact.pt
  mask_artifact.json
  batches/
    batch_000000.json
    ...
  summary.json
  report.md
```

Mask fitting is deterministic and persisted once complete. Evaluation batches are
atomic and are verified before reuse.

## Interpretation boundary

A positive result establishes only that a calibration-selected sparse KV gradient
component improves one-step held-out gold-answer loss at the frozen official checkpoint.
It does not establish exact-match accuracy improvement, human-interpretable reasoning,
or long-run distillation benefit. Those still require a separate frozen-mask causal
test and compute-matched training experiment.
