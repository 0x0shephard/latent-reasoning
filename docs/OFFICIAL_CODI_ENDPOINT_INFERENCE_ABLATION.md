# Frozen-checkpoint CODI answer-colon inference ablation

## Question

The previous retention experiment changed a training target and then fine-tuned the
model. It did **not** remove hidden-state directions during inference, so it could not
show that the retained directions caused CODI's 43% GSM8K accuracy.

This experiment asks the causal question directly:

> If a direction is removed from the already-trained CODI checkpoint at the exact
> student answer-cue colon, does held-out GSM8K accuracy fall?

No parameter is updated. Every arm loads the same official checkpoint and uses greedy
generation. The only difference is a temporary hidden-state edit at the endpoint.

## Exact intervention

The student generates

```text
question -> BOT -> z1 ... z6 -> EOT -> "The answer is:" -> answer
```

The code tracks the token IDs of `" The answer is:"`. When the current forward pass
consumes the first exact generated colon, it edits the selected GPT-2 block output. That
is the hidden state which produces the first answer-token logits. Every run records the
per-question cue-reach mask; analysis rejects arms whose mask differs from baseline.
The confirmatory analysis also stops if fewer than 95% of baseline questions reach the
exact cue, because a mostly unapplied intervention would not answer the endpoint question.

For state `s` and orthonormal basis `U_s`, the edit is

\[
h'_s = h_s - U_s U_s^\top (h_s - \mu_s),
\]

where `mu_s` is a frozen student answer-colon mean fitted on 1,024 fresh training
questions. These questions exclude all three selector experiments and the completed
retention-training partition. Thus the ablated coordinate is replaced by its ordinary
student mean; all orthogonal coordinates remain bitwise unchanged.

State 11 is block 10's output (the input to block 11). Hugging Face GPT-2 records state
12 after block 11 **and** the final `ln_f`, so the state-12 hook is placed after `ln_f`
rather than on the raw block output. These are the exact tensors used to fit the bases.
The embedding state is never touched. `alpha=1` is a complete removal, not attenuation
or amplification.

## Registered arms

The three source bases are loaded from their immutable completed artifacts and forced to
the same rank: three directions at state 11 plus three at state 12.

- `baseline`: no intervention.
- Three joint candidate arms: remove all six energy, answer-conditioned, or
  parameter-aware directions.
- Eighteen localization arms: remove each of the three directions at each of the two
  states for each selector. Reports retain the original residual-PC index, not only its
  position in the selected basis.
- Twenty random joint arms: remove rank 3 at both states.
- Forty random single-direction arms: twenty independently seeded directions at each
  state.

There are 82 full-GSM8K arms. Random bases are deterministic, orthonormal, and generated
before evaluation. They estimate how much accuracy can move merely because an arbitrary
rank-matched direction was removed.

## Paired accuracy analysis

All arms evaluate the same 1,319 GSM8K questions in the same order. For each candidate,
the analysis records:

- baseline and ablated exact-match accuracy;
- baseline-correct to ablated-wrong losses and reverse gains;
- the paired accuracy loss and bootstrap 95% interval;
- a one-sided exact McNemar test;
- losses in deterministic even/odd question halves;
- an empirical p-value against the matching random-ablation family.

A direction or joint group is called **accuracy-critical** only if all conditions hold:

1. accuracy loss is positive in both deterministic halves;
2. the one-sided McNemar p-value survives Holm correction within the joint or individual
   candidate family at 0.05;
3. its loss exceeds the matched random null with empirical p at most 0.05.

Twenty random replicates are required because fewer than 19 cannot produce an empirical
p-value below 0.05 after the standard plus-one correction.

## Interpretation

If an answer-aware direction passes, removing that exact block/residual-PC direction
causally harms the frozen checkpoint at the answer boundary. If none passes, the correct
conclusion is not that the selectors are useless; it is that these low-rank residual
directions are not individually necessary for endpoint accuracy under this intervention.
The model may use distributed, redundant, earlier, or nonlinear information.

This experiment does not claim an inference speedup. Projection hooks add work, and a
directional ablation does not shrink GPT-2's width or remove a transformer block. A speed
claim would require training/exporting a structurally narrower model after causal
necessity has first been established.

## Implementation

- `src/mech/endpoint_inference_ablation.py`: arm registry and causal block hooks.
- `scripts/collect_official_codi_endpoint_activation_stats.py`: fresh mean calibration.
- `scripts/run_official_codi_endpoint_inference_ablation.py`: one resumable frozen arm.
- `scripts/analyze_official_codi_endpoint_inference_ablation.py`: paired confirmatory
  analysis.
- `notebooks/kaggle_official_codi_endpoint_inference_ablation.ipynb`: complete Kaggle
  run-all workflow.
