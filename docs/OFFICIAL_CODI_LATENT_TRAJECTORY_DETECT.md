# Latent-trajectory detection gate

## Question

Do CODI's six latent thought states still hold linearly recoverable information that
the answer endpoint has lost — enough to justify designing an editing experiment at
those states?

Every completed intervention experiment acted at state 12 of the forced answer cue,
after computation has collapsed into a single token choice (§40, §43, §47, §49, §50).
The latent states are the one location that satisfies both requirements those nulls
identified: they enter the KV cache, so an edit would propagate, and they are captured
while reasoning is still in progress. This experiment is the §19-mandated cheap gate
before any expensive intervention design there. It is read-only: no weight is updated,
no state is edited, and no inference-speed claim is made.

## Collection

One observational GPU pass over the cached 1,319 GSM8K-test questions runs the
released generation path unmodified with two observers attached:

- the trajectory capture buffers every forward pass through hooks and commits a pass
  only when the released latent loop's per-position callback fires, so exactly the six
  latent iterations are kept — thirteen states each (embedding input, the outputs of
  blocks 0–10, and the `ln_f` output), the same 13-state layout every endpoint
  experiment used;
- a zero-noise endpoint capture records the live forced-cue state 12 of the same pass.

The live pass is validated on its own terms by two hard gates, mirroring the checks
that validated the original cache on *its* pass: an analytic parity gate
(`argmax(W·h12)` must reproduce the token the decoder actually emitted, agreement
≥ 0.99) and an accuracy-reproduction gate (live first-token accuracy within 0.02 of
the cache's). A first attempt gated instead on exact vector agreement with the cache
and failed at maximum relative deviation 2.35, identically at two different batch
sizes: the colon-state cache predates the environment pins (§40's recorded remaining
limit), so its exact vectors are not reproducible on the pinned image even though
aggregate accuracy is. The cache deviation is therefore reported as a diagnostic, and
the cache supplies only data — questions, gold first tokens, and the partition —
while every state consumed by the probes (trajectory, labels, margins, and the
endpoint baseline) comes from the one live, internally consistent pass. For this
experiment that also removes §40's last-dependency caveat.

The collection reproduces the exact chunking recorded inside the cache (batch size
16), because GPT-2's absolute position ids depend on each chunk's left-padding width.
The export is `[1319, 6, 13, 768]` float32 plus the live endpoint states, with the
collection request, partition, and gates embedded.

## Frozen population and probes

The deterministic partition is the same one used by §47, §49 and §50 — seed
`20260827`, fit 440 / select 440 / test 439, partition SHA
`c8316e46…` asserted in-run. Correctness labels and margins come from the validated
colon-state cache. Every probe is fitted on fit; every cell and ridge is chosen on
select; the final test split is read once per frozen arm.

Correctness labels and margins are derived from the live endpoint state, so probes,
labels, and baselines share one pass and one environment. The probe grid is all 78
cells (6 latent positions × 13 states):

- **correctness track**: ridge-logistic on `[cell state, endpoint margin]` with the
  convergence-certified L-BFGS solver (§45's), against a margin-only baseline fitted
  the same way. The cell and ridge maximizing select AUC are chosen.
- **answer-identity track**: an exact closed-form one-hot ridge classifier over the
  gold first-answer-token classes observed in fit, selected by gold-token accuracy on
  the select split's *wrong* questions. Its baselines are the same probe class fitted
  on the endpoint colon state, selected identically, and the majority class.

## Frozen gates

| gate | passes if |
|---|---|
| correctness | chosen cell's test AUC beats margin-only by ≥ **0.02** with a positive paired-bootstrap lower bound, both fits convergence-certified |
| answer identity | on final-test wrong questions, chosen cell's gold-token accuracy beats **both** the endpoint probe and the majority class, by ≥ **5.0 points** over the stronger of the two, with positive paired-bootstrap lower bounds against each |

The correctness threshold is 0.02, not §43's 0.01, because §49 measured that a ~0.013
AUC increment cannot be bounded away from zero on 439 test questions; a gate this
sample size cannot resolve would be theater.

**Decision rule.** Only a passed answer-identity gate justifies proposing a
latent-state editing experiment — it is the direct measurement of whether the
trajectory still knows an answer the endpoint discarded, which is what an editor
would exploit. A correctness-only pass is a detection finding and licenses nothing
further. A double failure closes the latent-trajectory question for linear probes.

This population's aggregate behavior has been inspected by earlier experiments; the
frozen split prevents tuning on the final 439 rows inside this run, but the result is
a corrective-lineage experiment, not a pristine preregistration.

## Run

Use
[`kaggle_official_codi_latent_trajectory_detect.ipynb`](../notebooks/kaggle_official_codi_latent_trajectory_detect.ipynb).
Attach the completed `colon_states.pt`, `readout.pt`, and official reproduction
`summary.json`. The collection pass requires a Kaggle GPU; the probes are CPU-capable.
The notebook pins the checkpoint-compatible package versions and removes Kaggle's
incompatible optional TorchAO build before importing PEFT.

## Status

Implementation and synthetic validation are complete. The real trajectory has not
been collected, so the repository makes no empirical claim about either gate.
