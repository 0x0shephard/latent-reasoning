# CODI request-adaptive xKV router

## Motivation

The storage-matched adaptive-allocation experiment selected the rank-64,
answer-Fisher profile on its fresh screen. On the locked GSM8K final slice it
preserved dense generation accuracy at roughly 1.48× modeled cache compression,
but the paired NLL interval crossed zero. The failure therefore does not show
that adaptive allocation destroys useful information; it shows that one global
allocation did not produce a stable average advantage.

This follow-up tests whether question types prefer different allocations.

## Frozen protocol

- Dataset: all 1,000 pinned SVAMP rows.
- Status: external xKV-method holdout, not a pristine model benchmark; dense
  CODI performance on SVAMP was reported previously.
- Deterministic hash split: 400 router-fit, 300 screen, 300 locked final.
- Storage budget: ordinary xKV rank 64 for every request.
- Profiles: the predecessor's already-fitted reconstruction (`w=0`), hybrid
  (`w=.5`), and answer-Fisher (`w=1`) utility curves.
- Router: multi-output ridge regression with fixed penalty 1.0.
- Inputs: twelve lexical/numeric question features available before decoding.
- Target: request-centered first-token KL from dense CODI on the fit split.

The router never receives a gold answer, a generated token, correctness, or a
post-answer activation. All three profiles are evaluated in identical batches,
and explicit padding remains part of modeled cache storage.

## Screen gate

The final 300 questions are opened only if all screen checks pass:

1. the paired 95% interval for global-answer-Fisher KL minus routed KL is positive;
2. dense first-token top-1 agreement is at least 95%;
3. routed generation accuracy is non-inferior to dense, ordinary rank-64 xKV,
   and the global answer-Fisher profile within two percentage points;
4. modeled storage matches ordinary rank-64 xKV; and
5. at least two profiles are selected, proving the result is actually routing
   rather than silently recovering another fixed profile.

The identical gate is applied once on the locked final split. Passing supports
the narrow claim that pre-answer request routing improves this frozen global
allocation at matched modeled storage. It does not establish a native latency
improvement; that requires a factor-consuming attention kernel.
