# Official CODI systems fast-path experiment

## Status

Implemented and awaiting the locked Kaggle run. No speed or accuracy outcome is
claimed before `summary.json` from the complete notebook is inspected.

## Question

Can the released CODI GPT-2 inference path be accelerated materially by removing
unused computation and framework overhead, then dropping one empirically redundant
continuous-thought pass, while retaining at least 98% of the reproduced GSM8K exact
match?

The experiment follows the completed position-conditioned readout study. That study
showed that a much cheaper vocabulary head produces only a small complete-model
speedup, so the present experiment targets repeated transformer execution.

## Frozen population and measurements

- Accuracy: complete 1,319-example GSM8K test.
- Timing: a seeded 128-question GSM8K-training subset, never used for fitting.
- Timing repetitions: three after warm-up.
- Batch sizes: 1, 8, and 32.
- Batch 1 is request latency. Batch 8 and 32 are throughput measurements normalized
  per question and must not be described as single-request latency.
- Every arm records generated-token count, exact answer, output-string agreement,
  model time, input-preparation time, and combined service time.

## Frozen cumulative arms

| Arm | Change from the preceding arm | Claim class |
| --- | --- | --- |
| `b0_reference_fp32_lora_m6` | Released eager forced-cue path | Reproduction baseline |
| `b1_body_only_fp32_lora_m6` | Do not compute discarded vocabulary logits | Exact semantic control |
| `b2_body_only_fp32_merged_m6` | Merge rank-128 LoRA branches | Intended lossless optimization |
| `b3_exact_fastpath_fp32_m6` | Parity-checked fast tokenizer; original batching | Intended lossless optimization |
| `b4_bucketed_fp32_m6` | Group similar lengths to reduce prompt padding | Quality-checked systems optimization |
| `b5_fastpath_fp16_m6` | FP16 model execution | Numerical optimization |
| `b6_fastpath_fp16_m5` | Five rather than six continuous thoughts | Primary deployment candidate |
| `b7_fastpath_fp16_m5_numeric` | Tokenizer-semantic numeric vocabulary | GSM8K-only exploration |
| `b8_compiled_fastpath_fp16_m5` | Dynamic `torch.compile` probe | Optional systems diagnostic |

The compiler arm fails safely. The pinned Transformers implementation uses a custom
continuous-latent loop and growing legacy cache, so compiler support is an empirical
property of the installed PyTorch/Kaggle combination rather than a prerequisite for
the other results.

## Computation removed by B1

The released evaluation calls `GPT2LMHeadModel` for the prompt and for every latent
thought. This produces logits for all returned hidden vectors even though the decoder
uses only hidden state and KV cache during those phases. B1 calls the injected GPT-2
transformer body directly and applies the output embedding only at answer decisions.

The mandatory B1 gate is 100% decoded-string agreement with B0 in FP32. Failure stops
the notebook before any fast-path interpretation.

## LoRA merge

The loaded checkpoint represents each adapted map as the base matrix plus a rank-128
branch. B2 calls PEFT's `merge_and_unload()` after checkpoint validation, forming the
combined dense weights once offline and removing the adapter branches from inference.
The notebook records output agreement because finite-precision addition order may
change rare argmax ties.

## Quality and speed gates

The intended-lossless B3 fast path preserves the original batch composition because
GPT-2's released left-padded execution is not guaranteed to be padding invariant.
Length bucketing is isolated in B4 and receives the same accuracy checks as numerical
optimizations. B3 is supported only if:

1. B1 first passes exact output parity;
2. B3 has 100% output-string agreement with B0; and
3. B3 reaches at least 1.20x batch-one service speedup.

The primary B6 deployment candidate is supported only if:

1. its exact-match accuracy is at least 98% of B0; and
2. its median batch-one service speedup is at least 1.50x.

A paired bootstrap 95% interval for each arm's exact-match difference from B0 is
reported. Full-test timing is recorded but the repeated timing subset is the primary
speed measurement.

## Numeric shortlist limitation

B7 builds candidates solely from tokenizer pieces consisting of digits and numeric
punctuation, plus EOS. It does not read evaluation labels, states, or predictions.
Nevertheless, it changes the task interface and cannot establish general-language
inference acceleration. It is retained because a forced-cue GSM8K deployment may
legitimately require only numeric outputs.

## Artifacts

- implementation: `src/inference/official_codi_fast.py`
- notebook builder: `scripts/build_kaggle_codi_systems_fastpath_notebook.py`
- notebook: `notebooks/kaggle_codi_systems_fastpath.ipynb`
- tests: `tests/test_official_codi_fast.py` and
  `tests/test_kaggle_codi_systems_fastpath_notebook.py`
