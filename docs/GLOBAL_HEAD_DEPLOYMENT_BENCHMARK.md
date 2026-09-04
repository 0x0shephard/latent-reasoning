# Global rank-96 head deployment benchmark

The notebook `notebooks/kaggle_global_head_deployment_benchmark.ipynb` tests whether
the validated trajectory-whitened rank-96 CODI head can produce a real deployment
speedup. It does not refit the head.

## Required Kaggle inputs

Attach both of these completed outputs:

1. the official CODI reproduction dataset containing
   `official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json`;
2. `jonraza15/trajectory-whitened-global-low-rank-lm-head`, containing
   `global_low_rank_head.pt` and its adjacent `summary.json`.

Internet and a Kaggle GPU must be enabled. The official checkpoint is discovered in
the attached inputs when present and otherwise downloaded from the pinned source.

## Arms

- dense tied FP16 head;
- ordinary eager rank-96 FP16 head;
- `torch.compile` rank-96 logits-plus-argmax;
- Triton rank-96 projection-plus-blockwise-argmax, which does not materialize the
  full vocabulary-logit tensor.

All arms share the same merged-LoRA CODI transformer, body-only decoder, prompt
preparation, maximum generation length, and greedy vocabulary boundary. Unsupported
compiled or Triton arms are recorded rather than silently substituted.

## Primary gate

An arm passes only if it retains at least 98% of the current dense FP16 full-test
accuracy, achieves at least 1.10x median batch-1 speedup per question, and does not
increase peak temporary CUDA allocation. Batch 8 and 32 are secondary throughput
regimes.

The output is written to
`/kaggle/working/codi_global_head_deployment/summary.json`, with per-example quality
JSONL files and compact profiler tables in the same directory.
