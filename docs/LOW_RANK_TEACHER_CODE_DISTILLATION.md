# Low-rank teacher-code distillation: experiment specification

Protocol dated 2026-09-20. A Kaggle implementation was added on 2026-09-21; see [the run guide](KAGGLE_TEACHER_CODE_DISTILLATION.md). No full GPU experimental result has been established.
Working title: Do task-fitted low-rank teacher codes preserve useful distillation signal?
Repository baseline: 2383cc5. Freeze the implementation commit and all artifact hashes before execution.

## 1. Research question and scope

Can a learned 96-dimensional code replace the teacher targets used to train a smaller student, preserving downstream quality while improving the accuracy-versus-total-cache-storage tradeoff against strong compact alternatives?

The existing result establishes approximate preservation of a frozen teacher's answer decisions when its vocabulary head is replaced. It does not establish preservation of the teacher distribution, of distillation gradients, or of what a different student can learn.

This experiment tests target compression. Every student arm has the same architecture, dense output head, initialization, training examples, supervised labels, optimizer budget, and inference procedure. The low-rank head is a frozen teacher-target encoder/decoder, never the student's inference head.

Primary setting: the released CODI GPT-2 checkpoint in its explicit-CoT path, distilled into DistilGPT2. This provides a conventional, token-aligned student and substantially more supervised positions than CODI's short answers. It tests a task-fitted codec on a teacher already supported by this repo; it does not establish latent-reasoning transfer. A separately gated CODI extension is specified below.

## 2. Hypotheses and expected outcomes

H1, target sufficiency: rank-96 targets can train a student within 1.0 absolute percentage point of full-target distillation on GSM8K exact match.

H2, practical competitiveness: at matched total serialized teacher-target bytes, rank-96 targets produce better student accuracy than both a tail-aware top-k representation and sampled-token targets.

H3, mechanism: preserving the whole distribution's useful directions gives more faithful distillation gradients than compact representations that discard or sample vocabulary information. H3 is explanatory and can fail even if H1 passes.

Expected, not established:
- Rank 96 should preserve targets better than rank 32, but independently realized student accuracy need not be monotone.
- Rank-96 student performance may be close to full KD, but current teacher answer-retention numbers do not predict this reliably.
- Top-k and sampling may tie or beat the learned codec. A close result against these controls is not a novelty claim.
- Full KD may fail to improve the student because the teacher is imperfect or the training set is small. Then similarity to full KD alone is uninformative about useful knowledge transfer.
- The codec should reduce per-token cache payload. It need not reduce total disk use on a small dataset, GPU peak memory, or training wall time.
- No inference speedup is expected between student arms: their deployed architecture is identical.

Interpretation limits: the dense linear readout already has rank at most hidden width 768. Rank 96 is an 8x reduction of that rank bound, not a 500x reduction of intrinsic task dimensionality. No claim about universal math subspaces follows from this experiment.

## 3. Teacher, student, and vocabulary contract

Teacher:
- Official zen-E/CODI-gpt2 checkpoint, revision fd641b3d3edc59e4f534b55588e906588c9e36bb.
- Checkpoint SHA-256 fd223b14932b8b66605c2a688e7d7193058ff2b40b2c6f6c1547c227d06d8417.
- Load through src/models/official_codi.py; merge released LoRA for inference only after parity checking. Freeze all teacher weights; dropout off.
- Explicit mode: question-only prompt, no latent iterations and no forced answer cue before generated reasoning. Do not substitute the local compute-limited pilot model.

Student:
- distilbert/distilgpt2: six transformer blocks, hidden width 768, original GPT-2 tokenizer/vocabulary.
- Resolve and save the exact immutable Hugging Face revision and file hashes before any pilot. No floating main revision in a run.
- Train the whole student, retaining its native tied embeddings/dense output head. No low-rank student head, LoRA, layer surgery, or latent projector.
- Every arm within a seed starts from identical checkpoint bytes and RNG state. Different seeds vary training order/dropout, not architecture.

Common vocabulary:
- Original GPT-2 token IDs 0..50256, V=50257, including EOS.
- Teacher-added PAD/BOT/EOT tokens are excluded from explicit-mode KD and visible decoding. The official wrapper may retain their embedding rows internally.
- Assert identical token-ID-to-string mapping across the original vocabulary; record excluded special-token probability mass before renormalization.
- This restricted explicit teacher is the reference for this experiment; do not copy numerical accuracy from historical runs using a different vocabulary boundary.

Numeric reference:
- Teacher body inference uses fixed FP16 on the selected GPU. Save final-normalized hidden states as FP16.
- Define canonical dense logits by FP32 linear projection of those serialized hidden states through the frozen teacher output matrix, converted from its recorded inference weights to FP32.
- All codecs are fitted and evaluated against this same canonical reference.
- Measure the numerical difference from direct model logits and from unrounded states on an audit batch. It must be reported, not folded into codec error.
- Compute softmax, log-softmax, normalization, and loss reductions in FP32. No quantized weights, compilation, or custom fused kernels in the scientific comparison.

## 4. Data and exact sequences

Use canonical GSM8K main/train, nominally 7,473 questions, with pinned dataset revision/hash. Never split augmented rows independently.

Question identity is whitespace-normalized, casefolded text. Sort unique keys, then permute with split seed 20260920. Stop on contradictory duplicate answers. Split before collecting states:

| Partition | Nominal unique questions | Permitted use |
|---|---:|---|
| Codec fit | 1,024 | Fit mean/covariance/factors and codec parameters |
| Codec selection | 256 | Select codec epoch using target KL only |
| Codec audit | 256 | Frozen target/gradient diagnostics; development gate |
| Student development | 512 | Pilot hyperparameters and feasibility only |
| Student training | Remaining, nominally 5,425 | All student optimizer updates |

If deduplication changes the total, keep the first four sizes and use the remainder for student training. Publish actual counts and exclusions.

Gold sequence construction:
1. Normalize the question exactly as the repository's official question normalization.
2. Split the canonical GSM8K answer at its last "####". The preceding text is the rationale; the following field is the numeric answer.
3. Completion is one space, rationale.strip(), " The answer is: ", normalized numeric answer, then the GPT-2 EOS token.
4. Preserve rationale content, including calculator annotations, verbatim. This avoids an unrecorded cleaning transformation.
5. Tokenize prompt and completion separately, concatenate their token IDs, and record the exact boundary. Use that same token construction everywhere.
6. Teacher and student see identical gold prefixes. Cache the teacher state predicting each completion token, including EOS; never align a state with the token it has already consumed.
7. Loss applies only to completion tokens. Mask prompt tokens and padding. Do not append rationale or answer content to inference prompts.

Context budget: 1,024 tokens, after verifying both model position tables support it. Exclude overlength training/codec/development sequences before any arm runs; never truncate away answers. All arms share the same eligible IDs. Record counts and answer lengths by split.

Teacher forcing is deliberate: the target at a prefix is identical for all arms. Do not mix teacher free-generation prefixes, compressed-head prefixes, or student on-policy trajectories into the primary training cache.

No teacher-correct-only filtering. Include erroneous teacher preferences and log how often teacher argmax disagrees with gold. Correctness filtering would change the population and conflate compression with teacher-example selection.

Tests:
- GSM8K main/test: all 1,319 questions, primary.
- SVAMP original 1,000 questions, secondary transfer.
- Check exact normalized-question overlap against every fitting/development split. Exclude and disclose overlaps under a frozen rule; do not silently call a reduced set the full benchmark.
- Both benchmarks have been inspected in this project. This is a prospective locked follow-up on reused benchmarks, not an untouched confirmatory holdout.

## 5. Fit the teacher codec

Refit on the primary explicit, gold-prefix distribution. Do not reuse CODI-answer factors or assume the old free-generation head covers this distribution.

Use the repo's affine form:
c = D h + d; reconstructed logits = U c + b.
Freeze all codec parameters before student training. Cache c as FP16. Store U and b as FP16 and decode in FP32 for the reference loss. Count the cast/rounding error as part of the codec.

Initialization: activation-whitened factorization using src/mech/global_low_rank_head.py.
Nested ranks: 32, 64, 96. Each smaller code is the corresponding prefix; no claim that these are independently optimal ranks.

Fitting defaults:
- Seed 89.
- At most 32,768 fit states and 8,192 selection states, uniformly sampled over eligible token positions using a fixed seed. Retain question/token IDs and document the resulting question coverage.
- Six fixed epochs, AdamW, learning rate 2e-4, betas (0.9, 0.999), weight decay 0, batch 32 states, gradient norm clip 1.
- Temperature T=2.
- Loss: average forward KL across the three nested ranks, multiplied by T squared. Use KL only; no teacher-top-token CE or margin penalty in the primary codec, because this experiment aims to preserve soft targets.
- Select the epoch with lowest rank-96 selection KL; ties choose the earlier epoch. Save all rank diagnostics.
- No on-policy recovery: it changes the state population away from the fixed-prefix offline KD problem.
- Quantized serialized decode, not a higher-precision in-memory version, must pass the fidelity audit.

This intentionally adapts the earlier inference-head recipe. The original inference head and this KD-target codec answer different optimization questions.

Initializer control: fit a nested head from ordinary weight SVD with identical training/selection budget. Its rank-96 result tests whether activation-aware initialization matters. This control does not isolate whether training itself is necessary; report the untrained initializers in the cheap fidelity audit.

## 6. Student arms and losses

Let y be the gold next token, q_T the student's temperature-T distribution, and p_T the canonical full teacher distribution. All KD arms use:
L = (1-alpha) * CE(y, q_1) + alpha * T^2 * H(target, q_T),
with alpha=0.5 and T=2 by default. Teacher entropy is constant with respect to student parameters, so cross-entropy gives the full-KL gradient. Report KL separately for diagnostics.

SFT uses CE alone. Full KD and compressed KD share exactly the same alpha, T and token normalization; never retune them independently by arm.

Primary arms:
- SFT: no teacher target.
- Full KD: canonical p_T reconstructed from cached 768-dimensional states and the frozen dense teacher head.
- LR96: full-vocabulary target reconstructed from a serialized 96-dimensional code.
- TopK-total: tail-aware top-k targets, with k chosen mechanically to match LR96 total target-package bytes.
- Sample-total: token samples from p_T, with sample count chosen mechanically to match LR96 total target-package bytes.

Secondary, predeclared arms:
- LR32 and LR64.
- WeightSVD96: the same trained nested codec recipe with weight-only initialization.

Full logits versus hidden-state cache:
A separate expensive full-logit student run is unnecessary: the hidden-state path defines exactly the same canonical targets. Verify equality of losses and gradients against explicitly materialized logits on small batches. Include materialized full-logit storage and throughput in systems microbenchmarks, but do not pretend two equivalent target sources are independent scientific arms.

Tail-aware top-k:
- Save k uint16 token IDs, k FP16 log probabilities at T=2, and one FP32 log tail mass.
- Decode and normalize the k+1 masses stably after quantization.
- Student top entries use their full-vocabulary probabilities; the remaining student mass is one aggregate tail category.
- KD loss is cross-entropy between these k+1 teacher/student categories. Never renormalize the teacher's top-k entries to 1 while silently discarding the tail.
- The tail category changes what information is supervised; it does not reconstruct a unique full-vocabulary distribution. Do not report full-vocabulary KL for it without explicitly adding a tail model.
- Compute student tail log-mass using a stable logsumexp over excluded logits, avoiding cancellation in 1-sum(top).

Sampled target:
- Draw m tokens independently with replacement from p_T once per training position, using a frozen cache seed; save uint16 IDs, including duplicates.
- KD loss is the mean negative log student probability of those m samples, multiplied by T squared and alpha.
- The gradient is unbiased over cache draws for fixed student logits. Reusing one finite cache introduces persistent sampling error; do not claim each repeated training step is a fresh unbiased draw.
- Pair sampling cache seed with the student seed in formal runs. The storage figure is for one deployment cache, not the sum of independent experimental replications.
- This is a direct-teacher sampling baseline inspired by sparse KD; do not claim exact reproduction of another paper's complete method.

Normalize training loss over nonpadding completion tokens across each effective batch. Preserve the identical order and accumulation boundaries across arms. Do not compare raw logged KD loss magnitudes across formats: their entropy constants and category spaces differ.

## 7. Fair storage accounting

Primary resource is TOTAL serialized target-package bytes:
token payload + shared decoder parameters + target-specific metadata.
Common questions, labels, token offsets and manifests are reported separately and also included in total experiment disk usage.

For FP16 and V=50257:
- Full logits: 2V = 100,514 bytes per position.
- Hidden state: 1,536 bytes per position; shared dense readout 2*V*768 = 77,194,752 bytes, plus any bias.
- LR96: 192 bytes per position; shared U,b = 2*V*(96+1) = 9,749,858 bytes.
- Top-k: 4k+4 bytes per position under the format above.
- Samples: 2m bytes per position.

For a train cache of N positions, choose:
k_total = floor((192 + 9,749,858/N - 4)/4)
m_total = floor((192 + 9,749,858/N)/2)
subject to vocabulary bounds. Verify actual serialization does not exceed LR96 total bytes; use packed binary arrays rather than per-token Python objects. Shared headers are included in the final exact check and may require rounding down once more.

Thus the main sparse controls receive MORE than the naive equal-payload k=47 or m=96 when N is finite. Equal-payload comparisons may be included as diagnostics, but cannot establish a total-storage advantage.

Example: at N=500,000 positions, LR96 is approximately 105.75 MB including its decoder. Full logits are 50.26 GB; exact hidden-state targets are about 845.20 MB including the dense head. TopK-total can use about 51 entries per position and Sample-total about 105 samples, subject to metadata. These are illustrations, not estimates of the actual dataset's token count.

Report:
- decimal MB/GB and raw bytes, both per-token and total;
- uncompressed serialized bytes (primary) and a common lossless-compressed representation (secondary);
- deployment package U,b separately from research archive including encoder D,d, optimizer and fitting caches;
- one-time teacher collection and codec-fitting time/storage;
- whether shared decoders are assumed already present. Use the cold-start case as primary, warm/shared-decoder case as secondary.

An about-500x payload ratio against dense logits is not an end-to-end speedup, not an intrinsic-dimension ratio, and not necessarily a 500x total-cache reduction.

## 8. Pilot, hyperparameters, and formal training

Hardware target: one T4-class 16 GB GPU, subject to measured feasibility. No promised runtime. Keep the chosen hardware and software stack fixed across arms.

Initial defaults:
- AdamW; LR 5e-5; betas (0.9,0.999); epsilon 1e-8; weight decay 0.01 except bias/normalization.
- Effective batch 16 questions via microbatch 1 and gradient accumulation 16; gradient clip 1.
- FP32 master weights/optimizer, FP16 autocast and loss scaling. Loss math FP32.
- Gradient checkpointing if needed; disable training KV cache. Teacher targets are read offline.
- Dropout retains the student's native setting. Train for 3 epochs.
- Linear warmup over 5% of optimizer steps, then cosine decay.
- Length bucketing/order shared within seed. Final partial batch uses its actual nonpadding token denominator.
- Optimizer, RNG, scheduler, scaler, batch position and cache fingerprints are checkpointed. Runtime interruptions resume, not discard slow/failing examples.

Bounded pilot:
1. Use a fixed 1,024-question subset of student-training IDs and the 512 student-development questions.
2. Train SFT and Full KD with the defaults. If both are unstable or clearly underfit, permit exactly one logged revision using LR in {2e-5,5e-5} and epochs in {3,5}; no compressed-arm-guided tuning.
3. If KD appears harmful, permit alpha in {0.25,0.5}; choose on development exact match, break ties with development gold NLL, then lower compute. T stays 2 throughout.
4. Pilot LR96 and both sparse controls under the selected common configuration.
5. If no full-KD learning benefit is visible and validation loss/accuracy indicate the task is not learnable in this budget, stop as an unsuitable testbed. Do not rescue the experiment by reporting codec-versus-KD equality near chance.
6. Publish the pilot decisions and freeze a final run manifest before final test inference.

Formal seeds: 89,90,91. Run all five primary arms for each seed: 15 student fits. All three secondary arms add 9 fits, for 24 total.
Before any final test outputs are inspected, development variance may trigger a predeclared extension to seeds 92,93 for EVERY primary arm. Record that decision first. Do not selectively add seeds to favorable arms after seeing test results.
Codec seed is fixed in this first study. Student-seed replication does not establish robustness to codec-fitting seeds; list this limitation and replicate codec seeds only in a later locked extension.

Primary scientific budget is equal updates/examples. Record wall time, but do not give a faster format additional steps. A later equal-wall-time experiment would answer another question.

All formal arms use the frozen final epoch budget and final checkpoint. No test-based or per-arm best-epoch selection.

## 9. Measurements

A. Target fidelity on the frozen codec-audit partition:
- KL(full teacher || reconstructed teacher) at T=1 and T=2 for dense-distribution codecs.
- Teacher top-1 agreement, top-5 overlap, top-two margin error, and gold-token NLL change.
- Tail-aware top-k retained mass and k+1 KL; sampled-target distribution error/coverage, with estimation limits stated.
- Stratify by rationale tokens, answer tokens, EOS, answer position, teacher-gold agreement and dense-teacher margin.
- Report means, quantiles, question coverage and question-clustered intervals. Token-level observations are not independent questions.
- Codec targets are judged after serialization/deserialization, including all rounding.

B. Gradient fidelity:
- At identical student weights and identical audited prefixes, compute the full-KD loss gradient and each compressed target's gradient.
- Measure cosine, relative L2 error and norm ratio for logit gradients; also use 16 fixed microbatches for aggregate trainable-parameter gradients.
- Evaluate at pretrained initialization and a common SFT pilot checkpoint. Never compare each arm at its own diverged weights when attributing gradient differences to the targets.
- Measure KD-only and complete CE+KD gradients separately; CE can hide poor teacher-gradient fidelity.
- Skip/flag near-zero reference gradients with a fixed epsilon; report their count.
- Efficiently accumulate layerwise dot products/norms; no need to retain all gradient vectors together.

C. Primary student quality:
- Free-generation numeric exact match on GSM8K, not teacher-forced token accuracy.
- Paired differences LR96 minus FullKD, LR96 minus SFT, and LR96 minus each sparse baseline.
- Per-seed accuracies and paired correct-to-wrong/wrong-to-correct counts.
- Absolute percentage points primary. Relative accuracy retention and fraction of KD gain retained are secondary; do not use the latter when FullKD-minus-SFT is small or negative.

D. Transfer and failure modes:
- SVAMP numeric exact match, no refitting or model selection.
- Gold completion NLL on held-out development questions, sequence agreement with teacher, visible length, EOS/empty/malformed output rates and cap hits.
- Report rationale/answer behavior separately where teacher-forced diagnostics permit.
- Student teacher-agreement is not a correctness metric.

E. Systems:
- Actual total cache bytes and bytes per labeled position.
- Teacher collection, codec fitting/encoding, and student training time separately; amortized totals for one student and reuse across three students.
- Training nonpadding completion tokens/second, GPU peak allocated/reserved memory, CPU RAM and target decode time.
- Measure clean uninstrumented training windows after warmup, with identical batch schedules and token counts. Use the same memory-safe loss chunking strategy where applicable.
- Report reconstruction compute: LR targets still reconstruct V logits and full student normalization remains.
- Record checkpoint/logging overhead and disk read throughput. No claim of systems benefit from FLOP counts alone.
- Student inference latency is optional descriptive data; target compression changes trained weights, not the inference architecture.

## 10. Evaluation and statistics

Inference: eager, batch 1, dropout off, greedy decoding, original common vocabulary, max_new_tokens=256, no answer oracle, no multiple samples, no tools. Prompt is the same normalized question used for training. EOS terminates; truncations count in ordinary accuracy and are separately flagged. Maximum prompt plus generation length must fit 1,024; report any violations without silently changing the cap.

Use the repo numeric answer extractor with a frozen version/hash, audited on signed numbers, commas, decimals, fractions and malformed outputs. A cap hit is not automatically incorrect if the frozen scorer finds a correct answer; do not invent a new scoring rule after looking at outputs.

Primary estimand: mean question accuracy over the prespecified student seeds, comparing paired arms.

Intervals:
- 10,000 paired bootstrap replicates, seed 20260920.
- Resample questions once per replicate, retaining the entire arm-by-seed correctness vector for each question.
- Also report a crossed bootstrap that resamples paired seeds and questions independently, and individual-seed deltas.
- Three seeds provide weak information about training-population uncertainty. Do not multiply test size by seed count or treat repeated tokens/epochs as independent replications.
- State which interval is conditional on these fitted students. Require the primary conclusions to survive the crossed interval; otherwise label seed robustness inconclusive.

Ordered decision procedure:
1. Teacher utility: FullKD must exceed SFT by at least 1.0 pp in point estimate and have a positive paired 95% lower bound. Otherwise no claim that compressed targets preserve useful distillation gains.
2. Target sufficiency: LR96-minus-FullKD must have its 95% lower bound above -1.0 pp. A nonsignificant difference is not evidence of equivalence.
3. Practical distinction: compare LR96 with BOTH total-byte-matched sparse controls. Require at least +0.5 pp point improvement and Holm-adjusted paired superiority at familywise 0.05 for both comparisons to claim a clear fixed-budget advantage. If it beats one only, report that narrower result.
4. SVAMP and rank/initializer ablations are secondary. Report all, with multiplicity-aware or explicitly exploratory intervals; they cannot rescue a failed primary claim.

For the two superiority comparisons, obtain one-sided bootstrap p-values from the null-centered paired difference distribution, with the finite-resample correction (1 + exceedances)/(B + 1), and apply Holm across the two tests. Use the crossed seed/question resampling for the seed-robust check; also show the question-only result. With very few seeds these are approximate inference procedures, not a substitute for more independent training runs.

These effect thresholds are proposed scientific tolerances, not derived guarantees. Freeze them before runs.

Power:
At roughly zero true paired difference and question discordance q, the approximate question-only SE is sqrt(q/n). With n=1319 and q=0.10, a two-sided 95% interval has about 1.7 pp half-width, too wide to establish a 1 pp noninferiority margin. Approximately 3,842 independent questions would be required for a 1 pp half-width at that discordance, before seed uncertainty.
Estimate discordance on development outputs BEFORE final evaluation and publish the minimum detectable effect. The existing benchmark may only support an informative pilot. Do not widen the margin post hoc, pool different benchmarks into GSM8K, or count more seeds as more independent questions. If underpowered, report estimates and an inconclusive gate and plan a separately locked larger/new evaluation.

## 11. Stop/go rules and result interpretations

Implementation gate: stop for vocabulary mismatch, special-token leakage, wrong prefix alignment, broken masks, nonfinite losses, failed serialization parity, or unaccounted data overlap. These are invalid runs, not negative scientific results.

Codec gate: inspect the untouched audit split once after selection. Stop student scaling if targets are numerically broken or grossly unfaithful (rank-96 teacher top-1 agreement below 90% OR mean T=2 KL above 0.5 nats/token). These are deliberately loose feasibility thresholds, not success criteria. Report any stopped result. One redesigned codec is a new protocol version, not silent tuning on audit data.

Pilot gate: full KD must show a usable learning signal on development before committing to the formal grid. A target reconstruction win alone cannot pass this gate.

Interpretation matrix:
- FullKD > SFT; LR96 noninferior; LR96 beats both compact controls: strongest evidence for a useful task-fitted target codec in this setting.
- FullKD > SFT; LR96 noninferior; sparse controls tie/win: compressibility is supported, but this codec has no demonstrated practical advantage.
- Good teacher top-1 retention; poor LR96 student: answer-preserving compression loses useful soft supervision.
- LR96 loses and target/gradient fidelity is poor: investigate codec mismatch; do not blame the student.
- All KD arms fail to improve SFT: teacher/student/data setup is inconclusive for the usefulness hypothesis.
- Wide intervals: inconclusive; show effect sizes and precision honestly.
- LR96 wins only against materialized full logits, not hidden-state/top-k/sample baselines: no strong compression-method claim.
- Lower payload but slower training or larger total package: a tradeoff, not an overall efficiency win.

Do not promise a paper before distinguishing this result from existing sparse/low-rank distillation methods.

## 12. Separately gated CODI extension

Only after a credible primary result, test the setting closest to the original head:
- Frozen official CODI teacher, six continuous thoughts, native question/BOT/latent/EOT/answer-cue contract.
- Smaller CODI student constructed from the same released weights with six blocks copied at zero-based teacher indices [0,2,4,6,8,10], preserving width 768, original token IDs, projector, final norm, positions and tied embedding/readout. Reindex caches explicitly. Train all student parameters.
- Student latent states are its own; never feed teacher hidden states into the student.
- Refitted codec on teacher-forced gold answer tokens after the teacher's latent computation. No gold rationale available to either latent inference path.
- Same output-target comparison and supervised answer loss; no extra endpoint/KV distillation losses.
- Establish a full-KD-versus-SFT pilot first: layer removal may break the latent mechanism.
- Recompute total-byte-matched sparse budgets using the much smaller answer-token cache. The decoder overhead may erase the compact-code advantage.
- Keep separate tables and conclusions. No averaging explicit and latent results to hide a failure.

This requires a new training adapter. src/models/latent_lm.py is not interchangeable with the official checkpoint wrapper; the repo documents different tokens, projector and generation semantics.

A successful explicit experiment alone does not prove CODI latent knowledge transfer, other-model-family generalization, or shared task subspaces.

## 13. Required artifacts and later implementation boundaries

Before execution:
- frozen specification version, full Git SHA, immutable model/data revisions and hashes;
- question partitions/exclusions, exact tokenized sequence manifests, special-token and masking checks;
- codec configuration/weights/selection history, audit report, storage budget calculation;
- pilot decisions and final configuration signed off in the run manifest before test access.

After execution:
- target-cache schema, typed packed arrays, decoder weights and hashes;
- every student checkpoint and training curve, RNG/resume metadata;
- per-question generated text, parsed answer, gold answer, correctness, token count and termination reason for every seed/arm;
- paired summaries and bootstrap settings; target and gradient diagnostics;
- actual bytes, timings, memory and complete failure log;
- completion marker only after all configured runs/evaluations finish.

Reuse:
- official model loader, numeric scorer and dataset provenance mechanisms;
- global_low_rank_head affine factorization and nested-rank utilities;
- global_head_ablations initializer controls where their assumptions apply.

New work later:
- fixed-prefix explicit teacher-state collector and target-codec serializer;
- dense/top-k-tail/sample KD losses and a conventional student trainer;
- storage-budget matcher, paired evaluator and audits.

Tests needed later: causal label shift; padding/EOS masks; vocabulary identity; codec serialization; exact H768 target reconstruction; top-k tail gradient finite differences; sampled-gradient Monte Carlo agreement; equal byte-budget validation; paired RNG/order; resume parity. CPU correctness tests do not establish GPU timing or scientific success.

This document defines the protocol. The primary notebook and trainer are now available; see the run guide for implemented stages and measurement limits. No full GPU result is claimed.

## 14. Sources and relation to prior work

- Repository: docs/GLOBAL_TRAJECTORY_WHITENED_LM_HEAD.md, src/mech/global_low_rank_head.py, src/mech/global_head_ablations.py, src/models/official_codi.py, docs/MODEL_DATASET_GLOBAL_HEAD_ABLATIONS.md.
- Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs (2025): https://arxiv.org/abs/2503.16870 . Motivates including sampling and a tail-aware sparse baseline; our fixed-byte protocol is not claimed as an exact paper reproduction.
- Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss (2026 preprint): https://arxiv.org/abs/2608.03796 . Establishes that offline sparse-target methods are serious practical controls; reported hardware gains do not transfer automatically to this setup.
- DistilGPT2 model/config: https://huggingface.co/distilbert/distilgpt2 and https://huggingface.co/distilbert/distilgpt2/blob/main/config.json . Resolve an immutable revision at implementation time.
- Breaking the Softmax Bottleneck (ICLR 2018): https://arxiv.org/abs/1711.03953 . Language-distribution expressiveness and task answer accuracy are different claims.





