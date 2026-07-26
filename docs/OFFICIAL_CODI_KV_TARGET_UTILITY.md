# Official CODI hierarchical KV target-utility screen

## Question

The completed low-rank experiments found reproducible teacher-predictive KV structure
but did not establish that the learned rank-four directions were more answer-causal
than matched random directions.

Before applying another denoising method, this experiment asks:

> Under matched update magnitude and loss weight, which teacher KV target families
> produce a reproducible reduction in held-out mathematical answer loss when their
> distillation gradient is included?

The screen categorizes **where useful supervision enters optimization**. It does not
yet select individual spectral directions or train a new model.

## Why raw distillation loss is not the outcome

Removing a non-negative target term necessarily lowers the displayed total objective.
At a frozen checkpoint, disabling the term also cannot change answer loss until an
update is taken.

The experiment therefore compares the consequence of three parameter updates:

```text
no target:
    answer-loss gradient

candidate:
    answer-loss gradient + correctly paired target gradient

shuffled control:
    answer-loss gradient + shuffled-pairing target gradient
```

Every combined update is normalized to the same parameter L2 norm. The checkpoint is
evaluated on a disjoint batch after each stateless update.

## Fixed model and data contract

- author-released CODI GPT-2 checkpoint
- checkpoint revision `fd641b3d3edc`
- official checkpoint must first pass the 1,319-example GSM8K reproduction gate
- equation-style augmented training examples are used for discovery
- one eligible row is selected per normalized question
- discovery and validation question groups are disjoint
- official six-step latent student path
- R-KV selects six explicit-teacher trace targets
- keys and values are separate target families
- the official `The answer is:` cue is teacher-forced context
- answer loss scores only the example-specific numeric answer tokens and EOS, preventing
  the fixed cue from dominating the utility ranking
- only checkpoint parameters marked trainable by the released LoRA/projection
  architecture receive virtual updates

The virtual updates use `torch.func.functional_call`; files and in-memory checkpoint
parameters are not overwritten.

## Hierarchical execution

Do not start with every layer, head, and direction.

### Level 1 — KV kind

Compare all key targets with all value targets:

```bash
python -u scripts/run_official_codi_kv_target_utility.py \
  --config configs/official_codi_gpt2.yaml \
  --reproduction-summary /path/to/full_gsm8k/summary.json \
  --output-dir /path/to/official_codi_kv_target_utility/kind_seed3 \
  --examples-per-split 128 \
  --batch-size 4 \
  --granularity kind \
  --kinds key,value \
  --positions 0,1,2,3,4,5 \
  --precision float32 \
  --device cuda
```

### Level 2 — latent position

Run this only for kinds that show positive kind-level utility. For example:

```bash
python -u scripts/run_official_codi_kv_target_utility.py \
  --config configs/official_codi_gpt2.yaml \
  --reproduction-summary /path/to/full_gsm8k/summary.json \
  --output-dir /path/to/official_codi_kv_target_utility/key_position_seed3 \
  --examples-per-split 128 \
  --batch-size 4 \
  --granularity position \
  --kinds key \
  --positions 0,1,2,3,4,5 \
  --precision float32 \
  --device cuda
```

### Level 3 — layer band

Run only selected kind-position branches. The default layer bands are:

```text
early  = layers 0–3
middle = layers 4–7
late   = layers 8–11
```

Example:

```bash
python -u scripts/run_official_codi_kv_target_utility.py \
  --config configs/official_codi_gpt2.yaml \
  --reproduction-summary /path/to/full_gsm8k/summary.json \
  --output-dir /path/to/official_codi_kv_target_utility/key_p4_bands_seed3 \
  --examples-per-split 128 \
  --batch-size 4 \
  --granularity layer_band \
  --kinds key \
  --positions 4 \
  --precision float32 \
  --device cuda
```

Use a distinct output directory for every hierarchy level and selection. The runner
refuses to merge incompatible requests.

## Smoke test

Before a full screen:

```bash
python -u scripts/run_official_codi_kv_target_utility.py \
  --config configs/official_codi_gpt2.yaml \
  --reproduction-summary /path/to/full_gsm8k/summary.json \
  --output-dir /path/to/official_codi_kv_target_utility/smoke_kind \
  --examples-per-split 8 \
  --batch-size 4 \
  --granularity kind \
  --precision float32 \
  --device cuda
```

The smoke result is diagnostic only.

## Metrics

For each target family the report includes:

- correctly paired distillation loss
- shuffled-pairing distillation loss
- gradient dot product and cosine with held-out answer loss
- answer loss before an update
- answer loss after the answer-only update
- answer loss after the correctly paired target update
- answer loss after the shuffled-target update
- paired batch-cluster bootstrap intervals; the update batch is the resampling unit
  because validation examples within one batch share a virtual parameter update

Utility is signed so that positive is better:

```text
candidate utility versus no target
    = answer loss after no-target update
      - answer loss after candidate update

candidate utility versus shuffled
    = answer loss after shuffled-target update
      - answer loss after candidate update
```

## Discovery classification

A family is classified as `helpful_target_family` when:

1. the paired 95 percent interval is positive versus no target
2. the paired 95 percent interval is positive versus shuffled targets
3. the median held-out gradient cosine is positive

It is `interfering_target_family` when both intervals and the median gradient cosine
are negative. Other outcomes are `neutral_or_inconclusive_target_family`.

These labels belong to a short-horizon discovery screen. They are not confirmatory
method-level claims.

## Resume and outputs

Each discovery/validation batch is written atomically:

```text
output_dir/
  run_manifest.json
  batches/
    batch_000000.json
    ...
  summary.json
  report.md
```

Rerunning the exact command verifies and skips completed batches. A changed checkpoint,
sample split, granularity, target selection, loss, update norm, or precision requires a
different output directory.

## Decision rule

- If neither key nor value targets show positive kind-level utility, stop this target
  definition and do not fit another TSV-style decomposition.
- Refine only target families that pass the preceding hierarchy level.
- After a target family survives the hierarchy, apply answer-conditioned spectral
  decomposition within that family.
- Require exact held-out intervention evidence before beginning distillation training.

## Interpretation boundary

Positive one-step utility means a teacher target supplies a locally helpful update under
this checkpoint, answer objective, update norm, and target alignment. It does not prove
that the target is answer-causal, human-interpretable, or beneficial after long
training. Those require separate causal and compute-matched training gates.
