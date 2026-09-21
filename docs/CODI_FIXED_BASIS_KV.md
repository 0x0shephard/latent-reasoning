# Fixed per-head K/V subspaces for CODI

## Question

Ledger §40 and §67 established that CODI's final hidden state has a *fixed*
low-dimensional answer subspace: one basis, fitted once on training questions,
that every question reuses. This experiment asks the same question one level
down, inside the KV cache:

> Does each transformer block's attention head write its keys and values into a
> fixed low-dimensional subspace that is stable across questions, and can CODI
> run at full accuracy when every cached vector is projected onto it?

This is deliberately not the xKV question. xKV (§75–§81) factorizes *each
request's* cache matrix, so every request stores both a token factor and its own
decoder. Per layer at rank `r`, that is `r*(tokens + 1536)` units against a dense
`tokens*1536`. A fixed basis stores no decoder at all, only `r` coordinates per
head per token, so per layer it costs `24r` units per token regardless of length.

| scheme | units per token, one layer | ratio at 100 tokens |
| --- | --- | ---: |
| dense | 1,536 | 1.00x |
| per-request SVD, rank 64 | 64 + 64*1536/tokens | 1.47x |
| fixed per-head basis, rank 32 | 768 | 2.00x |
| fixed per-head basis, rank 16 | 384 | 4.00x |
| fixed per-head basis, rank 8 | 192 | 8.00x |

The per-request row reproduces the 1.48x that §79 actually measured. The fixed-basis
ratio is `head_dim / rank` and does not decay at short context, which is the whole
reason to ask this question at CODI's scale.

## Method

For every (layer, kind, head) triple the calibration pass accumulates the
uncentred second moment of the cached vectors that head writes:

```text
M = sum_t  x_t x_t^T ,   x_t in R^64
```

Its eigenvectors, ordered by descending eigenvalue, are that component's fixed
basis `U`. At evaluation the projection `x -> U_r U_r^T x` is applied to every
key and value as the block writes it.

The projection is exact, not an approximation of a runtime. GPT-2 uses learned
absolute positions, so no rotary embedding sits between the projection and the
attention dot product, and

```text
q . (U_r U_r^T k)        = (U_r^T q) . (U_r^T k)
sum_i p_i U_r U_r^T v_i  = U_r (sum_i p_i U_r^T v_i)
```

Therefore the arm this experiment measures *is* an `r`-dimensional per-head
attention, executed unfused. The bases are model metadata; per-token cache cost
falls from 64 to `r` units per component.

Bases are per head. A 768-wide basis would mix twelve heads and could not fold
into per-head dot products, so the honest object is the 64-dimensional head
slice.

## Calibration and splits

The calibration pass runs the released forced-cue generation path unchanged and
records which row type produced each cached vector: `question` (prompt tokens
plus the `<bot>` marker), `latent` (the six continuous thoughts), `cue` (the
forced `The answer is:` tokens), and `answer` (greedily decoded tokens to EOS).
The fitted basis pools all four; the per-row-type moments are kept so the
diagnostic can ask whether the §55 workspace rows occupy a different subspace
than the question rows.

| split | rows | use |
| --- | ---: | --- |
| basis fit | 1,024 GSM8K train | second moments only |
| selection | 256 GSM8K train | rank grid, operating point |
| final | 1,319 GSM8K test | one locked read |

Sampling uses the repository's deterministic GSM8K-train sampler with seed
20260921 and its train/test disjointness proof. No test question enters fitting
or selection.

## Arms

Rank grid `8, 16, 24, 32, 40, 48` against a head width of 64.

- `uniform_r{r}`: every component gets rank `r`.
- `energy_r{r}`: the same total budget, allocated greedily by eigenvalue across
  all 24 layer-by-kind groups of heads. Eigenvalues are non-increasing, so the
  greedy allocation is exactly optimal for retained energy.
- `random_s{seed}_r{r}`: seeded random orthonormal bases at identical rank and
  identical storage. This is the control that matters; §61 showed the same
  comparison at the output head, where eigen-initialization beat random by 18
  points at equal rank.
- `key_only_r{r}` and `value_only_r{r}`: only keys or only values compressed, to
  separate the two. §79's utility curves suggested keys carry more answer
  sensitivity than values.

Selection and final arms both report greedy GSM8K exact match, dense first-token
top-1 agreement, exact sequence agreement, answer NLL, and modeled storage.

## Decision rules, frozen before any run

The operating point is chosen on the 256-question selection split only: the
**smallest** rank in the grid, preferring `uniform` over `energy` at equal rank,
whose arm retains at least 98% of dense exact match and at least 95% dense
first-token agreement. If no rank passes, the run stops and the test set is not
read, as in the sibling experiments whose screens failed (§77, §78).

The locked final gate, applied once on the full test set, requires all four:

1. paired exact-match difference versus dense has a 95% lower bound above
   −2 points;
2. the selected arm beats **every** seeded random basis at equal rank with a
   positive paired lower bound;
3. dense first-token top-1 agreement at least 95%;
4. point retention at least 98% of dense exact match.

The primary outcome is paired exact match, not answer NLL. Ledger §82 records
why: the §75–§81 chain gated on NLL and KL differences of a few thousandths,
which at n = 551 needed roughly 1,300 paired questions to detect.

## What this can and cannot establish

It can establish whether CODI's per-head cache geometry is fixed across
questions, whether that geometry is specifically useful rather than any
orthonormal basis of the same size, whether keys and values differ, and whether
the latent workspace rows occupy a different subspace than question rows.

It does not measure wall clock or allocated memory. The hook is the exact
unfused reference for the projected model; a latency claim needs a fused
`r`-dimensional attention kernel. Twelve layers of a hundred-token cache is a few
megabytes, so the value of a positive result here is mechanistic, and any
deployment claim would have to be made on a longer-context model.

Bounded to the frozen official CODI GPT-2 checkpoint, GSM8K, six latent
positions, and the forced-cue protocol.

## Inputs and outputs

Attach the completed official CODI reproduction dataset containing
`official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json`. The
checkpoint downloads from the pinned source. No predecessor artifact from the
xKV chain is required; this experiment does not depend on any of it.

Run `notebooks/kaggle_codi_fixed_basis_kv.ipynb` on a Kaggle GPU with Internet
enabled. It writes `summary.json`, `fixed_basis_kv.pt` (second moments, bases,
eigenvalues, per-arm correctness vectors), and `predictions.jsonl` under
`/kaggle/working/codi_fixed_basis_kv`.

## Files

- implementation: `src/mech/fixed_basis_kv.py`
- runner: `scripts/run_codi_fixed_basis_kv.py`
- notebook builder: `scripts/build_kaggle_codi_fixed_basis_kv_notebook.py`
- notebook: `notebooks/kaggle_codi_fixed_basis_kv.ipynb`
- tests: `tests/test_fixed_basis_kv.py`, `tests/test_fixed_basis_kv_runner.py`,
  `tests/test_codi_fixed_basis_kv_notebook.py`
