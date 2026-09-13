# CODI layerwise U28 and causal-protected xKV experiments

This pair of notebooks tests whether CODI's post-`ln_f` answer eigenspace can be
transported into earlier transformer blocks and used as a protected core in
cross-layer KV compression.

The first notebook fits layer-local ridge transports to the final 28 coordinates
on GSM8K train questions disjoint from the original U28 calibration set. It measures
held-out coordinate R², principal-angle overlap, and first-token retain/remove
effects against rank- and energy-matched random bases. It exports
`layerwise_u28.pt` only as evidence; its `gate.passed` field determines whether the
second notebook is confirmatory or merely exploratory.

The second notebook compares independent per-layer SVD, ordinary xKV-style
cross-layer SVD, random-protected xKV, and transported-U28-protected xKV. The three
xKV arms use the same total rank and factor-storage budget, while the independent
per-layer control reports its own storage. Only the contiguous layer run that passed
the first notebook's causal gate receives a protected core. For compatibility with
stock Transformers it reconstructs the cache
before decoding, so measured accuracy is valid but reported factor memory is
modelled. The included reduced-attention microbenchmark is operator-only. A genuine
end-to-end latency or allocated-memory claim requires a fused kernel that keeps the
cache factorized throughout decoding.

Run order:

1. `notebooks/kaggle_codi_layerwise_u28_stability.ipynb`
2. attach its Kaggle output dataset to
   `notebooks/kaggle_codi_causal_xkv.ipynb`
