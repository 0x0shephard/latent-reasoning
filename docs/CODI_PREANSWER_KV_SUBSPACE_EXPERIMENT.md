# CODI direct pre-answer KV subspace experiment

The earlier direct-layer notebook captured `ln_1` outputs and attempted to
differentiate the answer loss with respect to those hook outputs. Under the active
Transformers cache path those tensors were disconnected from the answer loss. The
reported rank-28 lists were therefore forced candidates, not validated directions.

The corrected experiment uses the exact key and value tensor objects passed as the
pre-answer cache. It fits an independent covariance eigensystem at every block's
`ln_1` input, maps each candidate through the effective LoRA-aware key and value
projections, and scores predicted removal damage using direct cache gradients.

Selection is variable-rank. A direction must be positive in both deterministic
selection halves, exceed an example-shuffled null, and survive global FDR correction.
A separate rank-selection split must confirm positive effects. The operational rank
is the smallest ranked prefix retaining 95% of that disjoint validation effect, with
a rank-64 ceiling.

Causal confirmation removes and retains the selected K/V subspace at all six latent
positions and compares removal damage with rank- and energy-matched random controls.
Compression is eligible only if at least two adjacent layers pass the causal gate.

Run `notebooks/kaggle_codi_preanswer_kv_subspaces.ipynb` on Kaggle with the completed
official CODI reproduction dataset attached. The required input suffix is
`official_codi_gpt2/eval/revision_fd641b3d/full_gsm8k/summary.json`.
