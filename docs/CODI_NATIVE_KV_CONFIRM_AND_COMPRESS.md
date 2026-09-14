# Frozen native-KV confirmation and protected residual compression

This combined experiment uses the output of the direct native-cache discovery as
a frozen source artifact. It reconstructs and verifies all four original split
hashes, then takes the next 512 questions from the same deterministic sampling
order as a completely held-out confirmation set.

Layer 11/value-rank-1 is the preregistered primary hypothesis. Layers 2, 3 and 5
are secondary candidates with a Bonferroni interval across all four layers. Every
candidate is compared causally with four equal-rank random bases selected to match
its analytical covariance energy. Four disjoint 128-question folds audit effect
consistency. Ranks 2, 4 and 8 and the six latent-position effects are diagnostics;
they cannot change the rank-1 primary decision.

Only a passing layer-11 primary result creates an eligible single-layer protected
artifact. The same notebook then compares ordinary SVD/xKV, random protection and
causal protection on the untouched GSM8K test set. This downstream implementation
is a dense-reconstruction quality and modeled-storage proxy, not a production
latency benchmark.
