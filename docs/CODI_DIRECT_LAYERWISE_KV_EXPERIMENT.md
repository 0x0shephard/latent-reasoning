# Direct layerwise answer subspaces and latent-KV compression

This is the corrected experiment for the question: can each transformer layer's
own answer-sensitive state subspace be discovered and then used to protect useful
information during KV-cache compression?

The discovery variable is the normalized attention input at every layer and all six
CODI latent passes. Each layer receives an independent covariance eigendecomposition.
Eigenvectors are ranked on a disjoint split by their excess absolute first-order
gold-answer-NLL effect over example-shuffled gradients. The top 28 are mapped through
the effective LoRA-aware key and value projections.

The causal test edits the newly appended latent K/V entry after each recurrent pass,
so all six cached latent positions are affected and later recurrence consumes the
edited cache. Learned removal is compared with four rank- and energy-matched random
controls using paired gold-token NLL differences and bootstrap intervals. Retain-only
top-1 agreement measures sufficiency.

Only a contiguous run of at least two layers that passes direction stability, causal
specificity and retention gates can seed the protected xKV-style comparison. That
comparison gives ordinary and protected cross-layer factors the same total rank and
factor-storage budget. Stock Transformers reconstructs the cache for quality testing;
actual factor memory and end-to-end latency remain future kernel work.
