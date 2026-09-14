# CODI task-sensitive KV subspace experiment

The exact-cache-gradient experiment established full graph connectivity but found
that no *individual covariance eigenvector* survived split stability and global FDR
correction. That result does not exclude a useful linear combination of covariance
directions.

For every layer and latent position, this experiment maps the exact pre-answer key
and value gradients back into the 768-dimensional `ln_1` hidden space:

`g_h = W_K^T g_K + W_V^T g_V`.

For centred state `x`, removing an orthogonal subspace with projector `P` has the
first-order loss change `-g_h^T P x`. The symmetric task matrix

`M = -0.5 * E[x g_h^T + g_h x^T]`

therefore has the property that its leading positive eigenvectors maximize expected
predicted removal damage. These directions may be arbitrary combinations of ordinary
covariance eigenvectors.

The discovery split fits the ordered task basis. A disjoint rank-selection split
chooses among ranks `1,2,4,8,16,28,32,48,64` using positive-spectrum coverage,
half-split subspace overlap, and a bootstrap interval against shuffled gradients. A
third untouched split causally removes and retains the selected K/V subspace.

Causal baselines are top covariance directions, key-only and value-only task bases,
and rank- and covariance-energy-matched random K/V bases. Protected-core xKV remains
blocked unless at least two adjacent layers pass the causal gate.
