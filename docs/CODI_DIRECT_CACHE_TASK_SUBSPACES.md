# Direct native-cache task-sensitive K/V experiment

This experiment corrects the hidden-to-cache projector mismatch in the preceding
task-sensitive CODI study. It learns distinct subspaces and ranks directly from
the six cached key and value vectors in every GPT-2 block.

For a centered native cache vector `x` and its exact full-answer-loss gradient
`g`, the task matrix is `M = -sym(E[x g^T])`. Its leading positive eigenvectors
maximize first-order loss damage under subspace removal. Key and value matrices
are fitted and rank-selected independently.

The data protocol uses four disjoint GSM8K-train splits: cache covariance fit,
task discovery, held-out rank selection, and causal confirmation. The causal
split compares task, covariance, a normalized task-covariance hybrid, key-only,
value-only, and energy-matched random interventions at equal K/V ranks. A paired
bootstrap interval is family-wise corrected across the 12 layers. Downstream xKV
compression remains blocked unless two adjacent layers pass both removal
specificity and retain-only fidelity gates.

Run `notebooks/kaggle_codi_direct_cache_task_subspaces.ipynb` on Kaggle with a GPU,
internet access, and the completed official CODI reproduction dataset attached.
