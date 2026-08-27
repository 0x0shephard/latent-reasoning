# Correct versus wrong covariance intervention

## Question

Can the 768-dimensional state-12 answer-cue representation be split into directions
that vary specifically when CODI is correct and directions that vary specifically
when it is wrong—and does projecting onto or away from those directions change the
answer?

This is distinct from the completed 28-PC accuracy-band result. PCs 4–31 were found by
class-blind total covariance. Correct-only PCA was also previously tested and was
nearly identical to class-blind PCA. The new estimator explicitly contrasts the two
class covariances:

```text
C_correct v = lambda C_wrong v
```

The 28 largest-ratio generalized eigenvectors define the correct-specific candidate;
the 28 smallest define the wrong-specific candidate. After solving, each candidate is
QR-orthonormalized because the intervention is a Euclidean projection.

## Leakage-safe design

The cached 1,319 GSM8K-test state-12 vectors are deterministically partitioned with
seed `20260827`:

| split | count | use |
|---|---:|---|
| fit | 440 | class means, covariances, and all bases |
| select | 440 | covariance shrinkage via held-out projection-energy specificity |
| test | 439 | one final read of accuracy, gold NLL, and gold margin |

No test outcome selects a rank, shrinkage, centre, or direction. Rank is fixed at 28
from the question being tested. This is still a corrective experiment on a dataset
previously inspected by the project, not a pristine preregistration.

## Interventions and controls

The primary arm keeps only the correct-specific subspace around the correct-class
mean:

```text
h' = mu_correct + U_correct U_correct^T (h - mu_correct)
```

The secondary arm removes the wrong-specific component around the global fit mean.
Required controls are correct-only PCA retention, retention of the established
class-blind accuracy band (PCs 4–31), and eight rank- and class-energy-matched random
bases. The top-28 variance PCs are also reported descriptively. A global-centre
diagnostic separates the effect of the basis from the effect of replacing the discarded
component with the correct-class mean.

The primary result passes only if correct-specific retention beats both required PCA controls
and the best matched-random arm by at least 1.0 accuracy point, with positive paired
bootstrap lower bounds. Wrong-specific removal has a separate 1.0-point gate versus
baseline and must beat its best matched-random arm. Both gates are required for the
strong conclusion that the two covariance classes expose a correction channel.

## What the result can mean

- A pass means the covariance ratio found a compact state-12 channel that is more
  answer-sufficient than ordinary or correct-only PCA, and deleting the opposite
  channel improves answers.
- A correct-retain pass alone means a useful sufficient representation was found; it
  does not show that a wrong answer can be repaired by a universal projection.
- A failure means correctness-conditioned covariance is descriptive but is not a
  demonstrated causal correction mechanism under this fixed linear intervention.
- Since state 12 is the final layer-normalized answer-cue state, the hook directly
  changes the first answer token. Later generated tokens may change downstream of that
  token, but the edited state itself does not enter the transformer KV cache.

## Run

The dedicated notebook is
[`kaggle_official_codi_correctness_contrastive_covariance.ipynb`](../notebooks/kaggle_official_codi_correctness_contrastive_covariance.ipynb).
Attach `colon_states.pt` and `readout.pt`. The analytic tier is CPU-capable. For the
optional paired exact-match tier, enable a Kaggle GPU and also attach the completed
official reproduction summary (or set its path explicitly).

The real cache is not checked into this repository, so this repository ships the
tested implementation and run-all notebook, not a claimed empirical outcome.
