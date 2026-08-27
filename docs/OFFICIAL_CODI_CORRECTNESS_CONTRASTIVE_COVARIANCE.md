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

The notebook pins the package versions recorded by the official reproduction. It also
detects and removes Kaggle's incompatible optional `torchao==0.10.0`, which otherwise
causes PEFT to fail while constructing the LoRA modules. No TorchAO functionality is
used by this experiment.

The real cache is not checked into this repository, so this repository ships the
tested implementation and run-all notebook. The completed empirical outcome below was
produced by the Kaggle run and verified locally from its export.

## Completed result (2026-08-27): `not_confirmed`

Kaggle export `jonraza15/can-codis-correct-and-wrong-be-separated`, pinned at commit
`fa49050`, all 54 SHA-256 checksums intact. Both the analytic report and the
generation report recompute bit-identically from the raw export with the local
analyzers. The realized split base rates are test-like — fit 44.1% / select 42.0% /
test 40.1% correct — so the §44 population-shift threat does not apply to this
experiment. Both tiers ran; shrinkage 0.2 was selected on held-out projection-energy
specificity (correct-specificity 0.952). The generalized eigenvalue ratios span
16.3 → 5.0 (largest 28) and 0.58 → 0.20 (smallest 28).

Analytic tier — state-12 first-token accuracy on the 439 frozen test questions:

| arm | accuracy |
|---|---:|
| baseline | 0.4009 |
| classblind top-28 variance PCs | 0.3440 |
| accuracy band PCs 4–31 | 0.3212 |
| correct-only PCA rank 28 | 0.3212 |
| wrong-only PCA rank 28 | 0.3166 |
| contrastive correct retain, global centre | 0.1367 |
| **contrastive correct retain (primary)** | **0.1207** |
| contrastive wrong retain | 0.0478 |
| matched-random correct-energy retains (8) | 0.018–0.043 |
| contrastive wrong remove (secondary) | 0.4009 |
| matched-random wrong-energy removes (8) | 0.374–0.401 |

Gate outcomes:

| gate | requirement | observed | passed |
|---|---|---|---|
| vs correct-only PCA | ≥ +1.0 pt, positive lower bound | −20.05 pts, CI [−24.15, −15.95] | no |
| vs accuracy band PCs 4–31 | ≥ +1.0 pt, positive lower bound | −20.05 pts, CI [−24.15, −15.95] | no |
| vs best matched random | ≥ +1.0 pt, positive lower bound | +7.74 pts, CI [+5.01, +10.71] | yes |
| wrong removal vs baseline | ≥ +1.0 pt and above matched randoms | 0.00 pts, CI [−1.59, +1.59] | no |

The paired exact-match generation tier confirms the analytic ordering on the same 439
questions: baseline 0.4237, contrastive correct retain 0.1298 (−23.2 pts against the
accuracy band's 0.3622, CI [−27.3, −19.1]), correct-only PCA 0.3394, contrastive wrong
remove 0.4123 (−1.14 pts vs baseline, CI [−2.96, +0.68]).

Interpretation, bounded exactly as the contract requires:

- The correct/wrong covariance ratio directions are genuinely class-specific — the
  held-out projection-energy specificity is high, and retention beats every
  energy-matched random subspace by +7.7 points — but they carry only a small fraction
  of the answer-bearing content. Retaining them costs 28 points against baseline and
  20 points against either PCA control.
- Removing the wrong-specific subspace leaves first-token accuracy **identical**
  (0.4009) and slightly hurts under real decoding. Per-question it is a cancellation
  rather than inertness — 14 of 439 predictions moved, 7 gained and 7 lost, with the
  mean gold margin and NLL both shifting — so the edit has an effect but no
  direction. There is no evidence of a removable "wrong" channel at state 12.
- Correctness-conditioned covariance is therefore descriptive, not a demonstrated
  correction channel, under any fixed linear projection tested. This replicates the
  §41/§43 orthogonality conclusion on a test-like population: class-specific variance
  and answer content live in nearly disjoint parts of the space.
