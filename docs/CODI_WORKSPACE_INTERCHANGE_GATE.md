# Workspace interchange go/no-go

Ledger §105. Contract `official_codi_workspace_interchange_gate_v1`. No training.

Before training a latent student by interchange intervention on its thought slots,
the official model must show (1) that swapping a slot's state between two questions
changes the answer, through a low-rank subspace; (2) that the effect is specific to
the odd, value-holding slots (§55); and (3) that a gradient score over the slot's
principal directions does *not* already rank them as the interventions do, otherwise an
intervention-selected trajectory target cannot differ from a gradient-selected one.

Instrument: the first answer token under the forced cue after a teacher-forced latent
path with the interchange applied at one slot (cheap; used for every rank and the
per-direction scans). Native greedy decoding is reported for the odd slots at full rank
and rank 64. Questions: 512 rows of the §86 fit split; pairs by seeded derangement.

| gate | rule |
|---|---|
| M1 | full-slot swap at each odd slot changes the first token for ≥ 30% of pairs |
| M2 | at each odd slot some rank ≤ 64 reaches ≥ 50% of the full-slot change rate |
| M3 | mean odd change rate ≥ 1.5× mean even change rate, at rank 64 and full |
| M4 | Spearman(gradient score, per-PC interchange effect) ≤ 0.7 at each odd slot |

GO = all four. Reported: directedness (changed answers equal to the donor's own
answer, against the derangement null), change-rate curves, top-8 directions by each
criterion. Notebook `notebooks/kaggle_codi_workspace_interchange_gate.ipynb`.
