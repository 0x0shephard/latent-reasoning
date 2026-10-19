# Confident latent slips: can the thoughts see them? (ledger §139)

AUROC vs correct rows; 'conf' = confident half (first-token probability ≥ pool median). Capture = share of confident positives caught at 10% / 20% false positives.

| model | class | n all / conf / top | confidence (all / conf / top) | transition residual (all / conf / top) | supervised states (all / conf / top) | conf+residual (conf / top) | conf+states (conf / top) | capture @10/20% (residual, conf) |
|---|---|---|---|---|---|---|---|---|
| gpt2 | latent_only | 94 / 43 / 3 | 0.86 / 0.89 / — | 0.65 / 0.65 / — | 0.79 / 0.77 / — | 0.79 / — | 0.86 / — | 0.16 / 0.40 |
| gpt2 | plan_wrong | 336 / 93 / 23 | 0.89 / 0.80 / 0.71 | 0.68 / 0.67 / 0.67 | 0.89 / 0.82 / 0.77 | 0.76 / 0.72 | 0.85 / 0.80 | 0.27 / 0.41 |
| codi1b | latent_only | 96 / 20 / 4 | 0.89 / 0.79 / — | 0.81 / 0.80 / — | 0.84 / 0.86 / — | 0.83 / — | 0.88 / — | 0.45 / 0.55 |
| codi1b | plan_wrong | 245 / 40 / 5 | 0.91 / 0.81 / 0.87 | 0.83 / 0.74 / 0.60 | 0.90 / 0.83 / 0.85 | 0.80 / 0.74 | 0.86 / 0.90 | 0.38 / 0.45 |
| simcot | latent_only | 86 / 20 / 3 | 0.88 / 0.82 / — | 0.79 / 0.86 / — | 0.82 / 0.86 / — | 0.90 / — | 0.88 / — | 0.45 / 0.75 |
| simcot | plan_wrong | 259 / 41 / 4 | 0.91 / 0.82 / — | 0.77 / 0.71 / — | 0.89 / 0.81 / — | 0.78 / — | 0.85 / — | 0.20 / 0.39 |
| codi3b | latent_only | 143 / 18 / 3 | 0.90 / 0.73 / — | 0.75 / 0.68 / — | 0.85 / 0.75 / — | 0.74 / — | 0.79 / — | 0.06 / 0.28 |
| codi3b | plan_wrong | 174 / 18 / 1 | 0.91 / 0.78 / — | 0.73 / 0.59 / — | 0.90 / 0.78 / — | 0.70 / — | 0.82 / — | 0.33 / 0.33 |
| codi8b | latent_only | 200 / 25 / 3 | 0.93 / 0.84 / — | 0.79 / 0.78 / — | 0.85 / 0.81 / — | 0.85 / — | 0.85 / — | 0.48 / 0.56 |
| codi8b | plan_wrong | 150 / 11 / 1 | 0.94 / 0.73 / — | 0.79 / 0.48 / — | 0.91 / 0.71 / — | 0.59 / — | 0.75 / — | 0.09 / 0.18 |
