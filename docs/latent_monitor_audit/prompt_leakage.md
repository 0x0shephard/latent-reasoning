# Prompt-leakage probes (ledger §125, Build 1)

Balanced accuracy (mean per-class recall) of a probe on each thought's state decoding a question attribute, standard profile (mixed chain lengths), ridge / MLP, pair-aligned 5 folds. 
Balanced baseline: 0.25 for the first operator, 0.50 for long chain. Carriers and the most readable computed-value slot come from the model's card on the same pool.

| model | pool | attribute | slot 0 | 1 | 2 | 3 | 4 | 5 | best slot | carriers | best slot for the computed value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 | gsm8k_aug_train | first_operator | 0.63 / 0.76 | 0.55 / 0.66 | 0.63 / 0.74 | 0.60 / 0.67 | 0.62 / 0.76 | 0.60 / 0.68 | 4 | [1, 3] | 4 |
| GPT-2 | gsm8k_aug_train | long_chain | 0.81 / 0.91 | 0.81 / 0.86 | 0.84 / 0.92 | 0.79 / 0.84 | 0.85 / 0.91 | 0.80 / 0.84 | 2 | [1, 3] | 4 |
| GPT-2 | gsm8k_test | first_operator | 0.65 / 0.72 | 0.54 / 0.60 | 0.58 / 0.59 | 0.53 / 0.52 | 0.53 / 0.58 | 0.49 / 0.50 | 0 | [3] | 4 |
| GPT-2 | gsm8k_test | long_chain | 0.79 / 0.86 | 0.75 / 0.79 | 0.77 / 0.83 | 0.76 / 0.76 | 0.81 / 0.82 | 0.79 / 0.76 | 0 | [3] | 4 |
| CODI-1B | gsm8k_aug_train | first_operator | 0.85 / 0.86 | 0.85 / 0.87 | 0.81 / 0.84 | 0.72 / 0.72 | 0.74 / 0.73 | 0.74 / 0.75 | 1 | [3] | 4 |
| CODI-1B | gsm8k_aug_train | long_chain | 0.89 / 0.91 | 0.92 / 0.93 | 0.91 / 0.94 | 0.91 / 0.91 | 0.90 / 0.92 | 0.92 / 0.92 | 2 | [3] | 4 |
| CODI-1B | gsm8k_test | first_operator | 0.76 / 0.76 | 0.77 / 0.72 | 0.73 / 0.73 | 0.56 / 0.56 | 0.62 / 0.61 | 0.65 / 0.64 | 1 | [3] | 4 |
| CODI-1B | gsm8k_test | long_chain | 0.87 / 0.88 | 0.85 / 0.89 | 0.86 / 0.89 | 0.86 / 0.87 | 0.86 / 0.85 | 0.85 / 0.86 | 2 | [3] | 4 |
| SIM-CoT | gsm8k_aug_train | first_operator | 0.84 / 0.80 | 0.71 / 0.71 | 0.73 / 0.74 | 0.74 / 0.73 | 0.74 / 0.75 | 0.72 / 0.72 | 0 | [1, 3] | 4 |
| SIM-CoT | gsm8k_aug_train | long_chain | 0.93 / 0.95 | 0.93 / 0.94 | 0.93 / 0.95 | 0.93 / 0.93 | 0.92 / 0.94 | 0.91 / 0.93 | 0 | [1, 3] | 4 |
| SIM-CoT | gsm8k_test | first_operator | 0.77 / 0.71 | 0.54 / 0.53 | 0.61 / 0.61 | 0.54 / 0.54 | 0.56 / 0.59 | 0.61 / 0.60 | 0 | [1, 3] | 3 |
| SIM-CoT | gsm8k_test | long_chain | 0.89 / 0.89 | 0.85 / 0.87 | 0.87 / 0.88 | 0.87 / 0.88 | 0.87 / 0.88 | 0.86 / 0.89 | 0 | [1, 3] | 3 |

Long-profile runs (three or more steps only) decode the operator less well (0.28–0.81) with the same slot ordering; `long_chain` is one class there by construction.
