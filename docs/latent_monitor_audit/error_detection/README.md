# Error detection from the carrier thoughts (ledger §125, Build 2)

Full GSM8K test set, native decoding, logistic classifiers (5 folds) for "the native answer is wrong"; AUROC with 95% bootstrap intervals. Slots chosen by each model's `gsm8k_test` card.

| model | accuracy | wrong | carriers | inert | confidence | carriers | inert | all slots | raw states | carriers + confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 | 0.434 | 746 | [3] | [5] | 0.889 [0.871, 0.907] | 0.850 [0.829, 0.871] | 0.853 [0.831, 0.872] | 0.851 [0.829, 0.872] | 0.859 [0.839, 0.880] | 0.897 [0.879, 0.914] |
| CODI-1B | 0.555 | 587 | [3] | [2, 5] | 0.912 [0.895, 0.925] | 0.821 [0.796, 0.844] | 0.596 [0.567, 0.627] | 0.827 [0.802, 0.850] | 0.835 [0.811, 0.856] | 0.915 [0.899, 0.929] |
| SIM-CoT | 0.557 | 584 | [1, 3] | [5] | 0.913 [0.897, 0.928] | 0.754 [0.727, 0.779] | 0.650 [0.621, 0.678] | 0.799 [0.776, 0.822] | 0.836 [0.814, 0.858] | 0.915 [0.899, 0.929] |

Single-feature AUROC of the readout entropy per slot:

| model | slot 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| GPT-2 | 0.62 | 0.80 | 0.68 | 0.86 | 0.72 | 0.86 |
| CODI-1B | 0.74 | 0.54 | 0.54 | 0.84 | 0.79 | 0.59 |
| SIM-CoT | 0.51 | 0.75 | 0.55 | 0.69 | 0.63 | 0.64 |
