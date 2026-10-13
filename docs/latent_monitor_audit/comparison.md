# Latent Monitor Audit: comparison

13 cards; 1 flag(s)

| model | pool | device | dtype | gate | carriers | placement | regime | transport long by slot | mask cost long by slot (pts) | register cost (pts) |
|---|---|---|---|---|---|---|---|---|---|---|
| codi1b | asdiv | Tesla T4 | float16 | 0.555 | [] | [] | not_evaluable | 0.02 0.02 0.00 0.12 0.00 0.00 | 6.9 0.0 0.0 3.4 0.0 0.0 | 0.0 |
| codi1b | gsm8k_aug_train | Kaggle GPU (§120 run) | float16 | — | [3] | [3] | backup_route | 0.04 0.01 0.01 0.34 0.04 0.00 | 4.2 2.7 1.5 2.7 4.6 0.8 | 3.4 |
| codi1b | gsm8k_test | Tesla T4 | float16 | 0.555 | [3] | [3] | backup_route | 0.06 0.01 0.00 0.30 0.03 0.00 | 5.7 4.1 1.9 3.8 4.3 1.1 | 4.3 |
| codi1b | gsm8k_train_original | Tesla T4 | float16 | 0.555 | [3] | [3] | backup_route | 0.06 0.02 0.00 0.35 0.03 0.00 | 9.2 7.2 3.2 2.3 4.6 0.9 | 3.2 |
| gpt2 | asdiv | Tesla T4 | float32 | 0.436 | [] | [] | not_evaluable | 0.00 0.12 0.00 0.15 0.00 0.00 | 13.8 6.9 6.9 3.4 10.3 3.4 | 13.8 |
| gpt2 | gsm8k_aug_train | Kaggle GPU (§120 run) | float32 | — | [1, 3] | [1, 3] | critical_path | 0.00 0.15 0.00 0.33 0.00 0.00 | 37.6 20.2 12.2 12.5 13.7 1.9 | 37.6 |
| gpt2 | gsm8k_aug_train | Tesla T4 | float32 | 0.436 | [1, 3] | [1, 3] | critical_path | 0.00 0.15 0.00 0.33 0.00 0.00 | 37.6 20.2 12.2 12.5 13.7 1.9 | 37.6 |
| gpt2 | gsm8k_test | Tesla T4 | float32 | 0.436 | [3] | [3] | backup_route | 0.00 0.13 0.00 0.36 0.00 0.00 | 18.7 12.2 9.2 6.5 9.8 0.8 | 18.7 |
| gpt2 | gsm8k_train_original | Tesla T4 | float32 | 0.436 | [3] | [3] | critical_path | 0.00 0.14 0.00 0.30 0.00 0.00 | 29.5 18.3 9.5 11.7 12.9 1.1 | 29.5 |
| gpt2 | svamp | Tesla T4 | float32 | 0.436 | [] | [] | no_transport | — — — — — — | — — — — — — | — |
| simcot | gsm8k_aug_train | Kaggle GPU (§120 run) | float16 | — | [1, 3] | [1, 3] | backup_route | 0.04 0.19 0.04 0.25 0.04 0.00 | 58.6 10.6 11.4 9.1 8.4 3.0 | 58.6 |
| simcot | gsm8k_test | Tesla T4 | float16 | 0.557 | [1, 3] | [1, 3] | backup_route | 0.03 0.22 0.05 0.20 0.04 0.00 | 32.5 9.8 7.0 4.3 5.7 1.9 | 32.5 |
| simcot | gsm8k_train_original | Tesla T4 | float16 | 0.557 | [1, 3] | [1, 3] | backup_route | 0.03 0.21 0.05 0.23 0.05 0.00 | 49.0 12.3 11.7 9.2 11.7 1.4 | 49.0 |

## Flags

- gpt2: regime differs across pools: gsm8k_aug_train=critical_path, gsm8k_aug_train=critical_path, gsm8k_test=backup_route, gsm8k_train_original=critical_path

## Notes

- gpt2: carrier set differs across pools: gsm8k_aug_train=[1, 3], gsm8k_aug_train=[1, 3], gsm8k_test=[3], gsm8k_train_original=[3] (threshold edge: intervals overlap at slot(s) [1])
