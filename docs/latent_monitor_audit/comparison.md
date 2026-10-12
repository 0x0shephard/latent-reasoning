# Latent Monitor Audit: comparison

4 cards; 1 flag(s)

| model | pool | device | dtype | gate | carriers | placement | regime | transport long by slot | mask cost long by slot (pts) | register cost (pts) |
|---|---|---|---|---|---|---|---|---|---|---|
| codi1b | gsm8k_aug_train | kaggle-gpu (§120 run) | float16 | — | [3] | [3] | backup_route | 0.04 0.01 0.01 0.34 0.04 0.00 | 4.2 2.7 1.5 2.7 4.6 0.8 | 3.4 |
| codi1b | gsm8k_test | Tesla T4 | float16 | 0.555 | [3] | [3] | backup_route | 0.06 0.01 0.00 0.30 0.03 0.00 | 5.7 4.1 1.9 3.8 4.3 1.1 | 4.3 |
| gpt2 | gsm8k_aug_train | kaggle-gpu (§120 run) | float32 | — | [1, 3] | [1, 3] | critical_path | 0.00 0.15 0.00 0.33 0.00 0.00 | 37.6 20.2 12.2 12.5 13.7 1.9 | 37.6 |
| simcot | gsm8k_aug_train | kaggle-gpu (§120 run) | float16 | — | [1, 3] | [1, 3] | backup_route | 0.04 0.19 0.04 0.25 0.04 0.00 | 58.6 10.6 11.4 9.1 8.4 3.0 | 58.6 |

## Flags

- codi1b: slot 2 long transport intervals do not overlap between gsm8k_aug_train and gsm8k_test
