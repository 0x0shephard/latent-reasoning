# Retrieval-based latent repair (ledger §126, Build 3)

Wrongly answered GSM8K test questions with ≥ 3-step chains; donor = bank row (2,048 training-pool rows) whose carrier readout equals the gold penultimate intermediate v*. Primary stratum: wrong targets whose own carrier readout ≠ v* and that have a donor. Rates in %, with 95% bootstrap intervals over targets.

| model | eligible | wrong | with donor | primary n | state_nearest | state_random_same | state_centroid | state_random_diff | kv_store_nearest | state_kv_nearest | kv_tail_nearest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 | 838 | 512 | 418 | 357 | 15.1 [11.5, 19.0] | 17.6 [13.7, 21.6] | 20.2 [16.0, 24.4] | 3.6 [2.0, 5.6] | 11.5 [8.4, 14.8] | 12.3 [9.2, 16.0] | 14.3 [10.6, 17.9] |
| CODI-1B | 838 | 422 | 343 | 266 | 22.6 [17.7, 27.8] | 21.8 [16.9, 27.1] | 23.7 [18.4, 28.9] | 2.6 [0.8, 4.5] | 10.2 [6.8, 13.9] | 13.2 [9.4, 17.3] | 17.3 [12.8, 22.2] |
| SIM-CoT | 838 | 418 | 335 | 273 | 15.4 [11.4, 19.8] | 9.9 [6.6, 13.6] | 13.2 [9.5, 17.2] | 2.6 [0.7, 4.4] | 4.0 [1.8, 6.6] | 4.4 [2.2, 7.0] | 11.0 [7.3, 14.7] |
| SIM-CoT, second carrier (slot 1) | | | | | 9.9 [6.2, 13.6] | | | | | | |

Harm on correctly answered controls (answer becomes wrong), %:

| model | n | state_nearest | state_random_same | state_centroid | state_random_diff | kv_store_nearest | state_kv_nearest | kv_tail_nearest |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | 123 | 18.7 [12.2, 26.0] | 20.3 [13.8, 27.6] | 17.1 [10.6, 23.6] | 62.6 [53.7, 71.5] | 19.5 [13.0, 26.8] | 19.5 [13.0, 26.8] | 56.1 [48.0, 65.0] |
| CODI-1B | 115 | 7.0 [2.6, 12.2] | 8.7 [3.5, 13.9] | 8.7 [3.5, 13.9] | 24.3 [16.5, 32.2] | 6.1 [1.7, 11.3] | 7.0 [2.6, 12.2] | 46.1 [36.5, 54.8] |
| SIM-CoT | 115 | 5.2 [1.7, 9.6] | 10.4 [5.2, 16.5] | 9.6 [4.3, 14.8] | 26.1 [18.3, 33.9] | 10.4 [5.2, 15.7] | 13.0 [7.8, 19.1] | 69.6 [60.9, 78.3] |
