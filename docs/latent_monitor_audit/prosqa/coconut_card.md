# ProsQA audit: coconut (connordilgren/llama32-1b-prosqa-coconut@026c321c/checkpoint_10)

- accuracy 97.6% on 500 rows (published 97.6%); twins 70 pairs; regime **no_transport**; carriers []; monitors not_evaluable: ProsQA twins share every intermediate node

| slot | transport short / long | K/V transport short / long | mask cost all / long (pts) |
|---|---|---|---|
| 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 1 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 2 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 3 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 4 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 5 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |

- tails: kv_all transport all 0.000, long 0.000; mean_all cost 0.000, meankv_all cost 0.000, mask_all cost 0.000

Readability of the prompt's names (balanced accuracy, logistic probe):

| target | classes | slot 0 | slot 1 | slot 2 | slot 3 | slot 4 | slot 5 |
|---|---|---|---|---|---|---|---|
| subject | 17 | 0.991 | 0.991 | 0.991 | 0.989 | 0.986 | 0.986 |
| option_1 | 38 | 0.254 | 0.279 | 0.267 | 0.283 | 0.267 | 0.269 |
| option_2 | 38 | 0.277 | 0.313 | 0.328 | 0.322 | 0.332 | 0.328 |
| gold_option | 38 | 0.646 | 0.657 | 0.661 | 0.656 | 0.649 | 0.626 |
| hops | 4 | 0.266 | 0.286 | 0.290 | 0.294 | 0.290 | 0.307 |

## Checks

- p25a_no_single_slot_costs_over_2_points: **True**
- p25a_all_six_cost_under_5_points: **True**
- p25b_no_single_site_transports_0_10: **True**
- p25b_tail_kv_swap_under_0_30: **True**
- p25b_regime_no_transport: **True**
- p25c_names_readable_from_every_slot_0_80: **False**
- p25d_accuracy_within_3_points_of_published: **True**
