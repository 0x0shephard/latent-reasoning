# ProsQA audit: codi (connordilgren/llama32-1b-prosqa-codi@78872dcd)

- accuracy 99.0% on 500 rows (published 99.0%); twins 108 pairs; regime **no_transport**; carriers []; monitors not_evaluable: ProsQA twins share every intermediate node

| slot | transport short / long | K/V transport short / long | mask cost all / long (pts) |
|---|---|---|---|
| 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 1 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 2 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 3 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 4 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |
| 5 | 0.000 / 0.000 | 0.000 / 0.000 | 0.0 / 0.0 |

- tails: kv_all transport all 0.005, long 0.007; mean_all cost 0.000, meankv_all cost 0.000, mask_all cost -

Readability of the prompt's names (balanced accuracy, logistic probe):

| target | classes | slot 0 | slot 1 | slot 2 | slot 3 | slot 4 | slot 5 |
|---|---|---|---|---|---|---|---|
| subject | 17 | 0.897 | 0.644 | 0.447 | 0.422 | 0.402 | 0.416 |
| option_1 | 38 | 0.377 | 0.347 | 0.298 | 0.305 | 0.265 | 0.269 |
| option_2 | 38 | 0.503 | 0.506 | 0.509 | 0.420 | 0.314 | 0.285 |
| gold_option | 38 | 0.404 | 0.367 | 0.331 | 0.314 | 0.325 | 0.322 |
| hops | 4 | 0.293 | 0.296 | 0.280 | 0.276 | 0.295 | 0.269 |

## Checks

- p25a_no_single_slot_costs_over_2_points: **True**
- p25a_all_six_cost_under_5_points: **True**
- p25b_no_single_site_transports_0_10: **True**
- p25b_tail_kv_swap_under_0_30: **True**
- p25b_regime_no_transport: **True**
- p25c_names_readable_from_every_slot_0_80: **False**
- p25d_accuracy_within_3_points_of_published: **True**
