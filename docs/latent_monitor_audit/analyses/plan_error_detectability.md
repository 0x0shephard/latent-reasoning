# Plan errors vs latent slips under the confidence gate (ledger §138)

| model | class | n | mean conf | AUROC low-conf vs correct [95%] | routed @10% | @25% | @50% | explicit chain right |
|---|---|---|---|---|---|---|---|---|
| gpt2 | correct (pool) | 297 | 0.836 | — | 0.10 | 0.25 | 0.50 | — |
| gpt2 | plan_wrong | 336 | 0.358 | 0.886 [0.86, 0.91] | 0.18 | 0.42 | 0.72 | 0.00 |
| gpt2 | latent_only | 94 | 0.447 | 0.860 [0.82, 0.90] | 0.06 | 0.22 | 0.54 | 1.00 |
| gpt2 | upstream_arithmetic | 43 | 0.325 | 0.909 [0.87, 0.95] | 0.19 | 0.40 | 0.77 | 0.00 |
| gpt2 | final_step | 21 | 0.473 | 0.838 [0.76, 0.90] | 0.05 | 0.19 | 0.52 | 0.00 |
| gpt2 | plan_wrong vs latent_only | | Δmean -0.090 | low-conf → plan_wrong 0.615 | | | | |
| codi1b | correct (pool) | 419 | 0.910 | — | 0.10 | 0.25 | 0.50 | — |
| codi1b | plan_wrong | 245 | 0.455 | 0.914 [0.89, 0.93] | 0.23 | 0.50 | 0.84 | 0.00 |
| codi1b | latent_only | 96 | 0.537 | 0.886 [0.85, 0.92] | 0.10 | 0.35 | 0.79 | 1.00 |
| codi1b | upstream_arithmetic | 36 | 0.425 | 0.916 [0.86, 0.96] | 0.22 | 0.53 | 0.89 | 0.00 |
| codi1b | final_step | 39 | 0.551 | 0.885 [0.83, 0.93] | 0.13 | 0.26 | 0.79 | 0.00 |
| codi1b | plan_wrong vs latent_only | | Δmean -0.081 | low-conf → plan_wrong 0.585 | | | | |
| simcot | correct (pool) | 421 | 0.894 | — | 0.10 | 0.25 | 0.50 | — |
| simcot | plan_wrong | 259 | 0.424 | 0.908 [0.88, 0.93] | 0.20 | 0.51 | 0.84 | 0.00 |
| simcot | latent_only | 86 | 0.510 | 0.877 [0.84, 0.91] | 0.13 | 0.34 | 0.77 | 1.00 |
| simcot | upstream_arithmetic | 46 | 0.416 | 0.927 [0.90, 0.95] | 0.20 | 0.41 | 0.87 | 0.00 |
| simcot | final_step | 26 | 0.550 | 0.859 [0.80, 0.91] | 0.19 | 0.27 | 0.73 | 0.00 |
| simcot | plan_wrong vs latent_only | | Δmean -0.085 | low-conf → plan_wrong 0.592 | | | | |
| codi3b | correct (pool) | 475 | 0.919 | — | 0.10 | 0.25 | 0.50 | — |
| codi3b | plan_wrong | 174 | 0.502 | 0.907 [0.88, 0.93] | 0.23 | 0.47 | 0.90 | 0.00 |
| codi3b | latent_only | 143 | 0.500 | 0.900 [0.87, 0.93] | 0.19 | 0.51 | 0.87 | 1.00 |
| codi3b | upstream_arithmetic | 29 | 0.416 | 0.933 [0.89, 0.97] | 0.28 | 0.69 | 0.90 | 0.03 |
| codi3b | final_step | 17 | 0.558 | 0.879 [0.80, 0.94] | 0.24 | 0.47 | 0.82 | 0.00 |
| codi3b | plan_wrong vs latent_only | | Δmean +0.002 | low-conf → plan_wrong 0.500 | | | | |
| codi8b | correct (pool) | 462 | 0.936 | — | 0.10 | 0.25 | 0.50 | — |
| codi8b | plan_wrong | 150 | 0.427 | 0.938 [0.92, 0.96] | 0.25 | 0.59 | 0.93 | 0.00 |
| codi8b | latent_only | 200 | 0.490 | 0.926 [0.90, 0.94] | 0.20 | 0.47 | 0.88 | 1.00 |
| codi8b | upstream_arithmetic | 21 | 0.447 | 0.938 [0.90, 0.97] | 0.19 | 0.57 | 0.86 | 0.00 |
| codi8b | final_step | 5 | 0.623 | 0.929 [0.89, 0.96] | 0.00 | 0.20 | 1.00 | 0.00 |
| codi8b | plan_wrong vs latent_only | | Δmean -0.063 | low-conf → plan_wrong 0.559 | | | | |
