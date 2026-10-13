# Latent Monitor Audit card: codi1b on `asdiv`

- model: `configs/official_codi_llama1b.yaml` (checkpoint sha `c6d4b4b096f0`, 16 layers)
- pool: asdiv · n = 1490 · hash `b5fde33de6fa` · steps {'1': 1044, '2': 360, '3': 52, '4': 14, '5': 6, '6': 10, '7': 3, '8': 1}
- environment: Tesla T4 · float16 · batch 32 · torch 2.11.0+cu128 · transformers 4.52.4
- gate: accuracy 0.5550 vs published 0.5560 (passed)

| slot | role | transport short | transport long [95% CI] | K/V short | K/V long | readability ridge / mlp | faithfulness ridge / mlp | mask cost short (pts) | mask cost long (pts) [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| 0 | inert | 0.007 | 0.018 [0.00, 0.05] | 0.036 | 0.000 | 0.16 / 0.07 | 0.00 / 0.00 | 0.8 | 6.9 [0.0, 17.2] |
| 1 | inert | 0.000 | 0.018 [0.00, 0.05] | 0.007 | 0.018 | 0.00 / 0.00 | — / — | 0.2 | 0.0 [0.0, 0.0] |
| 2 | inert | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.018 | 0.04 / 0.04 | 0.00 / 0.00 | 0.4 | 0.0 [0.0, 0.0] |
| 3 | weak | 0.011 | 0.125 [0.04, 0.23] | 0.000 | 0.000 | 0.18 / 0.14 | 0.20 / 0.12 | 0.8 | 3.4 [0.0, 10.3] |
| 4 | weak | 0.001 | 0.000 [0.00, 0.00] | 0.010 | 0.125 | 0.18 / 0.14 | 0.00 / 0.00 | 0.2 | 0.0 [0.0, 0.0] |
| 5 | terminal_inert | 0.000 | 0.000 [0.00, 0.00] | 0.000 | 0.000 | 0.16 / 0.11 | 0.00 / 0.00 | 0.2 | 0.0 [0.0, 0.0] |

- position-0 K/V (question register): transport short 0.036, long 0.000; mask cost long 0.0 pts
- tails: kv_all transport all 0.141 (other 0.001), long 0.339; mask_all cost long 31.0 pts, mean_all 6.9 pts

## Checks

- c1_no_single_site_transport_on_short_chains: **True**
- c2_chain_length_gate: **None**
- c3_terminal_inert_unfaithful_unnecessary: **True**
- c4_readability_dissociated_from_faithfulness: **False**
- c4_mlp_readability_dissociated_from_faithfulness: **False**
- c4b_readable_unfaithful_slot_exists: **False**
- c4b_mlp_readable_unfaithful_slot_exists: **False**
- c4b_both_families: **False** (slots [])
- c5_carriers_necessary: **None**
- register_position_0: **False**
- details: max short transport 0.011, max long transport —, most readable slot ridge 4 / mlp 4, spearman ridge 0.56 / mlp 0.54, carrier costs {}, register cost 0.0 pts

## Verdict

- regime: **not_evaluable**
- carrying slots: []
- placement slots: []
- store slots (K/V carries, state does not): []
- inert slots: [0, 1, 2, 5] (readable but inert: [])
- Long profile too thin (28 gated pairs, fewer than 50): carriers, faithfulness and necessity on long chains are not decided here; short-chain results stand.

- stage seconds: {'twins_standard': 1082.3, 'twins_long': 117.3, 'masking': 274.9}
