# Recent MEC Experiment Results

Generated from local `eval_outputs` summary files on 2026-07-01.
This file only lists experiment settings and numerical results; it does not include interpretation or next-step analysis.

Common held-out split where applicable: `mec_eval_seed=100000`, `seed_stride=13`, `episodes=24`, `episode_horizon=350`.

## Reset-Permutation 1.5M Main Comparisons

| variant | seeds | cost/slot mean | accept mean | W1 mean | overflow mean | HAP-freeze mean | source |
|---|---|---|---|---|---|---|---|
| Mean actor + Mean critic | 1/2/3 | 2.0991 | 75.07% | 684.6 | 9.96 | 59.19% | resetperm_mean_baseline.summary.json |
| Flat actor + Flat critic | 1/2/3 | 2.7326 | 66.55% | 800.6 | 21.42 | 12.90% | resetperm_combined_summary.json |
| Latent Slot-EqDec actor + Flat critic | 1/2/3 | 3.1458 | 60.54% | 972.8 | 16.70 | 8.09% | resetperm_combined_summary.json |
| Latent Slot-EqDec actor + invariant Set critic | 1/2/3 | 4.3119 | 44.87% | 1145.4 | 23.93 | -0.74% | resetperm_combined_summary.json |
| Flat descriptor actor + Flat critic, dim 256 | 1/2/3 | 4.4281 | 42.24% | 1212.0 | 6.46 | 0.32% | resetperm_flat_descriptor.summary.json |

## Reset-Permutation Mean Baseline, Per Seed

| variant | experiment | seed | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean actor + Mean critic, reset permutation seed 1 | resetperm1500_mean_baseline_mean_mean_seed1 | 1 | 1108800 | -720.20 | 2.1286 | 74.63% | 700.9 | 9.38 | 665.91 | 59.10% |
| Mean actor + Mean critic, reset permutation seed 2 | resetperm1500_mean_baseline_mean_mean_seed2 | 2 | 1209600 | -686.99 | 2.0832 | 75.58% | 660.6 | 16.43 | 641.10 | 73.36% |
| Mean actor + Mean critic, reset permutation seed 3 | resetperm1500_mean_baseline_mean_mean_seed3 | 3 | 1310400 | -705.70 | 2.0856 | 74.99% | 692.4 | 4.06 | 656.52 | 45.12% |

## Reset-Permutation Actor Baselines and Latent Slot Runs, Per Seed

| variant | experiment | seed | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|---|
| Flat actor + Flat critic, reset permutation seed 1 | resetperm1500_actor_flat_flat_seed1 | 1 | 1495200 | -910.70 | 2.6825 | 66.77% | 818.8 | 15.74 | 872.23 | 16.81% |
| Flat actor + Flat critic, reset permutation seed 2 | resetperm1500_actor_flat_flat_seed2 | 2 | 1310400 | -940.26 | 2.9845 | 63.67% | 831.4 | 32.18 | 953.79 | 6.84% |
| Flat actor + Flat critic, reset permutation seed 3 | resetperm1500_actor_flat_flat_seed3 | 3 | 1495200 | -876.81 | 2.5308 | 69.21% | 751.7 | 16.35 | 808.33 | 15.03% |
| Latent Slot-EqDec actor + Flat critic, reset permutation seed 1 | resetperm1500_actor_latent_slot_flatcrit_seed1 | 1 | 1411200 | -1069.8 | 3.1686 | 59.76% | 976.6 | 10.25 | 1056.4 | 9.02% |
| Latent Slot-EqDec actor + Flat critic, reset permutation seed 2 | resetperm1500_actor_latent_slot_flatcrit_seed2 | 2 | 1495200 | -1037.3 | 3.1027 | 61.50% | 959.0 | 24.00 | 1010.6 | 7.79% |
| Latent Slot-EqDec actor + Flat critic, reset permutation seed 3 | resetperm1500_actor_latent_slot_flatcrit_seed3 | 3 | 1411200 | -1056.1 | 3.1660 | 60.35% | 982.7 | 15.86 | 1040.8 | 7.45% |
| Latent Slot-EqDec actor + invariant Set critic, reset permutation seed 1 | resetperm1500_latent_slot_setcrit_separate_seed1 | 1 | 1495200 | -1382.2 | 4.3734 | 43.01% | 1166.6 | 3.60 | 1496.1 | 0.66% |
| Latent Slot-EqDec actor + invariant Set critic, reset permutation seed 2 | resetperm1500_latent_slot_setcrit_separate_seed2 | 2 | 1495200 | -1478.1 | 4.4760 | 44.36% | 1121.7 | 63.70 | 1460.5 | -1.37% |
| Latent Slot-EqDec actor + invariant Set critic, reset permutation seed 3 | resetperm1500_latent_slot_setcrit_separate_seed3 | 3 | 1495200 | -1363.3 | 4.0865 | 47.25% | 1147.7 | 4.48 | 1384.6 | -1.51% |

## Reset-Permutation Flat Descriptor, Dim 256, Per Seed

| variant | experiment | seed | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|---|
| Flat descriptor actor + Flat critic, reset permutation seed 1 | resetperm1500_flat_descriptor_flatcrit_seed1 | 1 | 806400 | -1460.5 | 4.3981 | 42.40% | 1219.6 | 2.99 | 1512.0 | -0.40% |
| Flat descriptor actor + Flat critic, reset permutation seed 2 | resetperm1500_flat_descriptor_flatcrit_seed2 | 2 | 1495200 | -1477.8 | 4.3427 | 43.73% | 1178.2 | 10.15 | 1477.0 | 2.09% |
| Flat descriptor actor + Flat critic, reset permutation seed 3 | resetperm1500_flat_descriptor_flatcrit_seed3 | 3 | 302400 | -1450.9 | 4.5433 | 40.58% | 1238.3 | 6.23 | 1559.7 | -0.74% |

## Flat Descriptor Dimension Sweep, Seed 1

| dim | variant | experiment | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | Flat descriptor actor + Flat critic, reset permutation seed 1 | resetperm1500_flatdesc_dim16_flatcrit_seed1 | 1495200 | -1456.5 | 4.2389 | 45.48% | 1165.0 | 20.70 | 1431.1 | -0.98% |
| 32 | Flat descriptor actor + Flat critic, reset permutation seed 1 | resetperm1500_flatdesc_dim32_flatcrit_seed1 | 1411200 | -1430.2 | 4.1678 | 46.91% | 1134.9 | 27.51 | 1393.5 | -2.89% |
| 64 | Flat descriptor actor + Flat critic, reset permutation seed 1 | resetperm1500_flatdesc_dim64_flatcrit_seed1 | 100800 | -1430.2 | 4.5236 | 40.72% | 1218.0 | 4.22 | 1556.2 | -0.81% |
| 128 | Flat descriptor actor + Flat critic, reset permutation seed 1 | resetperm1500_flatdesc_dim128_flatcrit_seed1 | 705600 | -1455.2 | 4.6397 | 39.18% | 1243.1 | 4.41 | 1596.5 | -0.96% |

## Mean Actor Critic Isolation, Seed 1

| variant | experiment | seed | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean actor + Mean critic, reset permutation seed 1 | resetperm1500_mean_baseline_mean_mean_seed1 | 1 | 1108800 | -720.20 | 2.1286 | 74.63% | 700.9 | 9.38 | 665.91 | 59.10% |
| Mean actor + separate Set critic | resetperm1500_mean_actor_setcrit_separate_seed1 | 1 | 1495200 | -1070.1 | 3.2847 | 59.74% | 872.4 | 42.38 | 1056.8 | 8.94% |
| Mean actor + Flat critic | resetperm1500_mean_actor_flatcrit_seed1 | 1 | 1495200 | -837.90 | 2.5767 | 67.38% | 828.9 | 3.02 | 856.33 | 19.53% |

## Slot-EqDec 1.5M Runs

| variant | experiment | seed | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|---|
| Full Set actor + slot-attention readout, Flat critic | diag1500_slot_eqdec_set_slot_flatcrit_seed1 |  | 1209600 | -867.14 | 2.6655 | 67.37% | 791.7 | 25.90 | 856.46 | 6.98% |
| Full Set actor + slot-attention readout, Set critic | diag1500_slot_eqdec_set_slot_setcrit_seed1 |  | 1411200 | -1154.5 | 3.3493 | 58.46% | 947.6 | 27.35 | 1090.5 | 4.70% |
| Full Set actor + token cross-attention readout, Flat critic | diag1500_slot_eqdec_set_cross_flatcrit_seed1 |  | 806400 | -1109.5 | 3.4053 | 57.13% | 873.1 | 22.90 | 1125.4 | 1.88% |
| mean-slot flat seed2 | next1500_slot_eqdec_set_slot_flatcrit_seed2 |  | 1108800 | -928.45 | 2.9208 | 64.49% | 863.0 | 38.17 | 932.06 | 4.54% |
| mean-slot flat seed3 | next1500_slot_eqdec_set_slot_flatcrit_seed3 |  | 1411200 | -860.95 | 2.5813 | 68.76% | 810.7 | 29.20 | 820.08 | 16.17% |
| latent-slot flat seed1 | next1500_slot_eqdec_latent_slot_flatcrit_seed1 |  | 1495200 | -877.59 | 2.6018 | 68.05% | 805.8 | 23.91 | 838.69 | 7.15% |
| mean-slot set separate seed1 | next1500_slot_eqdec_set_slot_setcrit_separate_seed1 |  | 1411200 | -965.55 | 2.8969 | 64.99% | 875.4 | 31.79 | 919.01 | 0.76% |
| mean-slot set shared_grad seed1 | next1500_slot_eqdec_set_slot_setcrit_sharedgrad_seed1 |  | 1209600 | -1068.1 | 3.1975 | 60.00% | 941.2 | 17.67 | 1050.0 | 16.59% |

Mean-pool Slot-EqDec + Flat critic seeds 1/2/3 aggregate:
| metric | value |
|---|---|
| cost_mean | 2.7225 |
| cost_min | 2.5813 |
| cost_max | 2.9208 |
| accept_mean | 66.87% |
| W1_mean | 821.8 |

## Descriptor Diagnostics 400k, Seed 1

| variant | experiment | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|
| sort_flat public descriptor |  |  |  | 3.5106 | 55.10% | 968.9 |  |  | 0.30% |
| Set actor + Flat critic |  |  |  | 4.6383 | 39.95% | 1230.4 |  |  | -1.64% |
| Flat actor + Set critic |  |  |  | 4.2400 | 44.38% | 1155.3 |  |  | -0.04% |

## Descriptor Diagnostics 1.5M, Seed 1

| variant | experiment | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|
| sort_flat public descriptor | diag1500_desc_sortflat_seed1 |  |  | 2.5370 | 69.52% | 800.0 | 20.36 | 800.18 | 8.64% |
| Set actor + Flat critic | diag1500_desc_setactor_flatcritic_seed1 |  |  | 4.0030 | 48.38% | 1087.8 | 6.96 | 1355.0 | 8.41% |
| Flat actor + Set critic | diag1500_desc_flatactor_setcritic_seed1 |  |  | 2.2201 | 73.16% | 710.8 | 3.68 | 704.42 | 11.17% |

## Actor Role Isolation 1.5M, Seed 1

| variant | experiment | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|
| Set-HAP + Flat-UAV actor | diag1500_roleiso_sethap_flatuav_seed1 | 1108800 | -765.42 | 2.2900 | 73.63% | 710.4 | 38.75 | 692.20 | 3.52% |
| Flat-HAP + Set-UAV actor | diag1500_roleiso_flathap_setuav_seed1 | 1411200 | -1440.6 | 4.2534 | 45.48% | 1133.8 | 19.85 | 1431.2 | -0.73% |
| Flat-HAP + Set-UAV relational actor | diag1500_roleiso_flathap_setuav_rel_seed1 | 1495200 | -1269.5 | 3.5498 | 55.76% | 941.5 | 26.81 | 1161.2 | 6.72% |

## Cross-Attention Readout 1.5M, Seed 1

| variant | experiment | selected_step | best_validation | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|---|---|---|
| Flat-HAP + Set-UAV pooled actor | diag1500_crossreadout_flathap_setuav_pooled_seed1 | 1411200 | -1440.6 | 4.2534 | 45.48% | 1133.8 | 19.85 | 1431.2 | -0.73% |
| Flat-HAP + Set-UAV relational actor | diag1500_crossreadout_flathap_setuav_rel_seed1 | 1495200 | -1269.5 | 3.5498 | 55.76% | 941.5 | 26.81 | 1161.2 | 6.72% |
| Flat-HAP + Set-UAV cross-attention actor | diag1500_crossreadout_flathap_setuav_cross_seed1 | 1495200 | -1061.4 | 3.1444 | 60.80% | 865.3 | 19.64 | 1028.9 | 20.77% |

## Full H350 Mean-Pool 896k

Source: `eval_outputs\full_h350_meanpool_896k\flat.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| flat.json | 2.3591 | 73.01% | 729.4 | 43.84 | 708.58 | 5.02% |

Source: `eval_outputs\full_h350_meanpool_896k\mean.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| mean.json | 2.2695 | 73.50% | 702.2 | 28.75 | 695.57 | 28.23% |

Source: `eval_outputs\full_h350_meanpool_896k\set_meanpool_detached.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| set_meanpool_detached.json | 4.4842 | 41.12% | 1220.4 | 1.90 | 1545.7 | -0.06% |

Source: `eval_outputs\full_h350_meanpool_896k\set_meanpool_sepcrit.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| set_meanpool_sepcrit.json | 4.5496 | 40.26% | 1248.6 | 0.96 | 1568.1 | -0.08% |

Source: `eval_outputs\full_h350_meanpool_896k\set_meanpool_sharedgrad.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| set_meanpool_sharedgrad.json | 4.3408 | 43.16% | 1209.3 | 2.04 | 1491.9 | -0.15% |

## FlatMLP 400k

Source: `eval_outputs\flatmlp_400k\set_flatmlp_sepactor_sepcrit.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| set_flatmlp_sepactor_sepcrit.json | 4.1688 | 45.57% | 1166.1 | 3.50 | 1428.8 | -0.76% |

## Reconstruction Aux 400k

Source: `eval_outputs\rec_aux_400k\set_meanpool_recchamfer_c01.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| set_meanpool_recchamfer_c01.json | 4.6367 | 39.04% | 1250.8 | 1.45 | 1600.3 | -0.04% |

## Set Actor Separate 112k

Source: `eval_outputs\set_actor_sep_112k\set_meanpool_sepactor_sepcrit.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| set_meanpool_sepactor_sepcrit.json | 4.5959 | 39.50% | 1287.3 | 0.00 | 1588.1 | -0.02% |

## Staged Pretrain 400k

Source: `eval_outputs\staged_pretrain_400k\set_latentslots_preenc_mixed64ep40_nofreeze_112k.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| set_latentslots_preenc_mixed64ep40_nofreeze_112k.json | 4.6197 | 39.22% | 1268.6 | 0.42 | 1595.6 | -0.19% |

Source: `eval_outputs\staged_pretrain_400k\set_meanpool_preenc_mixed64ep40_freeze12.json`
| file | cost/slot | accept | W1_m | overflow/ep | source_cost | HAP-freeze |
|---|---|---|---|---|---|---|
| set_meanpool_preenc_mixed64ep40_freeze12.json | 4.7404 | 37.69% | 1318.4 | 2.81 | 1635.5 | -0.31% |

## Action Projection Diagnostics

Source: `eval_outputs\action_projection_diag\resetperm1500_actor_flat_flat_seed1.deterministic.json`
| field | value |
|---|---|
| model_dir | D:\wyj\Projects\on-policy\onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo\resetperm1500_actor_flat_flat_seed1\run1\models\best |
| episodes | 8.0000 |
| eval_seed | 100000.0 |
| action_mode | deterministic |

Source: `eval_outputs\action_projection_diag\resetperm1500_actor_flat_flat_seed1.deterministic.rep2.json`
| field | value |
|---|---|
| model_dir | D:\wyj\Projects\on-policy\onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo\resetperm1500_actor_flat_flat_seed1\run1\models\best |
| episodes | 2.0000 |
| eval_seed | 100000.0 |
| action_mode | deterministic |

Source: `eval_outputs\action_projection_diag\resetperm1500_actor_flat_flat_seed1.stochastic.rep2.json`
| field | value |
|---|---|
| model_dir | D:\wyj\Projects\on-policy\onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo\resetperm1500_actor_flat_flat_seed1\run1\models\best |
| episodes | 2.0000 |
| eval_seed | 100000.0 |
| action_mode | stochastic |

Source: `eval_outputs\action_projection_diag\resetperm1500_actor_flat_flat_seed2.deterministic.json`
| field | value |
|---|---|
| model_dir | D:\wyj\Projects\on-policy\onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo\resetperm1500_actor_flat_flat_seed2\run1\models\best |
| episodes | 8.0000 |
| eval_seed | 100000.0 |
| action_mode | deterministic |

Source: `eval_outputs\action_projection_diag\resetperm1500_actor_latent_slot_flatcrit_seed1.deterministic.rep2.json`
| field | value |
|---|---|
| model_dir | D:\wyj\Projects\on-policy\onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo\resetperm1500_actor_latent_slot_flatcrit_seed1\run1\models\best |
| episodes | 2.0000 |
| eval_seed | 100000.0 |
| action_mode | deterministic |

Source: `eval_outputs\action_projection_diag\resetperm1500_actor_latent_slot_flatcrit_seed1.stochastic.rep2.json`
| field | value |
|---|---|
| model_dir | D:\wyj\Projects\on-policy\onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo\resetperm1500_actor_latent_slot_flatcrit_seed1\run1\models\best |
| episodes | 2.0000 |
| eval_seed | 100000.0 |
| action_mode | stochastic |

Source: `eval_outputs\action_projection_diag\resetperm1500_latent_slot_setcrit_separate_seed1.deterministic.rep2.json`
| field | value |
|---|---|
| model_dir | D:\wyj\Projects\on-policy\onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo\resetperm1500_latent_slot_setcrit_separate_seed1\run1\models\best |
| episodes | 2.0000 |
| eval_seed | 100000.0 |
| action_mode | deterministic |

Source: `eval_outputs\action_projection_diag\resetperm1500_latent_slot_setcrit_separate_seed1.stochastic.rep2.json`
| field | value |
|---|---|
| model_dir | D:\wyj\Projects\on-policy\onpolicy\scripts\results\MEC\v6_hap_loadbearing\mappo\resetperm1500_latent_slot_setcrit_separate_seed1\run1\models\best |
| episodes | 2.0000 |
| eval_seed | 100000.0 |
| action_mode | stochastic |
