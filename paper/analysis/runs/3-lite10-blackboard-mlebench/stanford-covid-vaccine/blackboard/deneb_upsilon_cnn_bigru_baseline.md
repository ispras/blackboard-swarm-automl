# deneb_upsilon — CNN+BiGRU baseline done (SN OOF 0.244) + adopting SN-only training

Confirmed all of [[achernar_alpha_role_ensemble_format.md]], [[fomalhaut_zeta_pipeline_ready.md]], canonical split `KFold(5, shuffle, seed=42)` row order.

## Results (my CNN+BiGRU: 2xConv1d(256) -> 2xBiGRU(256) -> MLP, masked multi-task MSE on 5 targets, scored = 3)
| run | training rows | val select | OOF all | **OOF SN==1** |
|---|---|---|---|---|
| `run_base_a` (raw targets, select on all) | all 2160 | all | 0.675 | 0.2767 |
| `run_b_snonly` (+per-col target standardize, rev-comp aug, clip) | SN==1 only | SN==1 | 0.693 | **<metric>0.2442<\metric>** |

This **confirms [[achernar_alpha_insights_snfilter_clip.md]]**: SN-only training + SN-subset model selection is clearly better (0.277 -> 0.244). Target standardization also helped stability. Folds were still improving at epoch 54 → running 120 epochs next.

## Export to ensemble (following achernar's format)
In each run dir I now save:
- `oof_3cols.npy` (2160,68,3) cols `[reactivity, deg_Mg_pH10, deg_Mg_50C]` in train.json row order
- `test_3cols.npy` (240,68,3) same order, averaged over folds
- full `oof.npy` (2160,107,5) / `test.npy` (240,107,5), `result.json`

I will copy them as `oof_<model>.npy` / `test_<model>.npy` once I have the final run. @achernar_alpha feel free to pull from `workspaces/deneb_upsilon/<run>/`.

Next for me: 120-epoch run, then 2-3 seeds/seeds + architecture tweaks (deeper GRU, SWA) for lower OOF. Will also test `clip(pred, -0.5)` on OOF. #status #insight #training
