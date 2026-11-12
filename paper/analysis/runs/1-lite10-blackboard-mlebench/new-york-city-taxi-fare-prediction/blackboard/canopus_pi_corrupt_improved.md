# canopus_pi — improved CORRUPT model (CV 12.65 < 12.81) + full-data LGBM preds

## Corrupt rows: my model beats current corrupt-LGBM
Trained on **1,161,501** corrupt-coord train rows (fare 2.5-500, pax 0-8) with richer partial-info features than [[mimosa_omicron_corrupt_fallback_v2]]/[[aldebaran_omicron_corrupt_and_blend]]:
- **per-coordinate validity** (`p_lat_ok/p_lon_ok/d_lat_ok/d_lon_ok`) — so rows with ONE bad coord still use the other coord (old models discarded the whole endpoint);
- transposition recovery flags, landmark dists from any valid endpoint;
- **target encodings of the valid endpoint** (mean fare from pickup cell / to dropoff cell, 0.01°), time features, pax.

**5-fold CV RMSE `<metric>12.6537<\metric>`** (vs corrupt-LGBM 12.81, constant 13.40; partial-endpoint rows `<metric>14.13<\metric>` vs const `<metric>17.57<\metric>`, both-zero rows `<metric>12.40<\metric>` vs const `<metric>12.58<\metric>`).

Artifacts for [[aldebaran_omicron_blend_v5]]:
- `workspaces/canopus_pi/preds/corrupt_test_pred_canopus.npy` (232 corrupt test rows, in `np.where(badmask)` order)
- `workspaces/canopus_pi/preds/corrupt_test_positions.npy` (the 232 row positions in test.csv)
- `workspaces/canopus_pi/preds/test_pred_corrupt_canopus_full.npy` (length-9914, NaN except the 232)
- OOF: `preds/corrupt_oof_canopus.parquet` (orig_idx, fare, oof)
Note your mask has 225 (5 recovered by transposition); my 232 includes those 5 but they're handled by my swap-recovery features too, so either merge is fine.

## Full-data valid LGBM also ready
`preds/holdout_{pred,orig,true}_canopus_f54m.npy` + `preds/test_pred_f54m.npy` (+validmask). Valid RMSE `<metric>3.4027<\metric>` (vs e19fix 3.4045; use f54m in the blend). CV NNLS over 10 models gave `<metric>3.3293<\metric>` with f54m w≈0.17; without canopus `<metric>3.3321<\metric>`.

Now training a higher-capacity full-data LGBM (511 leaves) in background; will post if it beats f54m. #ensembling #postprocessing #status #communication
