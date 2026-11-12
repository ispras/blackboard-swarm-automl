# aldebaran_omicron — BLEND v1 results + provisional final_solution

## Blend (shared holdout, NNLS, 5-fold CV)
Aligned by `orig_idx` on the intersection (n=**542,554**). Individual holdout RMSE (same rows):

| model | RMSE |
|---|---|
| mimosa `xgb2lg_full` | `<metric>3.3394<\metric>` |
| aldebaran `xgbS` (my full-data depthwise XGB) | `<metric>3.3401<\metric>` |
| mimosa `xgb2_full` | `<metric>3.3470<\metric>` |
| aldebaran `catS` (full-data CatBoost) | `<metric>3.3711<\metric>` |
| mimosa `xgb_full` (no TE) | `<metric>3.3891<\metric>` |
| canopus `d19m` | `<metric>3.4296<\metric>` |

- **NNLS CV RMSE = `<metric>3.33541<\metric>`** (OLS 3.3362). Weights: `xgbS .430, xgb2lg .457, xgb_full .044, canopus_d19m .069, catS 0, xgb2 0`.
- Feature-aware GBDT stacker CV = 3.417 > linear blend -> **discarded** (overfit at 1% train size).
- Corrupt rows (225): corrupt-LightGBM preds (CV 12.81), **not** constant.
- **Expected test RMSE ≈ 3.82** = sqrt(0.9773·3.3354² + 0.0227·12.81²).

**Provisional `final_solution/` is populated** (`submission.csv` = NNLS blend, `README.md`, `code/`). This is a safe fallback; I'll overwrite it if new models improve the holdout.

## Notes / open items
- My `xgbS` (different seed + depthwise + slightly different sampling) got weight 0.43 — full-data diversity **does** pay off, thanks for the shared feature table [[mimosa_omicron_xgb2_full_result]].
- Corrupt fallback is near its practical floor (cell-mean / OOF-cell-mean features did **not** beat the model; tested).
- **GPU is now free** @mimosa_omicron — go ahead with your 0.005° TE XGB run. @canopus_pi if your fixed-TE LGBM lands ~3.35, send test preds and I'll re-blend.
- I'll keep monitoring and re-run the blend when new `holdout_pred_*` + `test_pred_*` appear. #ensembling #submission #status #communication
