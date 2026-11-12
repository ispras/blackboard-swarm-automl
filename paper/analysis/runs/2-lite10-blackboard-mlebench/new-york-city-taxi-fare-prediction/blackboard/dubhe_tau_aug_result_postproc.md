# dubhe_tau: augmented XGB = 3.3905 clean + working postprocess (validated)

## Clean-model leaderboard (all on hash-holdout, cleaned rows)
| model | clean holdout RMSE |
|---|---|
| XGB full 53.7M, 29 feats, d13 | 3.41238 |
| **XGB full 53.7M, 47 aug feats, d13** | **<metric>3.39046<\metric>** |
| procyon LGBM 22M | 3.4816 (from [[procyon_alpha_fair_val_and_v2.md]]) |
| deneb LGBM 13.4M | 3.4994 (from [[deneb_phi_lgb_baseline_and_fallback.md]]) |

Aug features (my `xgb_train_aug.py`): rotated 30° coords for pu/do, cyclic hour/weekday, is_weekend, log1p(haversine/manhattan), same_point flag, sin/cos bearing, per-airport-trip flags, ddc-dpc. Cheap and effective (+0.022).

## Postprocess ready (use this!)
`workspaces/dubhe_tau/postprocess_final.py` + fitted `workspaces/dubhe_tau/postproc/{fallback_allzero.csv, partial_model.json, fallback_meta.json}`.
- Category logic per [[dubhe_tau_invalid_categories_refined.md]].
- **Validated on deneb's `lgb_v2` full holdout preds**: raw 5.4808 → **3.9044** (deneb's own fallback gave 3.9371). Gains come from: partialzero dedicated model (RMSE 18.3→15.07) and keeping swap preds (7.83 instead of fallback's 10.1).
- Expected official with clean model 3.3905: **~3.81-3.83**.
- CLI: `python postprocess_final.py --pred <any test preds key,fare_amount> --out submission.csv`. Requires only `data/test.csv` (and it recomputes datetime).

## Artifacts for blending (@procyon_alpha)
My holdout preds (key,fare_amount,pred; my cleaned 543,067-row holdout = subset of canonical holdout, valid rows only):
- `workspaces/dubhe_tau/models/xgb_full_d13_holdout_pred.csv`
- `workspaces/dubhe_tau/models/xgb_aug_d13_holdout_pred.csv`
Test preds: `.../models/xgb_aug_d13_test.csv`, `xgb_full_d13_test.csv`.
For a fair blend on **valid rows**, merge on `key` against canonical holdout. Note: blend weights on valid rows, then apply postprocess to the blended test prediction — don't blend the fallback rows with the models.

## Now running (GPU ~30 min)
XGB aug, `max_bin=512`, lr0.04 → for bagging/ensemble diversity. GPU busy until ~02:20 MSK. #status #model_selection #ensembling #postprocessing
