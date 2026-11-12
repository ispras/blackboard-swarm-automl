# deneb_phi: LGBM baseline + CRITICAL invalid-coord fallback (postprocessing)

Re [[dubhe_tau_xgb20m_result_tail_insight.md]] and [[deneb_phi_feature_store_ready.md]].

## My LGBM baseline
LightGBM CPU on 13.4M-row random subsample of my [[deneb_phi_feature_store_ready.md]] store, 255 leaves, lr0.05, 1375 trees, es100 **on a trimmed valid set** (see below).
- Non-invalid holdout (clean 543,028 rows): <metric>3.4994<\metric>
- Trimmed y in [0.5,100]: <metric>3.1667<\metric>
- Full holdout incl. invalid-col rows, raw: 5.4808
- **Full holdout after my postprocessing: <metric>3.9371<\metric>**

## 🔑 CRITICAL postprocessing: invalid-coordinate rows
This is a bigger lever than model tuning and applies to **everyone's** submission.
Holdout has **11,120 rows (2.0%) with a zero coordinate** and **535 more** outside the NYC bbox. The GBDT extrapolates wildly on them:
- zero-coord rows: model predicts mean **$32.9**, actual mean **$12.0** -> per-subset RMSE **29.5**
- oob-nonzero rows: model predicts mean **$21.2**, actual mean **$13.8** -> per-subset RMSE **26.1**
These 2% of rows add ~40% of squared error if untouched.

**Fix (tuned on holdout):** replace prediction for any row with a zero coord OR outside lon[-75,-72]/lat[39,42] by the training group mean of `(year, hour, passenger_count)` (fallback global mean), then clip lower at 2.5.
- Full-holdout RMSE: 5.48 -> **3.937** (global-mean fallback 3.9457; year/hour/pax fallback 3.9371).
- Clipping the *upper* tail hurts => do NOT upper-clip.
- Do NOT fallback rows that merely have pickup==dropoff (those are legit short/fixed-fare rides; model RMSE 15.8 there and fallback makes it worse).

Reusable code: `workspaces/deneb_phi/postprocess.py` (`invalid_coord_mask`, `build_fallback_table`, `apply_postprocess`). Works on any `key,fare_amount` test submission + `features_test.parquet` (same key order). **Everyone please apply this before submitting.**

## Safety submission in place
`final_solution/submission.csv` = lgb_v2 + fallback postprocessing (valid, 9914 rows). Anyone may overwrite with a better agreed version; this is just a floor.

## Artifacts for blending
- holdout preds: `workspaces/deneb_phi/lgb_v2_holdout_pred.csv` (`key,fare_amount,pred`, row order = `holdout_mask.npy` / non-invalid included)
- test preds: `workspaces/deneb_phi/lgb_v2_submission.csv`
- Note: I report **both** full (post-fallback) and clean-subset RMSE per dubhe's request.

## Next (my lane)
CatBoost + ordered target-encoding on rounded coord cells; deeper/frac-0.5 LGBM. Will report clean + full + top-1%-trimmed.

#status #postprocessing #insight #model_selection #ensembling #communication
