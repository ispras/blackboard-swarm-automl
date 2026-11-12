# dubhe_tau: TE features = BREAKTHROUGH (3.3449 clean) + interim submission written

## ✅ Route/cell target-encoding AS A FEATURE helps a lot (contrary to [[deneb_phi_catboost_and_route_negative.md]])
Deneb correctly measured route-mean *alone* (RMSE 5.1) is weak. But as **extra features in my full-data XGB** it helps:
- OOF (5-fold, leak-free on train) **dec2 pickup-cell / dropoff-cell / route TE + log-counts** (6 feats), computed only from non-holdout rows; holdout/test use full-train encoding (no holdout labels).
- XGBoost full 53.7M, aug47 + TE6 = 53 feats, max_bin512, d13, lr0.045, 1400 trees → **clean holdout `<metric>3.34492<\metric>`** (vs 3.38349 without TE). Artifacts: `models/xgb_te_{holdout_pred,test}.csv`.
- Code: `scripts/build_te.py` (TE), `scripts/xgb_train_te.py` (model). Reusable `features/TE_dec2.npy`.

## Blend now
NNLS on my 543,067 valid rows incl. nn members: **CV 3.34443** (xgb_te weight .76, xgb_aug .08, nn_te .06, nn_emb .09, others ~0). GBDT-only is saturated; the lift came from TE features, not ensembling.
**Interim `final_solution/submission.csv` written** = blend6 (xgb_te-dominant) + `postprocess_final.py`. Expected full-holdout ≈ **3.78-3.81** (was 3.7975 in [[deneb_phi_safety_blend_and_ridge_stability.md]] with xgb_b512 blend).

## Other members result
- `nn_te` (MLP+cell embeddings+TE) clean 3.48504; `nn_emb` 3.50526.
- procyon full-data LGBM v2 in progress (val ~3.43 at 375 iters).

## Next
Running `xgb_te2` = +dec3 pickup/dropoff cell TE + coarse-route TE (59 feats) to test if finer TE adds more. Will report; then final blend (Ridge vs NNLS compared by 5-fold CV) + postprocess.

## Division of final packaging (ack @deneb_phi)
I write `final_solution/submission.csv` (done, will update). **@deneb_phi please write `final_solution/README.md` + `final_solution/code/`** (reproducible pipeline + references). Final member list/weights I'll send when xgb_te2 + procyon full LGBM are in. @procyon_alpha send keyed `lgb_v2_full` preds when done.
#status #feature_engineering #ensembling #postprocessing #submission #communication
