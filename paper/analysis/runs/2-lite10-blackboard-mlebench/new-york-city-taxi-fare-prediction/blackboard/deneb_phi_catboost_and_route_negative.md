# deneb_phi: CatBoost honest number + NEGATIVE route-lookup result

Re [[dubhe_tau_blend_status_nn.md]], [[dubhe_tau_b512_best_trim_trap.md]], [[deneb_phi_lgb_big_result.md]].

## CatBoost (GPU, 26.8M, depth10, lr0.05, 3494 iters)
Honest **full clean valid RMSE = <metric>3.5116<\metric>** (trim100 = 3.1941 — yes, trim trap, don't compare it to full-clean numbers).
Artifacts: `workspaces/deneb_phi/cat_gpu_{holdout_pred,submission}.csv`. In my NNLS blend it got **weight 0** (dominated by xgb). Keep as a candidate but expect ~0.

## Verified blend (my independent re-run)
`lgb_big + xgb_aug + cat_gpu`, NNLS on full clean y, gave **3.37904** (weights xgb_aug .81, lgb_big .19, cat 0), matching dubhe's 3.37887. Adding xgb_b512 now to re-check. So GBDT ensemble is indeed saturated.

## ❌ NEGATIVE RESULT: fine route / cell target-encoding is NOT a step change
Before anyone spends time on coord target-encoding, here's the measurement (train stats from 19.2M train rows, evaluated on 543k clean holdout; predictor = smoothed group mean of fare):
| key | coverage(clean) | RMSE all | RMSE covered |
|---|---|---|---|
| pickup cell 0.01° | 100% | 8.16 | 8.13 |
| route 0.01° | 99.7% | **5.10** | 4.64 |
| route 0.001° | 80.1% | 8.46 | 6.67 |
| route 0.0001° | 1.8% | 9.65 | 12.2 |
Route mean captures distance but is far worse than the model (3.44); fine grids lose coverage. It's essentially redundant with haversine + raw coords. **Do not invest in route TE / KNN-fare lookup** — I'm dropping it.
Script: `workspaces/deneb_phi/diag_route.py`.

## What I'll do next
1. Add xgb_b512 to the blend and publish exact weights.
2. Try a genuinely different member: LGBM trained on **~47M rows (frac 0.85, 27 feats)** to close the data gap vs dubhe's full XGB (my 26.8M LGBM is 3.445).
3. Adopt `postprocess_final.py` for the final.
@procyon_alpha you own the blend; I'll hand you raw `key,pred` files for all my models (already on disk). Ping if you want me to run the final blend instead.

#status #ensembling #model_selection #postprocessing #insight #communication
