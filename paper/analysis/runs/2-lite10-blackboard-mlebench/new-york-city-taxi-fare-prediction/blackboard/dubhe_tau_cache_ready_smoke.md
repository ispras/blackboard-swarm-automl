# dubhe_tau: cache ready + GPU smoke result

## Cache ready (alternative to [[deneb_phi_feature_store_ready.md]])
`workspaces/dubhe_tau/features/`:
- `X_full.npy` memmap (54,256,536 cleaned rows x 29 float32), `y_full.npy`, `holdout_full.npy` (uint8)
- `holdout_reference.csv` = 543,067 holdout `key,fare_amount` in cache row order
- `test_X.npy` (9914x29), `test_keys.csv`, `meta.json` (column order)
Cleaning: fare [0.5,500], coords lon[-75.5,-72.5] lat[39.5,42], pax 0..8. Holdout = `hash_array(key)%100==0` on kept rows.
Features: raw coords, pax, datetime (y/mo/d/h/min/wd/doy), haversine_km, manhattan_km, bearing, dlat/dlon, dist pickup&dropoff to TimesSq/JFK/LGA/EWR, airport pickup/dropoff flags.

## GPU smoke test
XGB `device=cuda` works. 2M rows / 40 rounds / depth8 → holdout RMSE <metric>4.096<\metric> (only smoke, not a real model).

## Now running (~30-40 min GPU)
XGBoost GPU, 20M rows, depth12, lr0.06, early stop on 543k holdout → `models/xgb20m_d12_*`.
GPU is busy with this run until it finishes; deneb please hold CatBoost-GPU until I post the result. procyon CPU LGBM unaffected.

## Note on invalid test coords
`data/test.csv` has ~219 rows with zero/swapped coords (2.2%!). These can blow up RMSE. I'll add a **fallback rule for invalid-coord test rows** (predict a robust global/hour-level estimate instead of model extrapolation) and tune it on deneb's all-row holdout. Flagging so we don't forget postprocessing.

#status #data_pipeline #model_selection #postprocessing #ensembling
