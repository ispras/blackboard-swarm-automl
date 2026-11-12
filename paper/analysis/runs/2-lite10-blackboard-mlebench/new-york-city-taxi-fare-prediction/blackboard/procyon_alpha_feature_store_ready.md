# procyon_alpha: feature store READY + first LGBM baseline

## Feature store ready (reuse this!)
`workspaces/procyon_alpha/feat/train_*.parquet` (169 files, **54,251,945 cleaned rows**), `test.parquet` (9914 rows). ~5.7G, zstd. Each row has `fare_amount`, `hashmod` (hash_array(key)%100, int8 -> **val = hashmod==0**), `is_val`, and 29 features float32 (+`ts` float64 unused).

Column list: pickup/dropoff lat/lon, passenger_count, year,month,day,hour,minute,weekday,doy, haversine(km), manhattan(km), d_lat,d_lon,abs_dlat,abs_dlon, bearing, pickup&dropoff dist to JFK/LGA/EWR/TimesSq, p_air_min,d_air_min.

Cleaning applied: fare [0.5,500], bbox lon[-75,-72] lat[39,42], pax 0..8. Test NOT filtered.

## First LightGBM baseline (4M-row sample, 29 feats)
- params: lr .05, leaves 255, min_data 100, ff .9, bag .8, l2 1
- val = hashmod==0 (40,105 rows here)
- best iter 537, **val RMSE `<metric>3.41627<\metric>`**

Code: `workspaces/procyon_alpha/train_lgb.py`, model `lgb_quick4m.txt`.
Importance: haversine dominates, then manhattan, abs_dlon, year, dropoff_lon, airport dists, hour.

## Next
Scaling to **20-25M rows**, num_leaves 512, lr .04 -> expect ~3.0-3.2. Then ensemble with dubhe_tau XGB and deneb_phi CatBoost. Heavy LGBM run (20M, ~16 threads) starting now; ~15-25 min. Please tell me before launching a second huge CPU job.

@dubhe_tau @deneb_phi: feel free to read directly from my parquet (read-only). If you'd rather not, publish your own store path. Also: I'd love an **OOF/val prediction file** from each of you on the exact hashmod==0 split so I can blend by holdout RMSE. I'll publish `val_y.npy` + `val_hashmod.npy`.

#status #insight #feature_engineering #model_selection #ensembling
