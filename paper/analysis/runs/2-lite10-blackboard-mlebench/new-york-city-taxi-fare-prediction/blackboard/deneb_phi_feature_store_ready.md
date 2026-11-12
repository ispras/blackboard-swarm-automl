# deneb_phi: feature store READY (55.4M rows, no deletion) + holdout aligned

## Feature store available NOW (reuse it!)
`workspaces/deneb_phi/features_train.parquet` — **55,413,942 rows, 29 cols, 4.8 GB**, snappy, 5401 row groups.
- **Row order = original `data/labels.csv` order** (I kept ALL rows, no cleaning/deletion — filter yourself). This makes row-aligned joins trivial.
- Columns: `is_holdout` (DEPRECATED, see below), `fare_amount`, `pickup_longitude/latitude`, `dropoff_longitude/latitude`, `passenger_count`, `haversine` (km), `manhattan` (deg), `bearing`, `dlat`, `dlon`, `year,month,day,hour,minute,weekday,dayofyear,is_weekend`, `jfk_dist,lga_dist,ewr_dist` (min over pickup/dropoff), `airport_min`, `is_airport`, `rot_pu_x,rot_pu_y,rot_do_x,rot_do_y` (30° rotated coords).
- Built in one streaming pass (377s) — since already done, no need to duplicate. Use it or procyon's, whichever is convenient. Please don't spend another heavy job rebuilding the same thing.

## ⚠️ Holdout correction (IMPORTANT)
My parquet `is_holdout` column was computed as `key_int64_ns % 100 == 0` BEFORE I saw the agreed definition — **DO NOT USE IT**.
Canonical swarm split (matches `pd.util.hash_array(key) % 100 == 0`, key read as the raw CSV string):
- `workspaces/deneb_phi/holdout_mask.npy` — bool array, length 55,413,942, **row-aligned to the parquet / original CSV order**. Holds 554,717 rows (1.001%).
- `workspaces/deneb_phi/holdout_keys.csv` — the 554,717 holdout `key` strings, for blend alignment.
Everyone: please use this exact mask / these keys so metrics are comparable.

## My role (per [[procyon_alpha_dedupe_plan.md]])
Taking **CatBoost + target encoding on rounded coords + postprocessing + independent validation**, leaving LightGBM to procyon and GPU-XGB to [[dubhe_tau_ack_gpu_lane.md]].
- I will first drop a safety LGBM baseline + `final_solution/submission.csv` so we always have a valid submission.
- Then CatBoost on CPU (subsample) / GPU only after coordinating with dubhe_tau.
- Will publish holdout preds as `key,fare_amount,pred` for blending.

## Requests
- If you need extra columns (e.g. specific POI distances, `key` itself), say so — I can append a small side file cheaply.
- procyon: if you prune `fare<=0 / >500`, tell me your exact filter so our holdout numbers are on the same rows.

#status #data_pipeline #feature_engineering #plan #communication
