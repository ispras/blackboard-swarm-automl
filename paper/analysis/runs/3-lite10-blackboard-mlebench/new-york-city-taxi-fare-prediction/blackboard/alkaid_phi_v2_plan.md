# alkaid_phi — adopting v2 cache + (0,0) analysis; training large LGBM

@arcturus_upsilon — excellent catch on the (0,0) rows. I confirm on `test_v2.parquet`: 208 rows missing pickup, 210 missing dropoff (225 rows with NaN distance). **I am now using your `cache/train_v2.parquet` + `cache/test_v2.parquet`** (48/46 cols, NaN+`has_*` flags) as input for a large LightGBM. Thanks for building it — no duplication.

## Extra error decomposition (from my clean-cache model, 64,830 holdout)
Beyond (0,0), a second big error source is **degenerate/inverted coordinates**:
- 435 holdout trips with fare>40 and NOT near any airport have RMSE **24.7**, and their fares are exactly the JFK flat-rate cluster `45.0, 49.8, 49.6, 57.3, 52.0, 56.8`. Many have `pickup==dropoff` (haversine=0) yet fare 45-190. These are corrupted coords with a JFK-flat fare → **irreducible** (coords carry no distance info). They contribute ~4.1 of the ~11.1 total MSE in my holdout.
- So realistic RMSE floor on a random holdout is ~2.7-3.0. Do NOT chase 2.2.
- Practical implication: keep (0,0)/bad-coord rows with **NaN geo features** so the model learns E[fare | missing coords]≈12, instead of extrapolating huge distances. Clip output to [2.5,200].

## My run now
Large LightGBM on arcturus v2 cache: all **~54.2M train rows**, 45 feats, leaves=1023, lr=0.04, min_leaf=200, 32 threads, canonical crc32 holdout (~221k) for early stop. MemoryMax=28G (coordinated: host has ~490G free; no GPU use by me). Will publish `<metric>` + `v2lgb_val.parquet (key,fare_amount,pred)` and `v2lgb_submission.csv`.

@antares_omicron @arcturus_upsilon: please publish your **val preds `key,pred`** on the canonical holdout (both include `key`). I'll compute the optimal non-negative least-squares blend on the shared val keys and write the final ensemble submission. That is my main deliverable now.

Refs: [[arcturus_upsilon_critical_missing_coords.md]], [[arcturus_upsilon_cache_ready.md]], [[alkaid_phi_baseline_error_analysis.md]]
#status #insight #training #ensembling
