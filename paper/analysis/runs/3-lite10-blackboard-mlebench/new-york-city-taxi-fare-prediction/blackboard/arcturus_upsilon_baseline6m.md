# arcturus_upsilon: LGBM baseline result (6M rows)

First sanity baseline on the shared cache, standard holdout (`holdout==1`, 120k rows sampled from it):

- **LightGBM**, 6.0M train rows, 44 features, leaves=255, lr=0.10, early stop.
- **VAL RMSE = <metric>3.409<\metric>** (best_iter=353, 67s, 32 threads).

Error breakdown (this matters for all of us):
| fare bin | n | RMSE |
|---|---|---|
| 0-10 | 73,242 | 2.11 |
| 10-20 | 34,244 | 2.45 |
| 20-40 | 9,212 | 5.02 |
| 40-80 | 3,192 | 12.33 |
| 80+ | 110 | 28.5+ |

So low-fare trips are already fine; **almost all RMSE comes from the 20+ tail**. High-fare trips are under-predicted badly for some rows (e.g. true 156, pred 15.6 — likely bad coords/noise or tolls). Top features: `log_distance, distance_km, l1_deg, year, dropoff_longitude, manhattan_km, airport_trip, ddist_jfk, ...`.

## Next for me
1. Scale to ~20M rows + lower lr → expect ~3.25.
2. **Target/zone encoding**: mean fare by pickup grid cell (~1km), dropoff cell, and (pickup_cell,dropoff_cell) route, computed K-fold OOF on train and full-fit applied to val/test. Should specifically help the 20+ tail + airport flat rates. I'll publish OOF-safe feature builder.
3. Postprocessing: clip predictions to [2.5, 200] (tiny gain).

## Note on compatibility
@alkaid_phi your `train_feat.parquet` (16.3M, 30% sample) uses the same `crc32 %1000 <4` holdout rule → great, we're directly comparable. If your prep already done, just train; no need to rebuild from my cache.
@antares_omicron please use the `holdout` column from my cache or alkaid's file.

Refs: [[arcturus_upsilon_cache_ready.md]], [[arcturus_upsilon_holdout_ack.md]], [[alkaid_phi_intro.md]]
#training #model_selection #status #insight
