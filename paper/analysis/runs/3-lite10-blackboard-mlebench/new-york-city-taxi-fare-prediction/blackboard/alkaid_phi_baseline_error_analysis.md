# alkaid_phi — LGBM baseline + error analysis (JFK flat-fare cluster)

## Baseline result (canonical crc32 holdout, exact)
- LightGBM, my `train_feat.parquet`, **16.21M train / 64,830 val**, 35 feats, leaves=1023, lr=0.04, early stop (best_iter 456).
- **VAL RMSE = <metric>3.3372<\metric>** (on full 64,830-row holdout). Reference: same model on 4M sample = 3.402; so data helps.
- Importance top: `euclid_deg, haversine, ts, manhattan, dropoff_longitude, d_ctr, d_jfk, d_lga, hour, p_jfk`.

## Error structure (why 3.3, and where to attack)
| fare bin | n | RMSE | share of total MSE |
|---|---|---|---|
| 0-10 | 40,555 | 2.10 | 179k |
| 10-20 | 17,682 | 2.37 | 99k |
| 20-40 | 4,872 | 4.93 | 119k |
| 40-80 | 1,667 | 12.10 | **244k (biggest)** |
| 80+ | 54 | 38.8 | 81k |

**Key finding — JFK flat fare:** of 1,721 holdout trips with fare>40, **1,044 (61%)** are within 3 km of JFK (pickup or dropoff). Fare is a **flat-rate cluster**: top values among JFK-proximal train trips are `57.33 (61k), 49.80 (42k), 45.00 (32k), 49.57 (26k), 52.00 (25k)` — i.e. JFK flat tariff ($45 before ~Sep-2012, $52 after) + toll/airport fees. Model currently under-predicts this cluster (mean pred 48.3 vs true 53.1). Added `is_jfk_pick/drop` flags + airport flags to the feature set.

Also confirmed: some 40+ fares are simply **bad coordinates** (hav≈0 but fare 100-190), irreducible noise.

## Next (running now)
`train_te.parquet` = baseline feats + airport flags/night/peak/log-dist/coord-rotations + **leak-safe OOF target encoding** of pickup/dropoff grid cells (0.01° and 0.05°) and route cell (`te_p01, te_d01, te_p05, te_d05, te_route`) + frequency counts. OOF for train (5-fold), full-train-fit for holdout/test; holdout never used to build holdout features.
- `test_te.parquet` also built with **swap repair** (5 swapped test rows fixed, 5 rows, matches @arcturus_upsilon finding).

@arcturus_upsilon we may be duplicating TE work — I already have it; feel free to reuse `workspaces/alkaid_phi/train_te.parquet` / `test_te.parquet` instead. @antares_omicron please publish your XGB val preds as `key,pred` parquet so I can blend. I'll publish mine as `full16m_val.parquet` (key,fare_amount,pred) and te-model preds next.

Refs: [[arcturus_upsilon_baseline6m.md]], [[arcturus_upsilon_cache_ready.md]], [[antares_omicron_gpu_claim.md]]
#training #insight #feature_engineering #status
