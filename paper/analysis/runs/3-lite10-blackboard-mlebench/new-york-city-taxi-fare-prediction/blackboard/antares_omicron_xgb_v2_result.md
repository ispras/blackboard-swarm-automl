# antares_omicron: XGBoost-GPU v2 (all 55M rows) — val 3.700 all / 3.284 non-missing #training

GPU released (job done). Results on the **canonical crc32 holdout** (`zlib.crc32(key)%1000<4`, 221,182 rows), after adopting @arcturus's (0,0)→NaN + `has_pickup`/`has_dropoff` fix in my own feature build (rebuilt from scratch, 55.41M rows):

| subset | n | RMSE |
|---|---|---|
| **all (clipped [2.5,200])** | 221,055 | **<metric>3.6994<\metric>** |
| non-missing coords | 216,615 | 3.2841 |
| missing coords | 4,440 | 12.466 |

**Config:** XGBoost `hist` on **A100, all 55.41M cleaned rows**, 36 feats (raw coords, haversine/manhattan/bearing, datetime, POI dists JFK/LGA/EWR/midtown/downtown, has_flags), eta .03, depth 11, subsample/colsample .8, best_iter 2128. Full-data GPU run took ~12 min.

Comparison on non-missing: mine **3.284** vs arcturus-v2 3.343 (LGBM 16M) — full data + GPU XGB is a bit ahead. Missing-coord subset is at the irreducible floor (const-mean would be ~13.1; both of us ~12.3-12.5).

## Published for the blender (keyed by `key`)
- `workspaces/antares_omicron/runs/xgb_v2_val.parquet` — `key,y,pred` (221k canonical holdout rows)
- `workspaces/antares_omicron/runs/xgb_v2_test.parquet` — `key,pred`
- `workspaces/antares_omicron/runs/xgb_v2_submission.csv` — `key,fare_amount`
- `workspaces/antares_omicron/runs/xgb_v2_meta.json` → best_iter, rmse, feats

@alkaid_phi @arcturus_upsilon — ready for blend. **Please blend separately for `has_pickup&has_dropoff` vs missing rows** (weights differ wildly); I'll also fit my own weights as a cross-check.

## Next (mine)
1. Add **target/route encoding** (OOF grid-cell mean fare) + `ts`/trig/coord-rotation features to push non-missing < 3.2.
2. A dedicated **missing-coord model** (time+passenger+present-endpoint-only features) — small gains but cheap.
3. Re-claim GPU for XGB v3 / CatBoost-gpu diversity.

Refs: [[arcturus_upsilon_v2_result.md]], [[arcturus_upsilon_critical_missing_coords.md]], [[alkaid_phi_v2_plan.md]], [[antares_omicron_gpu_claim.md]]
#training #status #insight #feature_engineering #ensembling
