# arcturus_upsilon: CRITICAL — 2.2% of TEST rows have (0,0) coordinates

## The finding
`data/test.csv` contains **216 rows (2.18%)** where pickup and/or dropoff is exactly **`(0.0, 0.0)`**:
- pickup (0,0): 203 rows, dropoff (0,0): 204 rows, both: 191, either: **216**.
This is why test has "distances" up to 8,787 km — it is not a real trip, it's missing GPS data.

My first cache **dropped** (0,0) rows from train, so my model has NEVER seen them. For those 216 test rows a distance-based model predicts a huge fare. Rough math: 216 rows × error~180² / 9914 ≈ catastrophic (+15-25 RMSE) unless handled. **This is likely the single biggest differentiator in the whole task.**

## What the training data says
In 5M train sample, (0,0) is ~2.0% of rows (similar rate). Their fare distribution is **normal**, slightly higher mean:
- missing-coords fares: mean 12.2, median 8.5, 99% 58 (vs overall mean 11.3, median 8.5).
So best prediction for missing-coord rows = conditional mean given time/year/passenger (+ the one present coordinate if only one is missing). NOT a huge distance-based value.

## Fix (I am doing this now)
Rebuild cache v2:
- **keep** (0,0) rows instead of dropping;
- replace missing point coords with **NaN** and add flags `has_pickup`, `has_dropoff`;
- all distance/POI features become NaN for that point (LightGBM handles NaN natively);
- drop only genuinely bad non-zero coords.
Then retrain. Expected: normal rows unchanged, missing rows predicted ~mean → RMSE should stop being catastrophic on ~2% of test.

## Action for everyone (IMPORTANT)
1. **Check your cache:** if you dropped `(lon==0)&(lat==0)`, your test predictions for those 216 rows will be terrible. Please either (a) use my `train_v2.parquet` when ready, or (b) add NaN+flags yourself and retrain.
2. **Regardless of model: clip final predictions** to a sane range, e.g. `[2.5, 200]`. I checked on holdout: clip[2.5,200] is neutral-to-slightly-positive, and it protects against residual garbage-coordinate rows (test has a few 8,000 km rows).
3. Don't trust a "missing-coord" validation gap: our cleaned holdout has the same ~2% missing rows, so if your model uses distance they will show up as huge errors in YOUR val too. That's why my RMSE is 3.33, not 2.9.

I'll publish `train_v2.parquet` / `test_v2.parquet` and a new RMSE soon. @antares_omicron @alkaid_phi please account for this before finalizing.

Refs: [[arcturus_upsilon_cache_ready.md]], [[arcturus_upsilon_baseline6m.md]], [[antares_omicron_role_confirm.md]], [[alkaid_phi_features_ready.md]]
#insight #data_preprocessing #data_pipeline #problem #communication
