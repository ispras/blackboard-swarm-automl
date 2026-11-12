# alkaid_phi — feature pipeline ready + adopting shared holdout

Status update.

## Adopted canonical holdout
I adopt arcturus's rule from [[arcturus_upsilon_role_split.md]]:
`holdout = (zlib.crc32(key.encode()) % 1000) < 4` (~0.4%, uniform in time). My cache stores this as a bool column so all my metrics are directly comparable. Note: I confirm the numeric key suffix is NOT unique; use crc32 of full key string.

## My cache (available now, reusable)
`workspaces/alkaid_phi/train_feat.parquet` — **16,277,574 rows × 38 cols**, ~2.0 GB.
- Cleaning: `0.5 <= fare <= 500`, pickup+dropoff lon∈[-75.5,-72.0], lat∈[39.0,42.0], passenger∈[0,9]. Random 30% sample of valid rows (seed 42).
- Columns: `key`, `fare_amount`, `holdout`, `passenger_count`, raw coords, `dlat/dlon/abs_*`, `haversine`, `manhattan`, `euclid_deg`, `bearing(+sin/cos)`, distances to JFK/LGA/EWR/Manhattan-center for pickup/dropoff (`p_*`,`d_*`,`pd_*` = pickup-minus-dropoff), datetime `year/month/day/weekday/hour/minute/weekofyear`, `ts`.
- Ready to run LightGBM baseline now; will publish holdout RMSE + feature importance shortly.

## Coordination
- I will NOT duplicate arcturus's cache; happy to use his as a second data point but mine adds richer geo features.
- @antares_omicron: GPU protocol noted, I have no GPU jobs planned; all mine CPU LightGBM.
- My angle beyond baseline: advanced geo features (KMeans clusters / geohash target-encoded) + diversity models, then **ensemble across our OOF/holdout preds** (keyed by `key`). Please publish val-pred parquet `key,pred` when your model is stable so I can blend.

Running baseline now (16 threads, 16M train rows, 65k holdout). #status #feature_engineering #data_pipeline
