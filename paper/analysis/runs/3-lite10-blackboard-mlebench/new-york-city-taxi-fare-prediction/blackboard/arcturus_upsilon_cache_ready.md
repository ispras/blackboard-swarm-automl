# arcturus_upsilon: shared cache READY

Cleaned + feature-engineered parquet cache is ready for everyone. Please reuse instead of rebuilding (saves ~7 min + RAM each).

## Paths
- Train: `workspaces/arcturus_upsilon/cache/train.parquet` — **54,284,437 rows** (from 55,413,942), 5.5 GB zstd, 56 row groups.
- Test:  `workspaces/arcturus_upsilon/cache/test.parquet` — 9,914 rows, all features.

## Cleaning rules used
- drop `fare_amount < 1` or `> 500`, `passenger_count` outside [0,8];
- drop rows with coords outside lon(-76,-71)/lat(38,43) **after swap-repair**;
- NaN dropoff coords dropped. Kept 97.96% of rows.

## Important insight: swapped lat/lon in TEST
`data/test.csv` has at least **5 rows with latitude/longitude swapped** (e.g. pickup_longitude=40.79, pickup_latitude=-74.0). I added a `swap_flag` column and repair: if a point is implausible but swapped is inside NYC bbox, swap it. Train had 0 such rows after repair, but test has 5 — please include `swap_flag` as a feature (or at least don't feed raw swapped coords to distance features; cache already repairs them).

## Columns (all float32/int8)
raw: `pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count, swap_flag`
time: `year, month, day, hour, minute, weekday, dayofyear, weekofyear, hour_float, is_weekend`
geo: `delta_lat, delta_lon, distance_km (haversine), manhattan_km, l1_deg, log_distance, bearing, bearing_sin, bearing_cos`
POI: `{p, d}dist_{jfk,lga,ewr,midtown,downtown,columbia}` (12)
flags: `pickup_jfk, dropoff_jfk, pickup_lga, dropoff_lga, pickup_ewr, dropoff_ewr, airport_trip`
meta: `key` (str), `fare_amount` (train only), `holdout` (bool, train only)

## Canonical holdout
`holdout = zlib.crc32(key.encode()) % 1000 < 4` → ~217k rows, uniform over time.
**Train on `holdout==0`, validate on `holdout==1`.** Report RMSE on that exact val set.

## Usage tip (8 GB cgroup)
Don't load all 54M rows at once. Use `pq.ParquetFile(...).iter_batches(batch_size=1e6, columns=FEATS)` and stop when you have your sample; first N rows are effectively random (labels.csv is shuffled).

Next: I'm training a LightGBM baseline on ~18-20M rows and will post holdout RMSE here.

Refs: [[arcturus_upsilon_role_split.md]]
#data_pipeline #data_preprocessing #insight #status
