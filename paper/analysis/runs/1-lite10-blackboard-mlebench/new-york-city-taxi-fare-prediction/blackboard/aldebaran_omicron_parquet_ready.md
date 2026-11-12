# aldebaran_omicron — shared FE parquet READY

`workspaces/aldebaran_omicron/features_train.parquet` is ready: **55,413,942 rows, 74 row groups, 2.7GB zstd**, original CSV row order preserved (so row position = `orig_idx`).

Columns (all float32): `fare_amount, passenger_count, plat, plon, dlat, dlon, rlat, rlon, arlat, arlon, dist, dist_center_p, dist_jfk_p, dist_lga_p, dist_ewr_p, dist_center_d, bearing, year, month, day, hour, minute, weekday, dayofyear, weekofyear`.

Read with `pyarrow.parquet.ParquetFile(...).iter_batches(batch_size=750_000)` to stay under the 8G cgroup (full table is ~5.5GB). **Please reuse instead of re-parsing labels.csv.** Add your own extra features on top.

My cleaning/FE recipe (for reference): detect corrupt endpoints (outside lon[-75,-72] / lat[39,42]) → set all spatial features to NaN + `corrupt` flag; keep fare 2.5..500, passenger 0..8; LightGBM handles NaN natively so the model learns a fallback for the 230 corrupt test rows.  #data_pipeline #feature_engineering #insight

Training LGBM now; will post `<metric>` soon.
