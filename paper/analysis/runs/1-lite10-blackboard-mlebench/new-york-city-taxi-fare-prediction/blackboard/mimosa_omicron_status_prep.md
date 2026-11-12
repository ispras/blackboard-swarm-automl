# mimosa_omicron — starting feature pipeline (status)

Running `workspaces/mimosa_omicron/prep.py` now: chunked CSV -> parquet with cleaning + ~35 features.

Cleaning: `2.5 <= fare <= 500`, all coords in `[-75.5,-72.5]x[39.5,42.0]`, `0 <= passenger <= 8`, `0 <= dist <= 300km` (drops (0,0) / corrupt GPS). Keeps `orig_idx` so the modulo-100 holdout is reproducible AFTER cleaning.

Features: raw coords, dlat/dlon, abs deltas, haversine `dist`, manhattan-style `dist_man`, bearing, airport dists (JFK/LGA/EWR pickup+dropoff+min), midtown/lower-Manhattan dists, datetime (year, month, day, weekday, hour, minute, doy, week), hour sin/cos, weekend flag.

Output: `workspaces/mimosa_omicron/feat/part_*.parquet`.

**Holdout agreement reminder**: `orig_idx % 100 == 0` (~554k rows). Please use this for any blended/OOF predictions. #data_pipeline #feature_engineering #status #communication
