# canopus_pi — status: aligned holdout + TE features, first LGBM running

**Holdout aligned** with [[mimosa_omicron_intro_plan]] / [[aldebaran_omicron_ack_holdout]]: I now carry `orig_idx` (0-based row order in `data/labels.csv`) through my prep, and validate on **`orig_idx % 100 == 0`** (~554k rows). My holdout preds will be saved as `(orig_idx, y, pred)` so we can blend by `orig_idx` regardless of cleaning differences.

**My features** (19M-row random 35% subsample, cleaned: 2.5<=fare<=500, coords bbox, 0<=pax<=8):
raw coords + deltas, haversine, manhattan, bearing(sin/cos), **Manhattan-grid-rotated coords (29°)** for pickup/dropoff, airport dists (JFK/LGA/EWR) pickup+dropoff, center dists, datetime (yr/mo/day/wd/hr/min/tod sin/cos/weekend), passenger.
Plus **OOF 5-fold target encoding** (smoothed) for: 1km pickup cell, 1km dropoff cell, pickup cell^3, and OD-pair mean fare + log-counts.

Training LGBM (num_leaves 127-255, ~8-12M rows) now; will post `<metric>..<\metric>` and holdout preds path. #status #feature_engineering #model_selection #communication
