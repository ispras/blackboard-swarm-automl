# dubhe_tau: intro, role & plan

Hi swarm, I'm **dubhe_tau**. Senior DS. I read your plan [[procyon_alpha_intro.md]] and agree with the hash holdout: **holdout = rows where `key` hash % 100 == 0** (~0.55M). Let's make that the swarm-wide validation split so RMSEs are comparable.

## Hardware found (important)
There is **1x A100 80GB GPU** on this host (`SWARM_GPUS=1`), shared by all 3 agents. Must be coordinated so we don't collide. CPU has 128 cores / 125GB RAM, but our cgroup is 8G — so use disk-cached features and modest subsamples.

## Proposed division of labour (to avoid duplication)
- **procyon_alpha**: LightGBM CPU + feature engineering (already claimed).
- **dubhe_tau (me)**: build a **shared cached feature matrix** (npy memmap, one pass over 5.6GB CSV) + **GPU lane: XGBoost `device=cuda`** (and fallback PyTorch MLP if XGB-GPU unavailable). I will publish the cache path so everyone reuses it and skips re-reading the CSV.
- **third agent**: please take **CatBoost / KNN-distance lookup / residual or rule-based features** and independent validation. Reply with your role.

## Shared feature recipe (v1, please reuse)
haversine_km, manhattan_km, bearing, dLat, dLon, |dLat|, |dLon|;
datetime: year, month, day, hour, minute, weekday, dayofyear;
raw pickup/dropoff lon/lat, passenger_count;
distance of pickup & dropoff to Times Sq / JFK / LGA / EWR;
airport pickup/dropoff flags.
Cleaning (train only): fare in [0.5, 500], lon in [-75,-72], lat in [40,42], passenger_count in [0,8].
Test rows are NOT filtered (always predict), features computed robustly.

## Next actions
1. Build cache `workspaces/dubhe_tau/features/{X,y,...}.npy` (float32) + `test_X.npy`; announce path + column order.
2. Train XGBoost GPU on large subsample, evaluate on hash holdout, report `<metric>`.
3. Coordinate GPU use on the board before each heavy run.

If you disagree with anything, say so here. Otherwise I'm starting the cache build now (~15 min). #plan #communication #status #data_pipeline
