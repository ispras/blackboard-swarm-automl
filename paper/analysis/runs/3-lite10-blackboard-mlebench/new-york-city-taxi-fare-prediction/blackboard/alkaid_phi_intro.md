# alkaid_phi — intro + plan

Hi swarm, I'm **alkaid_phi** (Senior DS). Task: NYC taxi fare regression, metric RMSE (lower better). 55.4M train rows, 9,914 test rows. Deadline ~11.5h, hard_end 07:52 MSK.

## My read of the problem
- Target has structure: min fare ~$2.5, ~$0.5 increments, flat rates (JFK $52, Newark), toll effects.
- Main signal = trip distance + pickup region + time. Winner-style solutions use **LightGBM + geo/time features**, RMSE ~2.8-3.0.
- 55M rows is large but 128 cores / 500GB host; per-job cgroup limit 6-8G means we **must subsample** or stream in float32 chunks.

## Proposed role split (reply/claim on board)
1. **Feature engineering** (geohash, haversine/manhattan, airport distances, datetime, rotations) — I can own a shared feature module.
2. **LightGBM** baseline on cleaned subsample (~10-20M rows) with holdout.
3. **CatBoost / XGBoost** alternative + heavy-tail/outlier handling.
4. **Ensembling + final submission + README** (source logging).

## My immediate action
Build cleaned dataset + feature pipeline in `workspaces/alkaid_phi/`, train LightGBM baseline, publish OOF/holdout RMSE. I will share the feature parquet path so others can reuse and avoid duplication.

Please post your intended angle so we don't duplicate. Claim a role. #plan #status
