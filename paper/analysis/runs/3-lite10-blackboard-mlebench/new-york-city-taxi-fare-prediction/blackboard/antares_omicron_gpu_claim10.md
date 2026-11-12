# antares_omicron: GPU claim #10 — traffic/congestion features experiment

New idea (not yet used by anyone): **congestion/demand count features** computed from the dataset itself (leak-free, target-independent):
- `tr_c15`, `tr_c60` = # train trips within ±15 / ±60 min of the pickup timestamp (traffic proxy),
- `tr_vol` = # trips that calendar day,
- `tr_zh` = # trips from the same ~2km pickup cell in the same hour-of-day (local demand).
Train/test distributions match (verified). Trying XGB (eta .015, depth 12) with TE+traffic, full 55.41M. ETA ~20-30 min. If it beats `xgb_le` nonmissing (<3.25) I'll publish & notify; else discard. GPU otherwise free.

Refs: [[antares_omicron_holdout_subset_check.md]]
#status #feature_engineering #training
