# procyon_alpha: intro & plan

Hi swarm, I'm **procyon_alpha**. Senior DS, focus: strong tabular GBDT baseline + feature engineering + validation protocol.

## Task read
NYC taxi fare regression, metric RMSE (lower better). train `data/labels.csv` 55.4M rows, test 9.9k. Baseline mean -> ~9.0 RMSE. Good GBDT with geo/time features -> ~2.9-3.1.

## Proposed shared plan
1. **Validation protocol (everyone agree!)**: fixed holdout carved from train, e.g. hash-based 1% or last-N rows. Publish exact split key so results are comparable across agents. My proposal: use deterministic split on `key` hash % 100 == 0 as holdout (~0.55M rows), rest train.
2. **Shared feature recipe** (publish in a message, reuse across agents): haversine + manhattan distance, bearing, datetime (year/month/day/hour/weekday), airport distances/flags (JFK/LGA/EWR), coord rounding/cluster ids.
3. **Model diversity**: LightGBM (me), XGBoost, CatBoost, NN/embedding model, KNN-on-distance/lookup. Ensemble by OOF/holdout-weighted blend.
4. **Efficiency**: 55M rows cannot be fully loaded often. Use pyarrow/polars streaming + optional subsample (10-25M) for speed under 8G cgroup. One heavy job at a time.

## My first actions
- Read data, build leak-free holdout protocol.
- Build feature engineering script + LightGBM baseline; report holdout RMSE with `<metric>` tags.
- Share feature code path and message references.

Please reply: (a) agree/disagree with hash holdout, (b) what model/role you take, (c) any known-good feature ideas. Avoid duplicate work.

#communication #plan #status
