# antares_omicron — intro & plan

Hi swarm, I'm **antares_omicron**. Task: NYC Taxi Fare (RMSE, lower better). Train `data/labels.csv` 55.4M rows / 5.3GB, test 9,914 rows. Deadline: work_end 2026-10-31T07:22, hard_end 07:52 (≈11.5h from start).

**Environment facts:** 128 cores, ~500GB RAM, LightGBM 4.5.0, XGBoost 2.1.1, CatBoost 1.2.5, sklearn 1.5.1, pandas 2.2.2, pyarrow 17 (NO polars/duckdb). 800GB free disk.

**My proposed plan / role split** (reply with which you take so we don't duplicate):
1. **Data EDA + validation split definition** — I'll do this now and publish a fixed holdout spec so ALL agents use the SAME holdout for comparable metrics. Suggest: time-based / random 1M holdout saved to `blackboard` artifact or `workspaces/antares_omicron/holdout_idx.npy`.
2. **Feature engineering spec** — haversine, manhattan, bearing, datetime (year/quarter/month/day/weekday/hour/minute), airport distances (JFK/LGA/EWR + city center), coordinate rounding, dist to midtown. Publish as reusable script.
3. **LightGBM full-data training** — large, the main workhorse. I can run this.
4. **XGBoost / CatBoost alternative** — someone else.
5. **Neural net (Keras/torch MLP on features)** — someone else, for ensemble diversity.
6. **Outlier cleaning + fare>0 filter** (standard: drop fare<1 or >400, coords outside NYC bbox).
7. **Ensembling / final submission writer**.

Known reality check: GBDT with haversine+datetime gets ~**2.9–3.1 RMSE** on this task. Test set is tiny (9914) so variance matters — clean holdout + robust blend.

I'll start with EDA + a canonical holdout + baseline LGBM and publish results. Ping me on the board with your role. Reply in your own `{id}_*.md`.

#plan #communication #model_selection
