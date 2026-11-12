# antares_omicron: congestion/traffic count features — NEGATIVE result (no gain)

Tested new leak-free features from the dataset itself: `tr_c15/tr_c60` (# trips within ±15/±60 min), `tr_vol` (# trips that day), `tr_zh` (# trips from same ~2km pickup cell in same hour-of-day). Train/test distributions matched. XGB (eta .015, depth 12, TE+traffic, full 55.41M):

| model | all | clean(dist≥.1) | degenerate | missing |
|---|---|---|---|---|
| xgb_le (current best) | 3.6605 | **2.7772** | 13.405 | 12.321 |
| xgb_t (TE + traffic) | 3.6636 | **2.7764** | 13.442 | 12.342 |

Clean subset difference is −0.0008 (noise). → **Congestion/demand is already captured by hour/dow/zone; traffic counts add nothing.** Discarding `xgb_t`. This closes the last feature idea; we are firmly at the floor.

GPU free (claim #10 done). Final stands at 3.6033 (current `final_solution`). I'll continue monitoring and re-verify the submission near hard_end.

Refs: [[antares_omicron_holdout_subset_check.md]], [[arcturus_upsilon_final_confirmed.md]]
#feature_engineering #insight #status #model_selection
