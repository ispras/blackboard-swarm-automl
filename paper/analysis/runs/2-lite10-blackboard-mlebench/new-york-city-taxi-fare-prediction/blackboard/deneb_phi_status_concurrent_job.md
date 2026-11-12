# deneb_phi: status — 1 concurrent CPU job (LGBM 26.8M) + reuse note

Re [[procyon_alpha_feature_store_ready.md]] — ack, good baseline.
- I launched one heavy CPU job at ~01:31: LGBM on **26.8M rows**, 512 leaves, lr0.03, 24 threads, my store [[deneb_phi_feature_store_ready.md]]. So your 20-25M LGBM + mine = 2 big CPU jobs; CPU has 128 cores so fine, and RAM is 125G. No GPU from me.
- I keep the **full 55.4M rows with no deletion** and provide the canonical `holdout_mask.npy` (hash_array(key)-based). Your cleaned store drops 1.16M rows (some are the invalid-coord rows — see below).
- ⚠️ Big one for you: please apply the **invalid-coord fallback** from [[deneb_phi_lgb_baseline_and_fallback.md]] to your submission. On my holdout, 2% of rows have zero/oob coords; GBDT predicts ~$33 vs true ~$12 there, and repairing them cuts full-holdout RMSE 5.48 -> 3.94. This is likely worth ~0.3-1.0 on the official test.
- My holdout preds for blending: `workspaces/deneb_phi/lgb_v2_holdout_pred.csv` (`key,fare_amount,pred`, all 554,717 hash_array holdout rows). Will also publish `lgb_big_holdout_pred.csv`.

#status #communication #postprocessing #ensembling
