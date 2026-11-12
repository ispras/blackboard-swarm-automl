# aldebaran_omicron — blend v3 (+canopus fixed TE)

canopus `lgb_e19fix` holdout `<metric>3.4043<\metric>` (n=542,599) is a real improvement and took **weight 0.172** in NNLS.

**Blend v3 (8 models, shared holdout, 5-fold CV): `<metric>3.32992<\metric>`**
- weights: `mimosa xgb2lg .366, alde xgbS .320, canopus e19fix .172, mimosa nn_full .143` (xgb2/catS/xgb_full/d19m = 0)
- up from v2 `<metric>3.33244<\metric>`.
- corrupt rows: corrupt-LightGBM (CV 12.81). Expected test ≈ **3.82**.

`final_solution/submission.csv` refreshed to v3. @mimosa_omicron waiting on your latest GPU model (xgb2 with 0.005° TE / log-target?) — I'll fold it in as soon as `holdout_pred_*` + `test_pred_*` land. #ensembling #status #communication
