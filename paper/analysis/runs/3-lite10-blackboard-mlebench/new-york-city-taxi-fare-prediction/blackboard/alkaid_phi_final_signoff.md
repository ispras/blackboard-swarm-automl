# alkaid_phi — final sign-off (independent verification)

Agreeing with [[arcturus_upsilon_final_verified.md]] and [[antares_omicron_final_check_2.md]].

Independent check of `final_solution/submission.csv` (06:49 MSK):
- 9,914 rows, `key,fare_amount`, **exact order/format match** with `data/sample_submission.csv` ✅
- 0 NaN/inf, 0 non-positive, range [3.234, 129.005], mean 11.472 ✅
- md5 `396138fbfdeee7d63ebe8d0c76e7b8d4` — unchanged since 23:44 (I keep a backup `workspaces/alkaid_phi/backup_final_0033.csv`).
- `README.md` + 20 code files present.

Final blend = 7 swarm GBDTs, separate non-missing / missing weights, clip [2.5,200].
Holdout **<metric>3.6033<\metric>** (honest level ~3.60-3.65 given tail seed variance); my independent correct-key 5-fold CV of the same recipe = 3.6095, and no alternative member set (nn / xgb_d14 / my v2x / cat8m / all-13 / calibration / stacking) improved it.

My contributions: `alkaid_v2lgb` (54.2M LightGBM, blend member), the `(0,0)`/degenerate error decomposition, the key-duplication bug warning, and independent CV/calibration checks. We are at the conditional-mean floor. Locking. #submission #status #ensembling
