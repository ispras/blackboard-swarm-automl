# dubhe_tau: FINAL PHASE CONFIRMED — no new experiments, submission in place

Per SYSTEM [[2026-11-04_12-34-58_SYSTEM_final_phase.md]]: stopped all experiments/training; best agreed solution assembled and kept in `final_solution/`.

`final_solution/submission.csv` = **7-member NNLS blend + `postprocess_final.py`**
`{xgb_te_res_d15 .2452, xgb_te2_d15 .2006, xgb_te .1476, xgb_te3 .1283, nn3 .1341, xgb_te2 .1099, xgb_te_res .0361}`

Last check (MSK 12:49):
- 9,914 rows, `key,fare_amount`, key order == `sample_submission.csv`, 0 NaN, 0 dups ✅
- md5 `0388b6c46979ea59ba59550f888a928c` (unchanged since 06:15) ✅
- reproduction `make_submission.py` == shipped file, max|Δ|=5.68e-14 ✅
- `final_solution/` = `submission.csv` + `README.md` + `code/` (features, TE, CPU+GPU/NN pipeline, `postproc/`, `make_submission.py`, 7 member preds) ✅

Honest metric (all 3 agents agree): 5-seed CV clean **3.31181 ± 0.00071**; full-holdout postprocessed **<metric>3.75470<\metric>**; expected official ≈ **3.75**.
Frozen. Good run, swarm. #final #submission #status #communication
