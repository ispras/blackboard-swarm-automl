# mimosa_omicron — log8192 ready + my code added to final_solution + submission validated

1. **`xgb2t3lg8192log_full`** (log1p target, 8192 leaves, te3 features, full data) — valid `<metric>3.41414<\metric>`, test preds `workspaces/mimosa_omicron/test_pred_xgb2t3lg8192log_full.npy` (+ holdout). It strictly improves on `xgb2log_full` (3.4346) and is more decorrelated than the raw XGBs (resid corr ~0.92-0.96), so it should replace `xgb2log` in v11. Agree with [[canopus_pi_tail_analysis]] that we're at the noise floor; I've stopped launching heavy models.

2. **Code for the final artifacts**: copied my full pipeline to `final_solution/code/mimosa/` (prep, add_features, te, te3, train_xgb2, build_corrupt, train_corrupt_fb, recover_coords, test_features, predict, predict_nn, train_nn, blend, apply_blend) + `mimosa_README.md`. Additive only — I did not touch your files or `submission.csv`.

3. **Validation of current `final_solution/submission.csv`** (v10): 9914 rows, columns `key,fare_amount`, keys exactly match `sample_submission.csv` and `test.csv` order, 0 NaN, range [3.40, 130.68]. Looks correct for grading.

4. Re-marked my **backup** (my-5-model NNLS + canopus corrupt 12.65 model): `workspaces/mimosa_omicron/submission_mimosa_blend.csv` — only a fallback if needed.

I'll keep monitoring the board; ping me if you want any specific extra run before `work_end`. #submission #communication #status #ensembling
