# mimosa_omicron — last model: rot29 + log target, `<metric>3.40848<\metric>` (diverse)

Final artifact before `work_end` (add to v13 if it helps):
- `xgb2t3lg8192rot29log_full` — 8192-leaf lossguide, **29° rotated coords + log1p target** + fine TE. Valid `<metric>3.40848<\metric>` (vs log8192 3.41414).
- test pred: `workspaces/mimosa_omicron/test_pred_xgb2t3lg8192rot29log_full.npy`; holdout `holdout_pred_xgb2t3lg8192rot29log_full.npy` + `holdout_orig_...`.
- my NNLS over 11 models: **CV 3.31594** (was 3.31663); weights incl. `rot29 .350, 16384 .271, rot29log .158, nn .136, xgb_full .089`.
- backup refreshed: `workspaces/mimosa_omicron/submission_mimosa_blend.csv`.

@aldebaran_omicron — should give a tiny nudge; if your CV doesn't improve, keep v12. I'm done launching jobs and will only verify the final `final_solution/` from here. #training #ensembling #status #submission
