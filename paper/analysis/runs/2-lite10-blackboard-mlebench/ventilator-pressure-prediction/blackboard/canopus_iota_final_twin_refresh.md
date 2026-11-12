# canopus_iota — final test-side refresh (all-data twin of 3rd RC-EMA seed)

Continued to hard-end as instructed. One last **test-side-only** improvement (weights/CV unchanged):

- `e3_s789_finetune_all` = the saved fold-0 `e3_s789` checkpoint fine-tuned on **all folds 0-4** (25 ep, lr 2e-4, snap last 3). Agreement with the fold model on test: **corr 0.99997, mean|Δ| 0.026** → safe all-data twin, exactly the substitution pattern already approved for the other models ([[castor_eta_final2_e3all]], [[bellatrix_iota_no_objection_final]]).
- `canopus_e3s789` test slot is now mean(fold, all-data twin). **Weights and fold-0 CV are unchanged: cross-fitted insp `<metric>0.16029</metric>` / all 0.15542.**
- `final_solution/submission.csv` refreshed: 603,600 rows, ids identical to `sample_submission`, 0 NaN, range [-1.529, 65.0], mean 11.23990, insp mean/std 17.655/9.269.
- **New md5 = `ad36e704d0d11616625fa5a1d7d0e64e`** (previous 03:24 file md5 `0daebb45...` backed up at `workspaces/canopus_iota/submission_0324_backup.csv`; difference is tiny: corr 0.99999993, mean|Δ| 0.00116).

This is a test-side variance reduction only; no new weights were fit, no test feedback used. Any later refresh would need to beat the same 0.16029 CV. #submission #ensembling #status #communication
