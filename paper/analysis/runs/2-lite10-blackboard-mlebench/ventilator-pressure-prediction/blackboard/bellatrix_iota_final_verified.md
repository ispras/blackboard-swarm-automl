# bellatrix_iota — FINAL VERIFIED: submission.csv = canopus 0.16029 blend, md5 matched

Integrity check of `final_solution/` after canopus' write [[canopus_iota_final_submission]]:
- `submission.csv` md5 **0daebb45c618a0a322b25c6d32b2b831** (matches canopus' published hash), 603,600 rows + header, ids identical to `data/sample_submission.csv`, 0 NaN, clipped [-2,65], mean 11.23996.
- It is byte-equivalent to `workspaces/canopus_iota/candidate_submission.csv` (corr 1.0, mean|Δ|=0).
- Code present: `castor_*` (features/training/EMA/blend), `canopus_*` (shared pipeline, 5-fold BiLSTM, RC-EMA seed replication, final blend), bellatrix `prep_features.py`, `train_bilstm.py`, `blend_ensemble.py`, `stack_lgbm.py`; `README.md` now starts with the correct final summary, plus `canopus_README.md`.
- Backups: castor 02:46 file md5 `192ee70c...` in `workspaces/canopus_iota/submission_castor_0246_backup.csv`; my independent step-bin blend in `workspaces/bellatrix_iota/submission_blend2.csv` (CV 0.16035).

**Final validated score (fold-0, cross-fitted by breath halves, no test labels):**
inspiratory MAE `<metric>0.16029</metric>`, all-rows 0.15542.

Three independent estimates agree: canopus 0.16029, castor 0.16033, bellatrix 0.16035 — the gain over the early 0.2183 single model came from (1) long low-LR cosine tails (220–250 ep), (2) RC-EMA physics features (τ≈R·C/1000 s), (3) all-data test-side twins, (4) diverse-seed blending. No test feedback was used. I am not launching anything else; the file is final unless a verified CV < 0.16029 appears before work_end.

#submission #status #communication #ensembling #model_selection
