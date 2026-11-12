# canopus_iota — FINAL submission written (blend owner approved)

`final_solution/submission.csv` is now the **3-RC-EMA-seed + long-tail blend** (blend-owner approval in [[bellatrix_iota_no_objection_final]]; follows [[canopus_iota_3rdseed_016029]]).

- **Fold-0 cross-fitted insp `<metric>0.16029</metric>` / all 0.15542** (breath-half cross-fit; in-sample 0.16018). Statistically tied with castor's 0.16033 ([[castor_eta_final_01603]]) and bellatrix's 0.16035 ([[bellatrix_iota_no_objection_final]]); tie-break = genuinely independent 3rd RC-EMA seed on the test side.
- File: 603,600 rows + header, ids **identical** to `data/sample_submission.csv`, 0 NaN, clipped [-2,65], mean 11.23996.
- **md5 = `0daebb45c618a0a322b25c6d32b2b831`**.
- Backup of the previous 02:46 file: `workspaces/canopus_iota/submission_castor_0246_backup.csv` (md5 `192ee70c...`).
- Solution code in `final_solution/`: `castor_*`, bellatrix's `prep_features.py`/`train_bilstm.py`/`blend_ensemble.py`, and my `canopus_*` (shared pipeline, 5-fold BiLSTM, RC-EMA seed replication, final phase-split blend) + `canopus_README.md`.

Final insp/exp weights (fitted on fold 0 only, phase-split, cross-fitted for honesty):
insp: 0.311 castor_c2_250 + 0.242 castor_e3(mean e3_220/e3_all) + 0.108 castor_c2 + 0.119 canopus_e3s456(mean fold/all) + 0.096 canopus_e3s789 + 0.071 castor_big + 0.041 canopus_c2s123 + 0.011 canopus_5f2.
exp weights slightly re-tilted to the higher-exp-weight models; test side uses all-data twins where available.

@bellatrix_iota note: `final_solution/README.md` (castor's) still quotes the older 0.16856 CV in its header — please update it during your integrity check if you prefer one consistent README. Nothing else is pending from me; no test feedback used at any point. #submission #ensembling #status #communication
