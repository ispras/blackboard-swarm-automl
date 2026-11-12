# aldebaran_omicron — FINAL CONFIRMATION (no further changes)

Per [[2026-11-04_07-11-45_SYSTEM_final_phase]]: experiments stopped; best agreed solution uploaded and frozen.

**`final_solution/submission.csv`**
- md5 `de8b927a68545ac12ef7eb5d30c23066`
- 9,914 rows, columns `key,fare_amount`; row order matches `data/test.csv` / `sample_submission.csv`
- 0 NaN, float dtype, range [3.363, 128.600], mean 11.472

**Final blend v13**: 5-fold bagged NNLS over 6 weighted full-data models (canopus `j54m`/`g54m` LightGBMs, mimosa `xgb2t3lg16384`/`8192rot29`/`8192rot29log` XGBs, mimosa `nn_full`), valid-coord CV `<metric>3.28889<\metric>`, corrupt 225 rows via canopus corrupt model (CV 12.65). Expected test ≈ 3.77.

`final_solution/` contains `submission.csv`, `README.md`, and `code/` (all agents' pipelines). Consensus confirmed by [[mimosa_omicron_final_validation]], [[canopus_pi_final_consensus]], [[canopus_pi_integrity_check]]. Done. #submission #status #communication
