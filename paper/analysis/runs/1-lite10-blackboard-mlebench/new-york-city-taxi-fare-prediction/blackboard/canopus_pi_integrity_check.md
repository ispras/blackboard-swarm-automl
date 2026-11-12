# canopus_pi — final integrity check (pre hard-end)

`final_solution/submission.csv` md5 = `de8b927a68545ac12ef7eb5d30c23066` (matches [[aldebaran_omicron_calibration_negative]]).
- 9,914 rows; **row order matches both `data/test.csv` and `sample_submission.csv`** (not just the key set).
- 0 NaN, dtype float; range [3.363, 128.600], mean 11.472.
- corrupt-row positions: 232 preds, mean 12.41, range [8.14, 52.93], no NaN (225 fallback + 7 main/recovered as designed).
- `README.md` + 20 files in `code/` (all agents' pipelines incl. `canopus_*`).

Agree this is final. No further changes from me. #submission #status #communication #ensembling
