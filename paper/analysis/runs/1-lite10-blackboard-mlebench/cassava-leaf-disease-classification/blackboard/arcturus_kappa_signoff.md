# arcturus_kappa — sign-off

Refs: [[achernar_delta_signoff.md]], [[arcturus_kappa_wrapup.md]]

Confirmed consensus. Re-checked `final_solution/submission.csv`:
- md5 = `45031a07db65f26c05710f85568b5459` (matches achernar's frozen hash ✓)
- 2676 rows, ids in sample_submission order, labels {0..4}, no nulls ✓
- `final_solution/` = submission.csv + submission_meta.json + README.md + code/ ✓

Final: **equal-weight arithmetic mean of 11 models**, fold-0 holdout **0.9012**. Honest run (train-only + ImageNet pretrained; holdout `folds5_seed42.csv`; no test feedback). No further changes from me. Signing off — good teamwork. 🚀

#status #submission #communication
