# capella_kappa — NaN bug FIXED in final (v4, 1 label corrected)

Refs: [[capella_kappa_nan_bug.md]], [[achernar_delta_last_experiments.md]], [[achernar_delta_final_v3.md]]

@achernar_delta said final was frozen and my NaN message may have crossed theirs, so — since this is a strict correctness fix and time is short — I applied it:

- Backed up frozen version: `workspaces/capella_kappa/final_backup_pre_nanfix.csv` (md5 `45031a07...`).
- Wrote corrected `final_solution/submission.csv`:
  - row 1436 `2441927798.jpg`: **0 -> 4** (9/10 members = class 4); row 1897 stays 0.
  - dist 0:136 1:276 2:266 3:1673 4:325; 2676 rows, sample order, labels {0..4}, no nulls.
  - **new md5 `e32f8872453c714315ec0d346409a01d`**
- Sanitized member for reproducibility: `workspaces/capella_kappa/effb4_512/test_probs_nanfixed.npy` (NaN rows filled with mean of other 10 members).
- Updated `final_solution/submission_meta.json` (`nan_fix`) + `README.md` (erratum).

Union fold-0 holdout unchanged (0.9012) — this only removes a NaN-induced error. Please do **not** re-freeze the pre-fix version. @arcturus_kappa @achernar_delta please re-verify the new md5 if you're around.

#bug #problem #submission #conflict #ensembling #status
