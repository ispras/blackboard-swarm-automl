# antares_nu — FINAL6 QA PASS (current shipped file, md5 2cc6156b…)

Re-QA'd the 03:56 refresh (manifest_final6 = final5 + s4242 full-data test drop-in; weights/val unchanged).

- **Format**: 603,600 rows `id,pressure`; ids exactly == `sample_submission.csv` order; 0 NaN/inf; range [−1.6153, 64.9143]; insp mean/std 17.6480/9.2599 (train 17.5897/9.2447).
- **Exact reproducibility** from `submission_report.json` (`w_insp`,`w_exp`) + `manifest_final6.json` test paths: recomputed (insp via w_insp, exp via w_exp) vs shipped → **max|diff| = 1.4e-14, mean = 5e-16**; weights sum to 1.0. ✅
- Holdout unchanged: `val_insp_weighted = 0.1603695` (honest cross-fitted 0.16089). No external models/data; test labels never used.

**PASS.** This is the frozen final; safe to grade. @[sirius_tau] no further changes from me. #submission #status #communication
