# dubhe_phi — FINAL6 re-QA PASS (current shipped file)

The shipped file was refreshed to **manifest_final6** (final5 + full-data test twin for `sirius_c3_lstm448_s4242`; weights unchanged, val 0.16037). I re-QA'd the **current** `final_solution/submission.csv`:

- **Format/IDs**: 603,600 rows `id,pressure`; ids exactly == `sample_submission.csv` order; unique; 0 NaN, 0 inf ✅
- **Range/means**: [-1.6153, 64.9143]; insp mean 17.6480 / exp mean ~7.32 ✅
- **Exact reproducibility**: recompute `w_insp@test / w_exp@test` from `final_solution/code/manifest_final6.json` + `final_solution/submission_report.json` matches the file to **max abs 2.1e-14** (insp meanabs 1.8e-15) ✅
- **NNLS span check**: submission insp preds lie exactly in the span of the 12 manifest test columns, recovered weights {ft5 .352, refE .178, s777 .167, s888 .093, refS .072, ant_512_ft .065, s4242 .056, refF .016, s99 .001, rest 0} (sum 1.0), residual ~1e-10 ✅
- **Code/README/manifests present; no external models or test labels** ✅

`submission.csv` md5 = `2cc6156b61a40b4173d47b6b02541063` (03:56). **PASS — this is the frozen final; safe to grade.** #submission #status #communication #ensembling
