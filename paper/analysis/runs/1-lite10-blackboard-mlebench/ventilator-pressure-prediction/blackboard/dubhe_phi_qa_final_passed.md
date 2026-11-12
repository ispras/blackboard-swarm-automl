# dubhe_phi — QA of `final_solution/` PASSED

Independent QA of the current `final_solution/submission.csv` (03:04 version, = sirius `sub_final3.csv`):
- **Rows/format**: 603,600 rows, columns `id,pressure` ✅
- **IDs**: exactly equal to `data/sample_submission.csv` order; 603,600 unique ✅
- **Values**: 0 NaN, 0 inf; range **[-1.553, 64.924]**; insp mean 17.649, exp mean 7.317 (match train) ✅
- **Byte-consistency**: identical to `workspaces/sirius_tau/sub_final3.csv` (max abs diff 0.0) ✅
- **Recipe reproducibility**: recomputed `w_insp @ test_pred` on insp rows and `w_exp @ test_pred` on exp rows from `workspaces/sirius_tau/manifest_final3.json` + `sub_final3_report.json` → matches `submission.csv` to **max abs 2.1e-14** ✅ (weights sum to 1.0 for both insp and exp)
- **Code**: `final_solution/code/` has fe2.py, fe3.py, train.py, infer.py, export_run.py, final_blend.py, finetune_full.py + manifests ✅; README states no external models used ✅

Minor reproducibility nit (non-blocking, grading only reads submission.csv): `final_solution/code/manifest_final.json` is the 01:54 manifest, while the shipped submission corresponds to `manifest_final3.json` / `sub_final3_report.json` (03:04). @[sirius_tau] if convenient, copy those two into `final_solution/code/` for exact provenance; otherwise fine.

**Recommendation: FREEZE / ship as-is.** Reported honest blend ~0.1605–0.1610 insp MAE, exp-hedge ~0.1551. My independent 16-member check was 0.1626 (before s777/s888/ft5); all estimates agree the ensemble is sound.
#submission #status #ensembling #communication
