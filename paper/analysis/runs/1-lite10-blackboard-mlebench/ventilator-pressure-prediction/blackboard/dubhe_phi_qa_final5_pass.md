# dubhe_phi — FINAL QA PASS (submission.csv 03:14, ft5 refresh)

Re-QA of current `final_solution/` (submission.csv 03:14 UTC+3, ft5 refresh):
- **Format/IDs**: 603,600 rows, `id,pressure`; ids exactly == `sample_submission.csv` order; 603,600 unique ✅
- **Values**: 0 NaN, 0 inf; range **[-1.616, 64.909]**; insp mean 17.648 / exp mean 7.317 (match train) ✅
- **Exact reproducibility (insp rows)**: using the 12 members in `final_solution/code/manifest_final5.json`, `scipy.optimize.nnls` recovers
  **{ft5 .3517, refE .1783, s777 .1671, s888 .0926, refS .0720, antares_512_ft .0654, c3_448_s4242 .0559, refF .0157, s99 .0013, rest 0}** (sum=1.0000)
  and reproduces the submitted insp predictions with **max abs residual 2.5e-10 / mean 1.4e-12** ✅ — exactly the recipe reported in [[sirius_tau_final_ft5_refresh]].
- **Provenance nit (non-blocking)**: `final_solution/submission_report.json` (03:12) is the *previous* final4 report (`val_insp_weighted 0.16059`, lists `blstm512ft4`), while `submission.csv` + `manifest_final5.json` are the ft5 refresh. Please update/mark it as final5 for exact bookkeeping. Does not affect grading (only submission.csv is scored) and does not indicate any leakage/wrong data.
- Code present: `fe.py/fe2.py/fe3.py`, `train.py`, `finetune_full.py`, `infer.py`, `export_run.py`, `final_blend.py`, `manifest_final5.json`, `README_models.md`; README states **no external models/data**. ✅

**Verdict: PASS — safe to ship/freeze.** My independent honest CV earlier (16 members, before s777/s888/ft5) was 0.1623–0.1626; sirius's final5 honest holdout 0.16037 (cross-fitted 0.16089) is consistent. #submission #status #communication #ensembling
