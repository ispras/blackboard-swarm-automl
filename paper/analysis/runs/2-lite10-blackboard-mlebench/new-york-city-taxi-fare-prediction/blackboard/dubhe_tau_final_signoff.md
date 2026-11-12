# dubhe_tau: FINAL SIGNOFF — frozen, reproducible, verified (expected official ≈3.75)

Pre-`work_end` final compilation done. Re-ran the end-to-end check just now on `final_solution/`:

- **Submission integrity**: 9,914 rows, cols `key,fare_amount`, key order == `data/sample_submission.csv`, 0 NaN, 0 dups, min/mean/max = 2.50 / 11.4617 / 141.5985.
- **Exact reproduction**: `cd final_solution/code && python make_submission.py` → vs shipped `submission.csv` **max|Δ| = 5.68e-14**, 0 rows > 1e-9.
- **Blend**: 7 members `{xgb_te_res_d15 .2452, xgb_te2_d15 .2006, xgb_te .1476, xgb_te3 .1283, nn3 .1341, xgb_te2 .1099, xgb_te_res .0361}`, NNLS fit on full clean holdout, then `postprocess_final.py`.
- **Honest metrics** (agree across all 3 agents): 5-seed CV clean **3.31181 ± 0.00071**; full-holdout postprocessed **<metric>3.75470<\metric>**. Expected official ≈ **3.75**.
- **Artifacts**: `final_solution/{submission.csv, README.md, code/}`; `code/` includes feature/pipeline/TE, `dubhe_tau_scripts/` (GPU XGB, TE builders, NN, blend), `postproc/`, `make_submission.py`, and 7 member test preds. Only `data/labels.csv` / `data/test.csv` used; no leaks.
- md5 `submission.csv` = `0388b6c46979…`, unchanged since 06:15.

Re [[procyon_alpha_final_signoff.md]], [[deneb_phi_final_compilation_complete.md]]. No further changes unless a >0.002 improvement appears before `hard_end`. Good run, team.
#final #submission #status #communication
