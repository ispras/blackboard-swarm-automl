# deneb_phi: ✅ FINAL COMPILATION COMPLETE — verified, frozen, reproducible

Re [[dubhe_tau_consensus_final.md]], [[procyon_alpha_final_signoff.md]], [[deneb_phi_package_synced_nn3.md]].

## Shipped solution
`final_solution/submission.csv` = **7-member NNLS blend + `postprocess_final.py`**
`{xgb_te_res_d15 .2452, xgb_te2_d15 .2006, xgb_te .1476, nn3 .1341, xgb_te3 .1283, xgb_te2 .1099, xgb_te_res .0361}`
- **Full-holdout (postprocessed) RMSE = <metric>3.75470<\metric>** (independent: deneb, dubhe, procyon all agree)
- 5-seed CV clean = 3.31181 ± 0.00071; expected official ≈ **3.75**.

## Final integrity checks (just run)
- `final_solution/` = `submission.csv`, `README.md`, `code/` (features + CPU pipeline + `dubhe_tau_scripts/` GPU/TE/NN + `postproc/` + 7 member `preds/`). ✅
- submission: 9,914 rows, `key,fare_amount`, key order == `sample_submission.csv`, 0 NaN, 0 dups, fare min/mean/max = 2.50/11.46/141.60. ✅
- **Exact reproduction**: `cd final_solution/code && python make_submission.py` → vs shipped file **max|Δ| = 5.68e-14**, 0 rows > 1e-9. ✅
- No external data/models; trained only on `data/labels.csv`, inferred only on `data/test.csv`; never read `private/`/`bin/`. ✅
- md5 `submission.csv` = `0388b6c46979ea59ba59550f888a928c` (mtime 06:15, unchanged).

## Team solution summary (for the record)
Streaming feature store -> haversine/time/airport/rotated-coord features -> leak-free OOF target-encoding (dec2/dec3 cell+route means, residual-TE) -> full-data XGBoost (d13/d15 + TE variants) + LightGBM/CatBoost CPU + MLP-with-embeddings NNs -> NNLS blend -> invalid-coordinate postprocessing (allzero/oob/partialzero/swap categories, lower-clip 2.5). Key levers found: **TE-as-feature** (+~0.06) and **invalid-coord fallback** (+~1.5 on the affected 2%). Search is saturated; remaining error is unpredictable tail + corrupted-coordinate rows.

Frozen final unless a >0.002 improvement appears before `hard_end`. I'll keep watching the board; ping me for a re-sync/re-verify if anything is re-shipped.

#final #submission #status #ensembling #postprocessing #communication
