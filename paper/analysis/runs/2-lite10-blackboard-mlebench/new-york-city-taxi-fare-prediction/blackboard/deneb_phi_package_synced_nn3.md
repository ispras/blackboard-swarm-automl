# deneb_phi: package synced to nn3 7-member final — reproduces submission EXACTLY (max|Δ|=5.7e-14)

Ack [[dubhe_tau_final_nn3.md]]. Done:
- `preds/`: replaced `nn2_test.csv` with `nn3_test.csv` (7 member files present).
- `members.json` = exact final19 weights:
  `xgb_te_res_d15 .245200, xgb_te2_d15 .200645, xgb_te .147613, nn3 .134068, xgb_te3 .128333, xgb_te2 .109876, xgb_te_res .036052`.
- `README.md`: nn3 (3.4217) in model table; blend CV **3.31294** (5-seed 3.31189), in-sample 3.31013, full-holdout ≈**3.753**.

Exact reproduction verified:
```
cd final_solution/code && python make_submission.py --out X
# X vs final_solution/submission.csv : max|Δ| = 5.7e-14, 0 rows > 1e-9   ✅
```
`final_solution/` = `submission.csv` + `README.md` + `code/` (CPU pipeline + `dubhe_tau_scripts/` GPU/TE/NN + `postproc/` + 7 member preds) reproducing the shipped file. @procyon_alpha ready for your re-check (expect ≈3.753).
@dubhe_tau if another NN/seed clears >0.002, ping me and I'll re-sync in ~2 min; otherwise this is frozen.

#submission #final #ensembling #status #communication
