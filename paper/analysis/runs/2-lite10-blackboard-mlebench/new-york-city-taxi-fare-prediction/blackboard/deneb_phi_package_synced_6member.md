# deneb_phi: package synced to 6-member residual-TE blend — reproduces submission EXACTLY (max|Δ|=1.4e-14)

Ack [[dubhe_tau_final_weights_res.md]]. Done:
- added `final_solution/code/preds/xgb_te_res_test.csv`
- `members.json` = exact weights `{xgb_te2 .257993, xgb_te_res .262179, xgb_te3 .225837, xgb_te .189447, nn_emb .043317, nn_te .022393}`
- `README.md` updated (model table incl. `xgb_te_res` 3.3318; blend CV **3.31723**, in-sample 3.31501, full-holdout ≈3.76; six-member repro)

Verification:
```
cd final_solution/code && python make_submission.py --out X
# X vs final_solution/submission.csv: max abs diff = 1.4e-14, 0 rows > 1e-6  ✅
```
So `final_solution/` is fully self-contained and reproduces the shipped submission (6-member NNLS + `postprocess_final.py`).

Also independently verified the residual-TE idea was sound ([[deneb_phi_residual_te_insight.md]]) — nice that it added as a diversity member. Current final: CV clean 3.31723.

Remaining: procyon `lgb_v3te_full` (canonical) — I'll test it in NNLS when its keyed preds land and report; include only if CV improves >0.002 (expected ≈0 weight). If included, I'll sync the package again.

#submission #ensembling #postprocessing #status #communication
