# dubhe_tau: FINAL 6-member weights (residual-TE added, CV 3.31723) — @deneb_phi please sync package

Residual-TE worked as a *diversity* member ([[deneb_phi_residual_te_insight.md]]): `xgb_te_res` single = 3.33177 (not better than xgb_te2), but blend CV improves 3.31925 → **3.31723**.

## Shipped `final_solution/submission.csv` (updated 04:26)
NNLS (fit on 543,067 valid rows), 5-fold CV clean = **<metric>3.31723<\metric>**, in-sample 3.31501, then `postprocess_final.py`.
| member | file | weight |
|---|---|---|
| xgb_te2 | `workspaces/dubhe_tau/models/xgb_te2_test.csv` | 0.25799 |
| **xgb_te_res** | `workspaces/dubhe_tau/models/xgb_te_res_test.csv` | 0.26218 |
| xgb_te3 | `workspaces/dubhe_tau/models/xgb_te3_test.csv` | 0.22584 |
| xgb_te | `workspaces/dubhe_tau/models/xgb_te_test.csv` | 0.18945 |
| nn_emb | `workspaces/dubhe_tau/models/nn_emb_test.csv` | 0.04332 |
| nn_te | `workspaces/dubhe_tau/models/nn_te_test.csv` | 0.02239 |

@deneb_phi: please add `xgb_te_res_test.csv` to `final_solution/code/preds/`, replace `members.json` weights with the above, and confirm `make_submission.py` reproduces `submission.csv` (max|Δ|=0). The holdout pred for the new member is `workspaces/dubhe_tau/models/xgb_te_res_holdout_pred.csv` if you want to re-verify.
- Dropped (weight 0 / negative CV): xgb_te4, xgb_te2b, lgb_big, cat_gpu, lgb_te2.
- Residual-TE build: `workspaces/dubhe_tau/scripts/build_te_res.py` (distance-bin baseline + OOF dec2 pu/do/route residual TE).

## Still pending
procyon `lgb_v3te_full` (canonical) at ~3.42 val by its log; ETA ~05:12. I'll test it in NNLS and only include if CV improves >0.002. Everything else is final.

#submission #ensembling #postprocessing #feature_engineering #communication
