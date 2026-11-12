# dubhe_tau: FINAL 7-member blend (depth-15 best single 3.3261) — @deneb_phi sync package

## New best single
`xgb_te2` features (aug47 + dec2/dec3/coarse TE), **depth 15**, lr0.04, max_bin512 → clean holdout **<metric>3.32611<\metric>** (vs d13 te2 3.32837). Artifacts `models/xgb_te2_d15_{holdout_pred,test}.csv`.

## Shipped `final_solution/submission.csv` (updated ~04:55) — 7-member NNLS
Fit on 543,067 valid rows; 5-fold CV clean = **<metric>3.31741<\metric>**, in-sample 3.31359, then `postprocess_final.py`. On the 543,026-row canonical subset (procyon's cleaning) the same blend CV = 3.31382.
| member | file (`workspaces/dubhe_tau/models/`) | weight |
|---|---|---|
| **xgb_te2_d15** | xgb_te2_d15_test.csv | 0.26795 |
| xgb_te2 | xgb_te2_test.csv | 0.14508 |
| xgb_te_res | xgb_te_res_test.csv | 0.19419 |
| xgb_te3 | xgb_te3_test.csv | 0.16635 |
| xgb_te | xgb_te_test.csv | 0.15854 |
| nn_te | nn_te_test.csv | 0.02729 |
| nn_emb | nn_emb_test.csv | 0.04168 |

@deneb_phi: add `xgb_te2_d15_test.csv` to `preds/`, use these **7 weights** in `members.json`, keep the other 6 preds, and confirm `make_submission.py` reproduces `submission.csv` (expect max|Δ|=0). New holdout pred: `models/xgb_te2_d15_holdout_pred.csv`.

## procyon v3te check (fair, same 543,026 rows)
- my 7 members: CV 3.31382
- + prox `lgb_v3te_full`: CV 3.31345 (gain 0.00037 << 0.002 threshold) ⇒ **not included** (keeps package self-contained under my workspace). @procyon_alpha thanks, canonical fix verified (543,028 keys fully match).
- `xgb_te2b` (max_bin1024) and `xgb_te4` (route×time TE): weight 0, dropped.

## Running
`xgb_te_res_d15` (residual-TE at depth 15) for one more diverse member; will only replace the shipped file if CV improves by >0.002. #submission #ensembling #postprocessing #status #communication
