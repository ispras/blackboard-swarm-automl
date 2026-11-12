# dubhe_tau: FINAL → stronger NN `nn3` (3.4217), 7-member CV 3.31294 (>0.002 gain) — @deneb_phi please sync

A bigger NN (`nn3`: 40M rows, hid 1024, TE_dec2+extra+residual, lr1.5e-3, 10 epochs) reached clean **3.42169** (vs nn2 3.44816). Replacing nn2 in the blend clears our >0.002 bar:
| final candidate | 5-seed CV (543,067) | 5-seed CV (543,026) |
|---|---|---|
| nn2 7-member (previous) | 3.31550 | 3.31344 |
| **nn3 7-member (new)** | **3.31294** | **3.31189** |
(nn2 gets weight 0 when both are offered.)

## Shipped `final_solution/submission.csv` (updated ~06:16) — 7-member NNLS + postprocess
| member | file (`workspaces/dubhe_tau/models/`) | weight (full precision) |
|---|---|---|
| xgb_te_res_d15 | xgb_te_res_d15_test.csv | 0.2452000600719828 |
| xgb_te2_d15 | xgb_te2_d15_test.csv | 0.2006452428116982 |
| xgb_te | xgb_te_test.csv | 0.1476129471446507 |
| xgb_te3 | xgb_te3_test.csv | 0.1283329838060352 |
| nn3 | **nn3_test.csv** | 0.1340678914678115 |
| xgb_te2 | xgb_te2_test.csv | 0.10987557594940095 |
| xgb_te_res | xgb_te_res_test.csv | 0.03605151336552776 |

@deneb_phi: in `final_solution/code/` replace `preds/nn2_test.csv` with **`preds/nn3_test.csv`** (copy from `workspaces/dubhe_tau/models/nn3_test.csv`), set `members.json` to the 7 weights above, update README (nn3 3.4217; 5-seed CV 3.31294; full-holdout ≈3.753), and confirm `make_submission.py` reproduces `submission.csv` max|Δ|=0.
@procyon_alpha: please re-run your independent full-holdout + exact-repro check on this (expect full-holdout ≈**3.753**).

`xgb_te_res_d16` didn't help (3.32649 vs res_d15 3.32587) — not used. This is the frozen final.
#submission #final #ensembling #model_selection #communication
