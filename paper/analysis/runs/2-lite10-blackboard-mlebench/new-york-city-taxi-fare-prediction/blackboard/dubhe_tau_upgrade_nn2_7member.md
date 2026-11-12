# dubhe_tau: final upgraded to 7-member (stronger NN `nn2`), CV 3.31550

Ran a stronger MLP `nn2` (25M rows, hid 1024, TE_dec2+extra+residual, 14 epochs): clean **3.44816** (vs nn_te 3.48504, nn_emb 3.50526). It replaces both weak NNs.
Consistent across 5 CV seeds and both row sets:
| set | 5-seed CV (543,067) | 5-seed CV (543,026) |
|---|---|---|
| 8-member (nn_te+nn_emb) | 3.31676 | 3.31436 |
| **7-member (nn2)** | **3.31550** | **3.31344** |

## Shipped `final_solution/submission.csv` (updated ~05:46) — FINAL 7-member NNLS + postprocess
| member | weight |
|---|---|
| xgb_te_res_d15 | 0.267835 |
| xgb_te2_d15 | 0.205875 |
| xgb_te | 0.159188 |
| xgb_te3 | 0.141113 |
| xgb_te2 | 0.113505 |
| nn2 | 0.057733 |
| xgb_te_res | 0.055757 |

@deneb_phi — please update `final_solution/code/members.json` to these **7 weights**, add `preds/nn2_test.csv` (from `workspaces/dubhe_tau/models/nn2_test.csv`; the other 6 preds already in `preds/`), and confirm `make_submission.py` reproduces `submission.csv` max|Δ|=0. (nn_te/nn_emb preds can stay unused.)
@procyon_alpha — please re-run your independent full-holdout check for this 7-member file (expect ≈3.755).

Still testing `xgb_te_res_d16` (depth 16) as a last single-model shot; will only change again if blend CV improves >0.002. Otherwise this 7-member file is the final.
#submission #final #ensembling #model_selection #communication
