# deneb_phi: NEGATIVE result — nonlinear stacking is much worse than linear blend

Re [[procyon_alpha_v2_full_running_blend.md]] (procyon offered to try stacking meta-model).

Tested on canonical clean holdout (543,026 rows), members `{xgb_te2,xgb_te3,xgb_te,nn_te,nn_emb}`:
| combiner | 5-fold CV clean RMSE |
|---|---|
| **NNLS (linear)** | **3.3169** |
| LightGBM meta, member preds only | 3.51151 |
| LightGBM meta, member preds + {haversine, log_hav, hour, weekday, year, month, airport_min, is_airport, pax, bearing, member mean/std} | 3.43408 |

⇒ **Do not use a GBDT stacker** — it badly underperforms the simple linear blend (few member signals; tree cannot represent the near-linear optimal combination and overfits). Linear NNLS/Ridge is the right choice. This saves procyon the experiment.

Status: `final_solution/` ships the 5-member NNLS + `postprocess_final.py`; code reproduces submission.csv exactly (max|Δ|=0). Waiting on `xgb_te4` (dubhe) and `lgb_v3te_full` (procyon, canonical) to test additions; otherwise final stands.

#ensembling #model_selection #insight #communication
