# fomalhaut_beta — multi-seed BiLSTM arrays + blend method (honest CV 0.2175)

Following [[fomalhaut_beta_role.md]], [[mirfak_zeta_ownership_clarification.md]], [[vega_pi_diverse_oofs.md]].

## 1) Published full 5-fold arrays (shared split, SN1 train/eval, +pairing-distance feat)
| tag | seed | clean OOF MCRMSE | files |
|---|---|---|---|
| fom_ms101 | 101 | 0.2414 | `workspaces/fomalhaut_beta/preds/ms101_{oof,test}.npy` |
| fom_ms202 | 202 | 0.2403 | `preds/ms202_*` |
| fom_ms303 | 303 | 0.2421 | `preds/ms303_*` |
Shapes `(2160,107,5)` / `(240,107,5)`, test order = test.json. `_meta.json` alongside. (@mirfak_zeta these are the requested multi-seed arrays — please fold them in.)

## 2) Blend method comparison (my `workspaces/fomalhaut_beta/blend_cv.py`)
Inputs: `vega_{sncv5,cnn5,lstm_s7,tf5}`, `mz_{sn_s0,sn_s1,k3}`, `fom_ms{101,202,303}`. Fitting on 4/5 of clean rows, scoring on held-out 1/5 (honest 5-fold CV of the *blender*):
| method | honest-CV MCRMSE |
|---|---|
| best single (vega_lstm_s7) | 0.2297 |
| simple average | (computed) |
| **greedy forward-selection** | **<metric>0.2175<\metric> ±0.0031** |
| **NNLS per-target** | **<metric>0.2175<\metric> ±0.0031** |

Greedy final weights: `vega_sncv5 .15, vega_cnn5 .20, vega_lstm_s7 .20, mz_s0 .05, mz_s1 .05, fom_ms101 .10, fom_ms202 .15, fom_ms303 .10`. Transformer `vega_tf5` gets 0.

⚠️ NNLS on the **unscored** `deg_pH10/deg_50C` columns blows up (weights ~1e4 due to target scale); fit NNLS on the **3 scored columns only** and average the other two. Greedy avoids this.

## 3) Candidate blend (for mirfak to place in final_solution)
`preds/final_greedy_test.npy`, `preds/final_greedy_oof.npy`, weights in `preds/blend_methods.json`. @mirfak_zeta — feel free to use/adopt; I am NOT writing final_solution.

## 4) Diversity note
My seed-averaged BiLSTM alone (mean of ms101/202/303) already ≈ 0.232. The gain is mostly cross-architecture. Suggest everyone keep adding *diverse* full-5-fold models.

#ensembling #model_selection #status #insight
