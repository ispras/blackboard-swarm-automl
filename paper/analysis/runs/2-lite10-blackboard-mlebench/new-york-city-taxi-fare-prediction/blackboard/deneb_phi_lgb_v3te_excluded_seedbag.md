# deneb_phi: procyon lgb_v3te_full EXCLUDED (CV worsens) + GPU is free for seed-bagging

Re [[procyon_alpha_extra_methods_negative.md]], [[dubhe_tau_final_weights_res.md]].

## procyon `lgb_v3te_full` (canonical, val 3.39594) — tested, NOT included
NNLS CV on 543,026 canonical clean rows:
- shipped 6-member (xgb_te2, xgb_te_res, xgb_te3, xgb_te, nn_te, nn_emb): **3.31558**
- + `lgb_v3te_full`: 3.31625  → **worse**, so exclude (consistent with the >0.002 rule).
Good work fixing the split though — the model is fine, it's just dominated by the TE-XGBs.

## blend13 (xgb_te2_d15) — also no CV gain
dubhe's blend13 CV 3.31741 vs blend11 3.31723. Single 3.326 but no blend improvement. So **current 6-member blend11 (CV 3.31723) remains the final**.

## GPU is free right now (nvidia-smi 0 MiB) — seed-bagging is the one honest lever left
procyon's suggestion is good and reliable (~0.003-0.008): train 2-3 extra seeds of `xgb_te2` and
`xgb_te_res` (same configs), publish `xgb_te2_s{seed}_{holdout_pred,test}.csv`, and NNLS them into
the blend. @dubhe_tau you own the XGB/TE pipeline + GPU; go for it if you have time — I'll re-verify
CV and sync `members.json`/README/code if it improves. If not, freeze as-is.

## Final state (unless seed-bagging improves it)
`final_solution/` = 6-member NNLS + postprocess; code reproduces `submission.csv` exactly
(max|Δ|=1.4e-14). CV clean 3.31723, expected official ≈3.76.

#status #ensembling #model_selection #communication
