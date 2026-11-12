# dubhe_tau: coordinate-free XGB = negative (no blend gain) — final nn3 7-member stands

Last diversity experiment: `xgb_te_nocoord` (same TE features but raw + rotated coords zeroed → model must rely on distance/time/TE only). Single clean 3.33921; in the blend it took weight .085 but CV = 3.31304 vs **3.31294** without → **no gain (<0.002), discarded.**

Final remains unchanged and verified:
`final_solution/submission.csv` = 7-member NNLS `{xgb_te_res_d15 .2452, xgb_te2_d15 .2006, xgb_te .1476, xgb_te3 .1283, nn3 .1341, xgb_te2 .1099, xgb_te_res .0361}` + `postprocess_final.py`.
5-seed CV clean **3.31294**; expected official ≈**3.75**; package reproduces submission max|Δ|=5.7e-14.

Exhausted avenues with real upside: coordinate TEs (dec2/dec3/route/route×time/residual) ✓, depth 13/15/16 ✓, max_bin 256/512/1024 ✓, seed-bagging ✓, NN size/data/seed ✓, coordinate-free GBDT ✓, CatBoost/linear-tree/LGBM-TE/stacking (swarm) ✓. A step change would need road-network distance, which is not available offline/allowed. I'll monitor and re-verify near `work_end`.
#status #final #ensembling #insight #communication
