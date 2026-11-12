# dubhe_tau: NEGATIVE — 10-fold OOF TE is worse than 5-fold; final unchanged (nn3 7-member)

Tested whether reducing the train/test TE distribution mismatch (5-fold OOF → 10-fold OOF on dec2/extra/residual TE, same alpha/features/params as best single `xgb_te_res_d15`):
- `xgb_te10_res_d15` clean = **3.33053** (vs 5-fold `xgb_te_res_d15` 3.32587) → worse.
- In blend: CV **3.31325** vs 3.31294 (5-fold) → adds nothing (weight .043). Discarded.
⇒ The 5-fold OOF "noise" acts as helpful regularization; more folds (less mismatch) doesn't help. Another negative closed.

Final remains exactly as shipped: nn3 7-member, CV 3.31294, full-holdout ≈3.7547, package reproducible (max|Δ|≈6e-14).
Open negative results now on record: route dec3 TE, route×time TE, 10-fold TE, coordinate-free GBDT, max_bin1024, seed-bag NN/GBDT, duplicate matching for invalid rows, GBDT stacking — none clear the >0.002 bar. Remaining error is tail/invalid noise. #insight #feature_engineering #model_selection #final #status
