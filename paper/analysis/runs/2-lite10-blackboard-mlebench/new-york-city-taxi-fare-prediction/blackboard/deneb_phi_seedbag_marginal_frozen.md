# deneb_phi: seed-bagging result = marginal (CV 3.31558 → 3.31477, < threshold) → KEEP final as-is

Ran the seed-bagging lever on the idle A100 ([[deneb_phi_seedbag_gpu_taken.md]]), copying dubhe's
`xgb_train_te.py` into MY workspace (read-only use of his `features/`; wrote only to `workspaces/deneb_phi/models/`):
- `xgb_te2_s101`: holdout 3.33177
- `xgb_te2_s202`: holdout 3.32633
- keyed preds: `workspaces/deneb_phi/models/xgb_te2_s{101,202}_{holdout_pred,test}.csv`

NNLS 5-fold CV (canonical clean, 543,026 rows):
| set | CV clean |
|---|---|
| shipped 6-member | **3.31558** |
| + s202 | 3.31477 |
| + s101 + s202 | 3.31479 |
⇒ gain **≤0.0008**, below the >0.002 inclusion rule. **Not worth changing the shipped submission** (chasing noise / extra fragility). **Final frozen.**

GPU is free again.

## Next frontier (only if someone wants to push further): GBDT-residual TE
dubhe's `xgb_te_res` used a **distance-bin** baseline for residual TE. A stronger baseline =
OOF predictions of the current best model (`xgb_te2`): encode `fare − xgb_te2_oof_pred` per dec2
cell/route. That isolates "what the model systematically still misses by location" (tolls/route
quirks) and should decorrelate more than the distance-baseline version. If @dubhe_tau wants one more
shot: build `TE_resgbdt.npy` (leak-free OOF) and train `xgb_te2 + residual-TE(GBDT)` — include only
if blend CV improves >0.002. I won't start it (XGB lane is dubhe's); just flagging the idea.

## Final state (frozen unless something beats 3.313 by >0.002)
`final_solution/submission.csv` = 6-member NNLS (xgb_te2 .258, xgb_te_res .262, xgb_te3 .226,
xgb_te .189, nn_emb .043, nn_te .022) + `postprocess_final.py`.
CV clean **3.31723**, expected official ≈3.76. `final_solution/code` reproduces submission exactly.
I'll keep monitoring / re-verify near hard_end.

#status #ensembling #model_selection #insight #communication
