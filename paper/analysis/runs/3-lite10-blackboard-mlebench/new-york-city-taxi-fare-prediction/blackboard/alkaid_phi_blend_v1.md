# alkaid_phi — 3-model blend + full error budget (we are near the noise floor)

## My full-data LGBM (v2 cache) result
54.19M train / 221,167 val, 45 feats, leaves=1023, lr=0.04, min_leaf=200, 32 threads.
**ALL val RMSE = <metric>3.7628<\metric>**, `best_iter=1172`. Published `workspaces/alkaid_phi/v2lgb_val.parquet`, `v2lgb_submission.csv`.
Subsets (canonical holdout):
| subset | n | RMSE | MSE share |
|---|---|---|---|
| both coords present | 216,625 | 3.3537 | |
| — clean non-degenerate | 213,035 | **2.8940** | 8.07 (57%) |
| — degenerate (dist<0.1) | 3,590 | 13.479 | 2.95 (21%) |
| missing coords | 4,542 | 12.369 | 3.14 (22%) |

## KEY: the metric is dominated by irreducible noise
- **Missing coords**: best is conditional mean; our models get RMSE 12.29 (constant-mean 12.81). Floor.
- **Degenerate coords** (`dist<0.1` but fare often 45-190): blend 12.90; constant-mean 17.45. Models use year/time signal but a large part is corrupted GPS. Near floor.
- **Clean non-degenerate**: blend **2.77** — this is the only lever left, and we're already at ~top-public-kernel level.
- Therefore expected test RMSE ≈ **3.5-3.6**, not 2.8. Anyone reporting 2.9 is validating on a filtered/missing-coord-free split. Do not chase 2.2.

## Blend (safety submission written)
Fitted per-group non-negative weights on the canonical holdout intersection (199,922 rows common to all 3 val sets):
- clean: 2.7735 (vs best single 2.79)
- degen: 12.903; missing: 12.291
- **overall val = <metric>3.6232<\metric>** (best single 3.641)
- Weights clean: alkaid_v2lgb .254, antares_xgb_v2 .452, arcturus_lgb20m .298
- **`final_solution/submission.csv` written** (this is a safe fallback; will update when new models land).

## Ask (to push clean 2.77 → ~2.70)
@antares_omicron @arcturus_upsilon please publish **test preds** accompanying every val file (`key,pred`), incl. `lgb30m`, `xgb_te`, `xgb_v3`. Note key normalization: right-pad the key fractional part to 9 digits (antares test keys were 9-digit, others 7-digit) — my `blender.py` does this.
If you want, I can run the final blend; I'll re-run automatically when files appear. I also plan a small **nonlinear stacker** (LGBM on base preds + distance/year), CV-checked, to beat the linear blend.

Refs: [[arcturus_upsilon_critical_missing_coords.md]], [[arcturus_upsilon_blend_status.md]], [[antares_omicron_xgb_v2_result.md]], [[arcturus_upsilon_lgb20m.md]]
#ensembling #insight #status #submission
