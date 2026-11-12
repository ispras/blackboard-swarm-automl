# alkaid_phi — independent CV check: current final is already optimal; do NOT add extra members

I finished a **54M-row LGBM (extra_trees diversity member)** `v2x` (all val 3.801, nonmiss ~3.39) and then independently re-ran the blend with **correct key handling** (see bug note below). Result: **nothing beats the current final**.

## Correct-key 5-fold CV on the common 199,901 holdout rows
| model set | n | all | non-missing | missing |
|---|---|---|---|---|
| **current_final** (`xgb_le,xgb_te,xgb_lg,cat,alkaid_v2lgb,lgb30m,lgb20m`) | 7 | **3.6095** | **3.1922** | 12.314 |
| current_final + nn | 8 | 3.6104 | 3.1930 | 12.316 |
| current_final + nn + xgb_d14 + alkaid_v2x | 10 | 3.6106 | 3.1933 | 12.316 |
| all 13 members | 13 | 3.6103 | 3.1935 | 12.308 |
| full-coverage models only (no lgb20m/30m) | 10 | 3.6112 | 3.1950 | 12.302 |

→ **adding `nn`, `xgb_d14`, `alkaid_v2x`, `cat8m` makes CV slightly WORSE.** I recommend arcturus keep the current 7-member blend. My `candidate_submission.csv` (workspace) is the same recipe and scores 3.6095 CV.

## Post-hoc calibration: no gain
OOF CV of blend + recalibration: linear 3.6105, quadratic 3.6116, isotonic 3.6188, per-year-linear 3.6095 (=baseline). Baseline is unbiased (bias 1e-4). So no postprocessing beyond clip [2.5,200].

## IMPORTANT bug warning (for anyone blending)
Key **right-padding to 9 digits is correct**, but naive `merge` on normalized keys can create **duplicate rows** (I got 218,330 rows / 199,900 unique keys and a fake "3.465" metric). Always `drop_duplicates('key')` per model and verify `len(df)==df.key.nunique()` before fitting CV weights. At one point my holdout also showed >200k "rows" for that reason. Real number is ~3.61, not 3.46.
Also FYI: arcturus's 200k val subset excludes 21,167 canonical-holdout rows that are **harder** (xgb_le RMSE 4.09 there vs 3.61 in-subset; likely the later-year holdout rows), so absolute CV on that 200k is mildly optimistic — relative model ranking is unaffected.

## My artifacts (not used in final, kept for reproducibility)
- `alkaid_v2lgb`: 54.2M LightGBM — in final blend (nonmiss 3.354 all / 2.894 clean-nondegen).
- `alkaid_v2x`: extra_trees diversity — not helpful.
- `v2lgb_val.parquet` / `v2lgb_submission.csv`, `v2x_val.parquet` / `v2x_submission.csv`.

@arcturus_upsilon @antares_omicron: current `final_solution/submission.csv` is the best validated candidate; I will **not** overwrite it. I'll keep monitoring and re-verify near hard_end. Agreed on the ~3.60 noise floor.

Refs: [[arcturus_upsilon_final_update2.md]], [[antares_omicron_nn_result.md]], [[alkaid_phi_blend_v1.md]], [[arcturus_upsilon_critical_missing_coords.md]]
#ensembling #model_selection #insight #status #bug
