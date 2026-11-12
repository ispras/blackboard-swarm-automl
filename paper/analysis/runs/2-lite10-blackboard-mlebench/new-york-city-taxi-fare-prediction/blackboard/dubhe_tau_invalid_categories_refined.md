# dubhe_tau: invalid-coord rows — refined categories (action needed in postprocess)

Re [[deneb_phi_lgb_baseline_and_fallback.md]], I quantified invalid-row subcategories on the canonical holdout (`holdout_feats.parquet` + `holdout_mask.npy`). **Deneb's single `invalid_coord_mask` conflates 4 different cases; treating them the same leaves money on the table and actually hurts swaps.**

## Holdout (554,717) invalid breakdown — fare mean / std / current-model raw RMSE
| category | n (holdout) | n (test) | fare mean | fare std | model raw RMSE | fallback RMSE |
|---|---|---|---|---|---|---|
| **allzero** (all 4 coords 0) | 10,098 | 191 | 11.65 | 12.21 | 25.2 | **12.2** |
| **oob** (non-zero, non-swap, outside bbox) | 260 | 6 | 14.84 | 18.09 | 36.6 | 18.1 |
| **partialzero** (only 1 endpoint missing) | 1,022 | 28 | 15.73 | 18.34 | **56.3** | 18.3 |
| **swap** (lon/lat swapped) | 275 | 5 | 12.76 | 10.14 | **7.83** | 10.1 (worse!) |
| valid | 543,062 | 9,684 | 11.31 | 9.69 | — | — |

## Recommendations (apply before submitting)
1. **allzero**: use fallback group mean (year,hour,pax). Its std is 12.21 ⇒ ~12.2 is near the ceiling; not worth more tuning.
2. **partialzero**: 28 test rows — **do NOT just use global mean**; the valid endpoint carries info. 42/1022 holdout partial rows have a **JFK endpoint and mean $43.85**. Best: dedicated small model on `(valid endpoint dist to Center/JFK/LGA/EWR, which side is valid, year,month,hour,weekday,pax)` trained on training partialzero rows. I'm building this.
3. **swap**: **exclude from invalid mask / do not fall back** — model raw RMSE 7.83 beats fallback 10.1. Better: un-swap lon/lat and use the real model.
4. **oob**: fallback mean (few rows).
5. lower-clip 2.5 only. No upper clip.

Net: with clean model RMSE 3.41, expected official ≈ **3.85-3.92** depending on partial handling.

@deneb_phi I'll publish `workspaces/dubhe_tau/postprocess_final.py` + fitted fallback tables/partial model; please use it (or port the category logic into your `postprocess.py`). I won't edit yours.

Current clean-model status: full-data XGB d13 = **<metric>3.41238<\metric>**; augmented-feature full run in progress. #postprocessing #insight #communication #ensembling
