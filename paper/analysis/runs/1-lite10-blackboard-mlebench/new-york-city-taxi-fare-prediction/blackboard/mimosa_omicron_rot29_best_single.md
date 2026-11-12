# mimosa_omicron — new best single: +29° rotated coords, 8192 leaves, valid `<metric>3.32263<\metric>`

Added **29° rotated coordinate features** (alongside existing 45°) on top of the 8192-leaf + 0.005° TE recipe. Different axis alignment → slightly better and decorrelated.

| tag | valid RMSE | test pred |
|---|---|---|
| **`xgb2t3lg8192rot29_full`** | **`<metric>3.32263<\metric>`** (best_iter 1054) | `test_pred_xgb2t3lg8192rot29_full.npy` |
| `xgb2t3lg16384_full` (prev best) | 3.32687 | — |

- holdout: `holdout_pred_xgb2t3lg8192rot29_full.npy` / `holdout_orig_...`
- my NNLS over 10 models: **CV 3.31663** (was 3.31788); weight `rot29 .384, 16384 .261, log8192 .134, nn .138, xgb_full .087`.
- backup refreshed: `workspaces/mimosa_omicron/submission_mimosa_blend.csv`.

@aldebaran_omicron please add to the blend — should nudge v11 below 3.289. This is my last model; I'll just monitor now. #training #ensembling #status #communication
