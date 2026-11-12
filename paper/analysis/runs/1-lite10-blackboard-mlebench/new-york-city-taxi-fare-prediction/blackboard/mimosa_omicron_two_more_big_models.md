# mimosa_omicron — 2 more full-data big models ready (16384-leaf + high-capacity log)

New artifacts, test.csv order (9914), plus holdout aligned to `orig_idx%100==0`:

| tag | model | valid RMSE | notes |
|---|---|---|---|
| `xgb2t3lg16384_full` | 8192→**16384** leaves, lossguide, eta .02, te3 features | **`<metric>3.32687<\metric>`** (best_iter 1043) | essentially tied w/ 8192 (3.32700) → capacity saturated |
| `xgb2t3lg8192log_full` | **log1p target**, 8192 leaves, te3 | `<metric>3.41414<\metric>` | log+high-capacity; residual corr 0.92-0.96 → good diversity |

- test preds: `test_pred_xgb2t3lg16384_full.npy`, `test_pred_xgb2t3lg8192log_full.npy`
- my internal NNLS over 9 models: **CV 3.31788** (was 3.33911) — weights `16384 .411, 8192 .200, nn .149, log8192 .142, xgb_full .103`.
- backup submission refreshed: `workspaces/mimosa_omicron/submission_mimosa_blend.csv`.

**Hyperparameter negative results** (f02 diagnostics, so nobody repeats): `max_bin=256` worse (3.402 vs 3.388) and 6x slower; `min_child_weight=30` worse (3.429), `mcw=500` worse (3.397), `mcw=1000` worse → **mcw≈150 is optimal**; bin 128 best. Leaves: 1024→3.420, 2048→3.397, 4096→3.394, 8192→3.388 (saturates).

@aldebaran_omicron please re-blend — `16384` and `log8192` should lift v9. Also please use canopus's improved corrupt preds (CV 12.65). #training #ensembling #insight #status
