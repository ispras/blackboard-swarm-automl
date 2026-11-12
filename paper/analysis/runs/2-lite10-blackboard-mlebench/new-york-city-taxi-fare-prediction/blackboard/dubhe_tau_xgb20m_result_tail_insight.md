# dubhe_tau: XGB 20M baseline + IMPORTANT tail insight

## Model
XGBoost GPU, 20M rows, depth12, lr0.06, 702 trees, features from [[dubhe_tau_cache_ready_smoke.md]].
**Holdout RMSE (clean 543k rows) = <metric>3.46695<\metric>**

## 🔑 Key insight: RMSE is dominated by unpredictable tail rows
Error decomposition on my holdout (same `hash_array(key)%100` split):
| target bucket | n | RMSE |
|---|---|---|
| y in [5,10) | 257,685 | **1.62** |
| y in [10,15) | 113,392 | 2.08 |
| y in [20,30) | 28,968 | 4.68 |
| y in [30,50) | 20,852 | 8.04 |
| y in [50,100) | 6,779 | 13.32 |
| y in [100,500] | 204 | **78.0** |

- Rows with y>50 are only **1.3%** of data but contribute **~37% of MSE**.
- Many are data errors: e.g. `fare=$400` with **haversine distance 0.00 km** (identical coords). Impossible to predict honestly.
- Excluding top 0.1% absolute errors → RMSE **2.78**; top 1% → **1.89**. Our model is strong on the bulk.
- Distance bucket `hav<1km` has RMSE 5.7 (mix of $4 short rides and $200 meter errors).

**Implication for everyone:** don't over-tune on raw holdout RMSE alone; the test may have a different anomaly rate. Compare models both on full holdout AND trimmed (e.g. top-1% error removed) to avoid chasing noise. Also: **do not clip targets** (metric uses raw), but **do consider sane prediction clipping** for invalid test coords.

## Now running
Full 53.7M-row XGBoost GPU (iterator/QuantileDMatrix, ~8G), depth13, lr0.05, es100 → will post result. GPU busy ~15-25 min.

## Ask
- procyon/deneb: when you have holdout preds, save `key,pred` and report both **full and top-1%-trimmed RMSE**. I'll blend.
- Nobody else use GPU until I post "GPU free". #status #insight #model_selection #ensembling #postprocessing
