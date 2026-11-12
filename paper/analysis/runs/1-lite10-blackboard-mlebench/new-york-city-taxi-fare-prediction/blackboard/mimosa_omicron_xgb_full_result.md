# mimosa_omicron — full-data XGB + TE result + test preds ready

Agrees with [[aldebaran_omicron_corrupt_insight]]: I too train/evaluate **valid-coord only**, and use a **separate non-spatial fallback model** for the 230 corrupt test rows. My fallback holdout RMSE = 9.497 (≈ fare std; unavoidable ~2.07 MSE floor for 2.3% of test).

## Model A: XGBoost GPU, full 54.26M valid rows (recommended comparable metric)
- Features (54): raw coords, deltas/abs, haversine, manhattan, bearing, airport dists (JFK/LGA/EWR pickup+dropoff+sum+flags), midtown/lower dists, datetime, **45°-rotated coords**, pickup/dropoff **cell counts + route count**, and **OOF 5-fold smoothed target encodings** for pickup cell, dropoff cell, OD-route (0.01° grid; only 194k unique routes).
- depth 10, eta .03, 4048 best iters, 54M rows, ~12 min on A100.
- **valid-coord holdout RMSE = `<metric>3.35785<\metric>`** (baseline no-TE full-data XGB was 3.39224).
- Expected actual test ≈ sqrt(0.977·3.358² + 0.023·9.497²) ≈ **3.62** (corrupt floor included).

## Artifacts for blending / reuse (in `workspaces/mimosa_omicron/`)
- OOF-aligned holdout: `holdout_pred_xgb2_full.npy`, `holdout_orig_xgb2_full.npy` (= orig_idx values, sorted ascending), `holdout_true_xgb2_full.npy`.
- Holdout preds from other tags also follow same naming.
- Test predictions (fallback applied, clipped [2.5,500]): `submission_xgb2_full.csv` — currently my best; safe as current swarm fallback.
- Reusable code: `prep.py` (raw->features), `add_features.py` (v2 + rotated + counts), `te.py` (OOF TE + test maps), `predict.py`.

**To blend**: load my `holdout_orig` + preds and align to your own `orig_idx` arrays. I'll write a blender script that takes any `(orig_idx, pred)` pairs.

Next: PyTorch MLP/embedding net on A100 (diversity), plus XGB variants (different seeds/params/feature subsets) and a LightGBM-on-my-features for ensemble. #training #feature_engineering #ensembling #status #insight
