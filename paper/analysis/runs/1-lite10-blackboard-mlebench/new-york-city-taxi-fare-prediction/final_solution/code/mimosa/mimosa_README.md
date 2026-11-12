# NYC Taxi Fare Prediction — mimosa_omicron solution

Predict `fare_amount` (RMSE). Training only on `data/labels.csv`; no external data or pretrained competition-specific models were used. All ML code uses the preinstalled `sis-black` environment (xgboost 2.1.1, lightgbm 4.5.0, torch 2.2.0, pandas/numpy/pyarrow).

## Metric / validation
- RMSE, lower is better.
- Shared swarm holdout: `orig_idx % 100 == 0`, where `orig_idx` = 0-based row order in `data/labels.csv`.
- Main model trained/evaluated on **valid-coordinate rows only** (both endpoints inside `lon∈[-75.5,-72.5], lat∈[39.5,42]`, `0<dist≤300 km`, `2.5≤fare≤500`, `0≤passenger≤8`). Holdout RMSE of the final single model: **3.3519**.

## Pipeline (files)
1. `prep.py` — chunked CSV → parquet; cleaning; base features (raw coords, deltas, haversine, manhattan-style distance, bearing, airport/landmark distances, datetime).
2. `add_features.py` — 45°-rotated coordinates, airport distance sums/flags, 0.01° pickup/dropoff cell ids and route id + frequency counts.
3. `te.py` — **out-of-fold (5-fold, smoothed) target encodings** of pickup cell, dropoff cell, OD-route; saves full-train encodings for test-time lookup.
4. `te3.py` (optional) — finer 0.005° cell/route TE.
5. `train_xgb2.py` — XGBoost GPU (`device=cuda`, `tree_method=hist`) on ~54.2M valid rows. Final variant uses `grow_policy=lossguide`, `max_leaves=1024`, `eta=0.025`, `max_depth=0`, `min_child_weight=200`, `subsample/colsample=0.8`.
6. `build_corrupt.py` + `train_corrupt_fb.py` — separate fallback model for test rows with corrupt/out-of-bbox coordinates (non-spatial + valid-endpoint landmarks + missingness flags).
7. `recover_coords.py` — recovers lat/lon-transposed endpoints (5 of the 230 corrupt test rows).
8. `test_features.py`, `predict.py` — test feature construction, TE lookup, coordinate recovery, main prediction on clean rows, corrupt fallback on corrupt rows, clipping to `[2.5, 500]`, submission writer.
9. `blend.py` — optional NNLS blend of multiple models aligned by `orig_idx` on the shared holdout.

## Reproduce
```bash
pyenv shell sis-black
python prep.py            # -> feat/part_*.parquet
python add_features.py    # -> feat_v2/part_*.parquet
python te.py              # -> te_maps.npz
python train_xgb2.py --frac 1.0 --rounds 8000 --eta 0.025 --depth 0 \
  --grow lossguide --max_leaves 1024 --min_child_weight 200 \
  --subsample 0.8 --colsample 0.8 --early 200 --tag xgb2lg_full --threads 16
python build_corrupt.py && python train_corrupt_fb.py
python predict.py --model model_xgb2lg_full.json --fallback model_corrupt_fb.json \
  --out submission.csv
```

## Notes / insights
- Extreme fares dominate RMSE: ~0.03% of valid rows with fare>100 add ~1.5 MSE.
- The ~2.3% test rows with corrupt coordinates are a large irreducible MSE floor; a corrupt-specific model (RMSE≈12.8 on corrupt population) is far better than a constant (~13.4-15).
- Models sharing target-encoding features are highly correlated (residual corr >0.95), so blending gains only ~0.01 RMSE within a single agent.

## Additional models used in the final swarm blend (mimosa)
- `xgb2t3lg8192rot29_full` (best single, valid RMSE 3.32263): 8192-leaf lossguide XGB, eta .02, mcw 150, max_bin 128, on FEATS_V2 + 0.005° cell/route TE + **29°-rotated coordinates** (`--rot29`).
- `xgb2t3lg16384_full` (3.32687), `xgb2t3lg8192_full` (3.32700), `xgb2t3lg8192log_full` (3.41414, log1p target), `nn_full` (3.44317), plus no-TE `xgb_full` (3.39224).
- Blend: NNLS with non-negative weights, 5-fold CV on the shared holdout; test predictions from all models combined for valid rows, corrupt rows replaced by the corrupt-trained model, clipped to [2.5,500].
- Negative results: training-label denoising, max_bin=256, extreme min_child_weight, and >16384 leaves all did not help.
