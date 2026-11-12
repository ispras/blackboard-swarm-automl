# dubhe_tau training pipeline (GPU lane)

Reproduces the GPU/feature-engineering members of the shipped ensemble.
All models trained only on `data/labels.csv`; validation = canonical holdout
`pd.util.hash_array(raw_key_str) % 100 == 0`. No external data, no pretrained
weights (only standard libraries: numpy/pandas/pyarrow/xgboost/torch).

Env: `pyenv shell sis-black` (Python 3.11, xgboost 2.1.1, torch 2.2.0, CUDA A100).

## 1. Feature cache
`build_cache.py` streams `data/labels.csv` once -> `features/{X_full.npy, y_full.npy,
holdout_full.npy, holdout_reference.csv, test_X.npy, test_keys.csv, meta.json}`.
Cleaning: fare in [0.5,500], coords lon[-75.5,-72.5] lat[39.5,42], pax 0..8.
29 base features: raw coords, pax, datetime, haversine, manhattan, bearing,
deltas, distances to Times Sq/JFK/LGA/EWR (pickup & dropoff), airport flags.

## 2. Target-encoding features (leak-free, OOF on train only)
- `build_te.py`  -> `TE_dec2.npy` (dec2 pu-cell / do-cell / route fare mean + log-count, 5-fold OOF)
- `build_te2.py` -> `TE_extra.npy` (dec3 pu/do cells + coarse route dec1)
- `build_te_res.py` -> `TE_res.npy` (residual TE vs a 0.2km-binned distance baseline)
- (`build_te3.py`, `build_te4.py` were negative ablations, kept for the record)
Holdout/test encodings use full-train statistics (no holdout labels).

## 3. Models (GPU)
`xgb_train_te.py` builds an augmented float32 feature matrix per batch
(rotated coords, cyclic time, log distances, airport flags, bearing sin/cos)
plus TE columns, then trains XGBoost `hist` on cuda via a streaming
QuantileDMatrix (memory-light).
Members shipped (clean-holdout RMSE on the 543,067 valid rows):
- `xgb_te_res_d15`: d15 lr.04 te2+residual-TE  -> 3.32587
- `xgb_te2_d15`   : d15 lr.04 te2             -> 3.32611
- `xgb_te2`       : d13 lr.045 te2            -> 3.32837
- `xgb_te_res`    : d13 lr.045 te2+residual   -> 3.33177
- `xgb_te3`       : d13 lr.045 te2+route dec3 -> 3.33310
- `xgb_te`        : d13 lr.045 te2 minus extra-> 3.34492
`nn_train.py` (with `--te 1`) -> MLP with spatial-cell + calendar embeddings
(`nn_emb` 3.50526, `nn_te` 3.48504).

## 4. Blend + postprocess
`final_blend.py` fits NNLS weights on full clean holdout (5-fold CV used to choose
NNLS vs Ridge) then applies `postprocess_final.py`:
- valid rows: model prediction
- all-zero coords: group-mean fallback (year,hour,pax) fitted on training all-zero rows
- partial-zero coords: dedicated small XGB on valid-endpoint features
- out-of-bbox: global mean of 'other' training rows
- swapped lon/lat: keep model prediction
- lower clip 2.5, no upper clip.
Shipped blend: 5-fold CV clean = 3.31637, full-holdout after postprocess ~3.76.

## Update (final ship): stronger NN
`nn_train2.py` (40M rows, hid=1024, TE_dec2+TE_extra+TE_res, lr1.5e-3, 10 epochs) -> `nn3` clean 3.42169
(replaces the weaker NNs in the final blend). Final 7-member NNLS:
`xgb_te_res_d15 .2452, xgb_te2_d15 .2006, xgb_te .1476, xgb_te3 .1283, nn3 .1341,
xgb_te2 .1099, xgb_te_res .0361`; 5-seed CV clean = 3.31294 (full-holdout ~3.753).
