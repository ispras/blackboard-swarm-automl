# NYC Taxi Fare Prediction — swarm final solution

Metric: RMSE on `fare_amount` (lower is better). Data used: **only** `data/labels.csv`
(train) + `data/test.csv` (features). No external data, no pretrained checkpoints,
no test labels. Validation/early-stopping used a holdout cut from train only.

## Result (local, valid-coordinate holdout)
- Best single model: XGBoost lossguide 8192 leaves + fine TE + 29° rotated coords (mimosa `xgb2t3lg8192rot29`) — **3.3226**
- Final NNLS blend of 6 full-data models (5-fold CV on the shared holdout, `orig_idx%100==0`):
  **RMSE 3.2889** (valid-coord rows only, n=542,554; NNLS)
- Corrupt-coordinate rows (225/9914 after coord recovery) handled by corrupt-trained
  LightGBM models: **5-fold CV RMSE 12.67** (canopus model, used; 12.81 for the aldebaran
  variant as backup) vs 13.40 for the best constant.
- Expected test RMSE (incl. corrupt floor):
  `sqrt(0.9773·3.2889² + 0.0227·12.65²) ≈ 3.77`

## Approach
1. **Feature engineering** (full 55.4M rows, chunked): raw coords, deltas/abs deltas,
   haversine + Manhattan distance, bearing, distances to JFK/LGA/EWR/Midtown/Lower-Manhattan
   (pickup+dropoff+sums+near-airport flags), 45°- and 29°-rotated coordinates,
   datetime (year/month/day/hour/minute/weekday/doy/week + cyclic hour + weekend),
   passenger count, pickup/dropoff/OD-route **frequency counts** and **OOF smoothed
   target encodings** (0.01° grid; 194k unique routes).
2. **Cleaning**: `2.5 ≤ fare ≤ 500`, `0 ≤ passenger ≤ 8`, train only on valid-coordinate rows
   (`lon∈[-75.5,-72.5], lat∈[39.5,42]`); separate treatment for corrupt rows (below).
3. **Models** (all trained on ~53.7M valid rows, early-stopped on the same 542k holdout):
   | # | model | features | local holdout RMSE |
   |---|-------|----------|--------------------|
   | -2 | LightGBM full data, multi-res TE (canopus `j54m`) | canopus feats | 3.3258 |
   | -1 | XGBoost lossguide **16384 leaves** + 0.005° TE, full data (mimosa `xgb2t3lg16384`) | 54 + fine TE | 3.3269 |
   | -1b | XGBoost lossguide **8192 leaves** + 0.005° TE, full data (mimosa `xgb2t3lg8192`) | 54 + fine TE | 3.3270 |
   | -3 | XGBoost lossguide 8192 leaves + fine TE + 29° rotated coords, full data (mimosa `xgb2t3lg8192rot29`) | 60 | 3.3226 |
   | -2 | LightGBM full data, multi-res TE (canopus `j54m`) | canopus feats | 3.3258 |
   | 0 | LightGBM **full 53.7M rows**, multi-res TE, 511 leaves (canopus `g54m`) | canopus feats | 3.3332 |
   | 1 | XGBoost lossguide 1024 leaves, finer 0.005° TE (mimosa `xgb2t3lg_full`) | 54 + finer TE | 3.3457 |
   | 1c | XGBoost lossguide 8192 leaves, **log1p target**, fine TE (mimosa `xgb2t3lg8192log`) | 54 + fine TE | 3.4141 |
   | 1d | XGBoost lossguide 8192 leaves, **log1p target** + fine TE + 29° rot (mimosa `xgb2t3lg8192rot29log`) | 60 | 3.4085 |
   | 1b | XGBoost lossguide 1024 leaves (mimosa `xgb2lg_full`) | 54 + TE | 3.3519 |
   | 2 | XGBoost depthwise depth 11 (aldebaran `xgbS`) | 54 + TE | 3.3523 |
   | 3 | XGBoost depthwise depth 10, no TE (mimosa) | 34 | 3.3922 |
   | 4 | XGBoost depthwise depth 10 + TE (mimosa) | 54 + TE | 3.3579 |
   | 5 | CatBoost depth 10 (aldebaran `catS`) | 54 + TE | 3.3838 |
   | 6 | LightGBM **full 54M rows**, multi-res TE (canopus `lgb_f54m`) | canopus feats | 3.4027 |
   | 6a | LightGBM 19M rows, fixed TE (canopus `lgb_e19fix`) | canopus feats | 3.4043 |
   | 6b | LightGBM 19M rows, previous TE (canopus `lgb_d19m`) | canopus feats | 3.4299 |
   | 7 | Embedding MLP, full data (mimosa `nn_full`) | 54 + TE + embeddings | 3.4395 |
   | 8 | XGBoost, log(fare) target, full data (mimosa `xgb2log_full`) | 54 + TE | 3.4346 |
   Weights from **NNLS fit on the shared holdout** (5-fold CV to avoid overfitting):
   `canopus_j54m 0.307, xgb2t3lg16384 0.220, canopus_g54m 0.144, xgb2t3lg8192rot29log 0.140, xgb2t3lg8192rot29 0.129, nn_full 0.064` (rest 0). Weights are 5-fold bagged NNLS (non-negative); OLS reached ~3.287 CV only with large negative weights and was rejected for robustness.
   A feature-aware GBDT stacker was tried but its CV (3.417) was worse than the linear
   blend (3.335), so it was discarded.
4. **Corrupt coordinates** (230 test rows): 5 are recovered by trying lon/lat swaps and
   sign flips; the remaining 225 get predictions from a LightGBM trained **only on the
   1.15M corrupt train rows** using date/time, passenger count, and valid-endpoint
   landmark distances/coords (NaN where invalid). This beat the constant-mean fallback
   (12.81 vs 13.40 CV RMSE) and is applied to those 225 rows. Final predictions clipped
   to `[2.5, 500]`.

## Files
- `submission.csv` — final predictions (`key,fare_amount`), 9,914 rows, test.csv order.
- `code/blend.py` — aligns holdout/test predictions by `orig_idx` and fits/applies NNLS blend.
- `code/train_full_shared.py` — full-data depthwise XGBoost on the shared feature table.
- `code/train_full_cat.py` — full-data CatBoost on the shared feature table.
- `code/train_corrupt_final.py` — corrupt-coordinate fallback model + test predictions.
- `code/corrupt_cv.py` — 5-fold CV of the corrupt fallback vs constant.
- `code/predict_shared.py` — build test features and predict with a shared-feature model.
- `code/build_features.py`, `code/build_extra.py`, `code/build_shared.py` — feature pipelines.
- `code/stacker.py`, `code/train_xgb_full.py`, `code/corrupt_const.py` — experiments/utilities.

Model artifacts and intermediate prediction arrays live in the swarm workspaces
(`workspaces/<agent>/...`); they are large and not duplicated here. Mimosa's own
pipeline is additionally copied under `code/mimosa/` (prep, features, TE, XGB/NN training,
corrupt fallback, prediction) with its own README.

## Swarm collaboration / provenance
This is a swarm solution; the final ensemble combines models and features produced by
three agents. Board references:
- `[[mimosa_omicron_xgb_full_result.md]]`, `[[mimosa_omicron_testpreds_blend.md]]` — XGB/NN models, feature table (`feat_v2`), OOF TE maps.
- `[[canopus_pi_result_lgb19m.md]]`, `[[canopus_pi_artifacts_blend.md]]` — LightGBM `lgb_d19m` + preds.
- `[[mimosa_omicron_corrupt_fallback_v2.md]]`, `[[mimosa_omicron_coord_recovery.md]]` — corrupt-row analysis, recovery, corrupt feature set.
- `[[aldebaran_omicron_corrupt_and_blend.md]]`, `[[aldebaran_omicron_blend_request.md]]` — corrupt model, blending protocol.
- `[[aldebaran_omicron_gpu_full.md]]`, `[[aldebaran_omicron_parquet_ready.md]]` — full-data training, shared FE parquet.

## External sources
- Python libraries: `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `xgboost` (2.1.1),
  `lightgbm` (4.5.0), `catboost` (1.2.5). No pretrained model checkpoints or external
  datasets were used.
