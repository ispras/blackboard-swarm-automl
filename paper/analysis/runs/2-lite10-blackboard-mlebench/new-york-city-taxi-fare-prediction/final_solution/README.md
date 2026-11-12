# NYC Taxi Fare Prediction — swarm solution

Predict `fare_amount` for NYC taxi trips. **Metric: RMSE (lower is better).**
Only `data/labels.csv` (55,413,942 rows) was used for training/validation; `data/test.csv`
(9,914 rows) is used only for inference. No external datasets, pretrained models,
or test-label feedback were used.

Expected score on the official test: **RMSE ≈ 3.76** (estimated on a held-out slice of the
training data after the full pipeline, see below).

---

## 1. Validation protocol
A single deterministic holdout of ~1% was carved from the training data so all swarm members
report comparable numbers:

```
holdout = pandas.util.hash_array(key_string) % 100 == 0   # 554,717 rows
```

`key` is read as the raw CSV string. "Clean" metrics below are computed on the rows with valid
coordinates and `fare_amount ∈ [0.5, 500]` (≈543k rows). Models are trained on the remaining
rows only; all validation/early-stopping uses this holdout.

## 2. Data pipeline / feature engineering
One streaming pass over the 5.6 GB CSV (pyarrow, ~6 min) produces a compact float32 parquet
(`features_train.parquet`) with no row deletion, plus the test features:

- raw pickup/dropoff lon/lat, `passenger_count`
- haversine distance (km), Manhattan distance, bearing, Δlat/Δlon
- `log1p(haversine)`, haversine/Manhattan ratio, same-point flag, sin/cos(bearing)
- datetime: year, month, day, hour, minute, weekday, day-of-year, is-weekend
- airport distances/flags (JFK/LGA/EWR min over endpoints, per-endpoint min, flags)
- 30°-rotated coordinates for pickup/dropoff (aligns trees with the Manhattan street grid)

Training rows are filtered to `fare∈[0.5,500]`, `passenger_count∈[0,8]`, coords inside
lon[-75,-72] × lat[39,42]. Invalid-coordinate rows are excluded from training (they poison the
model) but are always predicted (see §4).

**Target encoding (key gain).** Out-of-fold (5-fold, leak-free) means + log-counts of
rounded pickup cell / dropoff cell / route (decimals 2 and 3) are added as features. Computed
from non-holdout training rows only; holdout/test use full-train encodings. This lifted the best
XGBoost from 3.383 → 3.329 clean. (Route-mean used *alone* is weak — see
[[deneb_phi_catboost_and_route_negative.md]]; the value is only as features combined with distance.)

## 3. Models (all trained only on the training split)
| member | description | clean holdout RMSE |
|---|---|---|
| **xgb_te2** | XGBoost, full 53.7M rows, aug features + dec2/dec3 cell+route TE, max_bin512, d13 | **3.3284** |
| **xgb_te_res_d15** | XGBoost d15, TE2 + **residual-TE** (target minus 0.2km-distance baseline) | **3.3259** |
| **xgb_te2_d15** | XGBoost d15, TE2 features | **3.3261** |
| **xgb_te3** | XGBoost, full data, TE variant 3 (finer/coarse route TE) | **3.3331** |
| **xgb_te_res** | XGBoost, full data, **residual TE** (target minus distance baseline, per cell/route) | **3.3318** |
| **xgb_te** | XGBoost, full data, aug + dec2 TE | 3.3449 |
| xgb_b512 | XGBoost, full data, aug features, max_bin512 | 3.3835 |
| xgb_aug | XGBoost, full data, 47 aug features | 3.3905 |
| xgb_full | XGBoost, full data, 29 features | 3.4124 |
| **nn_te** | MLP + cell/time embeddings + TE features, 20M rows | 3.4850 |
| **nn3** | MLP (hid 1024) + cell/time embeddings + TE feats, 40M rows, 10 epochs | **3.4217** |
| nn2 | MLP + cell/time embeddings + TE feats, 25M rows | 3.4482 |
| nn_emb | MLP + cell/time embeddings, 20M rows | 3.5053 |
| lgb_big | LightGBM, 26.8M rows, 32 features, 512 leaves | 3.4449 |
| lgb_te2 | LightGBM, 18.8M rows, TF features | 3.4439 |
| cat_gpu | CatBoost GPU, 26.8M rows, depth 10 | 3.5116 |

Owners and details are on the message board; key references:
[[dubhe_tau_te_breakthrough_interim_sub.md]], [[dubhe_tau_b512_best_trim_trap.md]],
[[deneb_phi_lgb_big_result.md]], [[deneb_phi_catboost_and_route_negative.md]],
[[procyon_alpha_v2_full_running_blend.md]].

## 4. Blending
Non-negative least squares (NNLS) on the clean holdout, 7 members (exact weights in `code/members.json`):
`xgb_te_res_d15 .2452, xgb_te2_d15 .2006, xgb_te .1476, nn3 .1341, xgb_te3 .1283, xgb_te2 .1099, xgb_te_res .0361`.
- 5-fold CV clean RMSE = **3.31294** (5-seed: 3.31189 ± ~0.0006)
- in-sample clean RMSE = 3.31013
- full-holdout after postprocessing ≈ **3.753** (expected official ≈ 3.75)

Other members (xgb_b512/aug/full, lgb_*, cat_gpu) received ~0 weight and are not used.
(Ridge gives the same result to within 0.001; NNLS is used for robustness.)

## 5. Invalid-coordinate postprocessing (CRITICAL, ~1.5 RMSE)
~2% of rows have corrupted coordinates. A GBDT extrapolates badly there (predicts ~$33 vs true
~$12). Categories (see [[dubhe_tau_invalid_categories_refined.md]]) and fixes in
`postprocess_final.py`:
- **allzero** (all 4 coords 0) → group mean by (year, hour, passenger_count)
- **oob** (non-zero but outside NYC bbox) → global training mean
- **partialzero** (one endpoint missing) → dedicated small XGB model on the valid endpoint
- **swap** (lon/lat swapped) → keep the raw model prediction (do **not** fall back)
- lower-clip predictions at 2.5; **no** upper clip.

Effect on full holdout (all 554,717 rows): raw blend ~5.6 → **3.7624** after postprocessing.
Test categories: valid 9684 / allzero 191 / other 34 / swap 5.

## 6. Reproduction
Environment: `pyenv shell sis-black` (Python 3.11, lightgbm 4.5, xgboost 2.1, catboost 1.2.5,
torch 2.2). All heavy jobs run under `systemd-run --scope` with an 8 GB memory cap.

```bash
# 1) feature store (once, ~6 min, streaming)
python code/build_features.py          # -> features_train.parquet
python code/build_test_features.py     # -> features_test.parquet

# 2) train members (examples; see scripts for full args)
python code/train_lgbm.py --frac 0.5 --leaves 512 --lr 0.03 --tag lgb
python code/train_catboost.py --frac 0.5 --depth 10 --device GPU --tag cat
python code/train_lgb_te.py --frac 0.35 --leaves 384 --lr 0.04 --tag lgb_te2

# 3) blend + postprocess (self-contained; uses precomputed member test preds in code/preds/)
python code/make_submission.py         # -> submission.csv
```

`code/preds/` contains the seven member test predictions and `code/members.json` the exact blend
weights; running `make_submission.py` reproduces the shipped `submission.csv` **exactly** (verified
max |Δ| = 0) without retraining.
Full GPU/TE/NN training scripts are in `code/dubhe_tau_scripts/` (see its `README.md`);
my CPU pipeline scripts are in `code/` (`train_lgbm.py`, `train_catboost.py`, etc.).

## 7. External sources
**None.** No pretrained checkpoints or external data were used. Training used only
`data/labels.csv`; inference used only `data/test.csv`. Models: LightGBM, XGBoost, CatBoost,
PyTorch (all open-source libraries).
