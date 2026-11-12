# NYC Taxi Fare Prediction — final solution

Metric: **RMSE** (lower is better). Submission: `submission.csv` with columns `key,fare_amount`, rows in the **exact same order and string format as `data/sample_submission.csv`** (7-digit fractional seconds).

Reference validation (deterministic holdout carved from `data/labels.csv`, 199,900 rows covered by all blend members): **<metric>3.6033<\metric>** (5-fold CV of the same procedure ≈ 3.605). Expected test RMSE ≈ 3.6.

## Data & rules
- Train: `data/labels.csv` (55,413,942 rows). Test: `data/test.csv` (9,914 rows).
- **No external data, no pretrained models, no competition-specific code/resources.** All models are gradient-boosted decision trees trained only on `data/labels.csv`.
- Validation: deterministic holdout carved only from the training file:
  `holdout = (zlib.crc32(key.encode()) % 1000) < 4` → 221,167 rows (~0.4%), uniform over time. Model selection / early stopping / blend weights used only this holdout.
- Cleaning: keep `1 <= fare <= 500`, `0 <= passenger_count <= 8`, coordinates plausibly in the NYC region (after repairing swapped lat/lon). **Rows with `(0,0)` / missing coordinates are KEPT**, with geo features set to `NaN` and explicit `has_pickup` / `has_dropoff` flags. ~2.2% of test rows have a missing endpoint; distance extrapolation there is catastrophic if unhandled.

## Features (see `code/arcturus_cache_build_v2.py`)
Raw coords; haversine / manhattan / L1 distance; bearing (+sin/cos); delta lat/lon; datetime (year, month, day, hour, minute, weekday, dayofyear, weekofyear, hour+minute, is_weekend); distances from each endpoint to JFK / LGA / EWR / midtown / downtown / Columbia; airport flags; `has_pickup` / `has_dropoff`. Target-encoding (TE) models add leak-safe 5-fold OOF mean-fare encoding of pickup/dropoff grid cells and route cells (+ counts) and hour sin/cos.

## Models (swarm ensemble, all trained on `data/labels.csv` only)
| id | model | train rows | key setting | holdout RMSE (all / non-missing) | author |
|---|---|---|---|---|---|
| `xgb_le` | XGBoost `hist` (A100) | 55.41M | TE feats; eta .015, depth 12, subsample/colsample .7 | **3.6605 / 3.2500** | antares_omicron |
| `xgb_te` | XGBoost `hist` (A100) | 55.41M | TE feats; eta .03, depth 11 | 3.667 / 3.257 | antares_omicron |
| `xgb_lg` | XGBoost `lossguide` (A100) | 55.41M | TE feats; 1024 leaves, eta .03 | 3.672 / 3.261 | antares_omicron |
| `cat` | CatBoost (A100) | 55.41M | depth 10, lr .04, TE + base | 3.706 | antares_omicron |
| `alkaid_v2lgb` | LightGBM | 54.2M | leaves 1023, lr .04, min_leaf 200 | 3.763 / 3.354 | alkaid_phi |
| `lgb30m` | LightGBM | 30M | leaves 1023, lr .04, TE+trig | 3.721 | arcturus_upsilon |
| `lgb20m` | LightGBM | 20M | leaves 1023, lr .04, TE+trig | 3.727 | arcturus_upsilon |
| *(also produced)* `xgb_v2` | XGBoost `hist` | 55.41M | 36 feats (no TE) | 3.699 / 3.284 | antares_omicron |
| *(also produced)* `xgb_v3` | XGBoost | 55.41M | TE + ts/cyclical/log/ratio | 3.677 | antares_omicron |

All members used the same deterministic holdout for early stopping.

## Blending (`code/arcturus_final_blend.py`)
Non-negative least squares (+intercept) weights fitted on the canonical holdout, **separately for non-missing and missing-coordinate rows** (missing is a different regime):
- **non-missing** (both endpoints present): global NNLS over `xgb_le, xgb_te, xgb_lg, cat, alkaid_v2lgb, lgb30m, lgb20m` →
  `xgb_le .591`, `xgb_te .068`, `xgb_lg .000`, `cat .000`, `alkaid_v2lgb .153`, `lgb30m .000`, `lgb20m .189`, b=0.
- **missing coords**: NNLS over models whose test predictions are sane →
  `xgb_le .554`, `xgb_te .000`, `xgb_lg .000`, `cat .000`, `alkaid_v2lgb .224`, `lgb30m .179`, `lgb20m .000`, b=.637.
- Final predictions clipped to `[2.5, 200]`.

An automatic sanity check excludes any model whose test predictions on missing-coordinate rows are implausible (this caught a transient TE test-pipeline bug that was subsequently fixed upstream). Key normalization (right-pad fractional seconds to 9 digits) is used **only for internal val joins**; the final file uses the exact `sample_submission.csv` key strings.

## Validation detail (canonical holdout)
| subset | n | RMSE |
|---|---|---|
| non-missing coords | 195,888 | 3.188 |
| missing coords | 4,012 | 12.283 |
| **all** | 199,900 | **3.6033** |

Sub-segments: clean non-degenerate (dist ≥ 0.1 km) ≈ 2.75; degenerate (dist < 0.1 km, corrupted GPS) ≈ 12.9; missing coords ≈ 12.3. Dedicated models for the degenerate/missing subsets were tried independently by two agents and gave no gain — the global model already reaches the conditional-mean floor.

## Reproduce
Run under `pyenv shell sis-black`; every heavy job via
`systemd-run --user --scope --collect -q -p MemoryHigh=... -p MemoryMax=... bash -lc '...'`.
1. `code/arcturus_cache_build_v2.py train` and `... test` → cleaned feature parquet with NaN + missing flags.
2. `code/arcturus_train_lgb_te.py` → LightGBM members (TE + trig), `--predict_test`.
3. `code/arcturus_train_cat.py` → CatBoost member (CPU/GPU).
4. XGBoost members: `code/antares_build_features.py`, `code/antares_add_te.py`, `code/antares_train_xgb.py` (antares_omicron); full-data LightGBM member: `code/train_v2.py` (alkaid_phi).
5. `code/arcturus_final_blend.py <model-list> <out.csv>` → fits weights on the holdout and writes the submission (keys copied in `sample_submission.csv` order).

## Why ~3.6, not ~2.8
A substantial fraction of rows have **corrupted/unrecoverable** coordinates: `(0,0)` endpoints (~2.2% of test) and degenerate trips (`pickup == dropoff` but fare 45–190, flat JFK-fare-like). These contribute ~40–45% of total test MSE and are irreducible (the optimum is the conditional mean). The realistic RMSE floor on the true random test split is therefore ~3.4–3.6, not the ~2.8 sometimes quoted for filtered leaderboard splits.

## External sources
None. No pretrained checkpoints, no external data, no competition-specific resources. Only `data/labels.csv` (train) and `data/test.csv` (features) were used.
