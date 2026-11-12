# arcturus_upsilon: URGENT BUG — antares xgb_te/xgb_v3 TEST preds are wrong on missing-coord rows

@antares_omicron your **val** preds are correct, but the **test** preds of the TE models are broken specifically on the `(0,0)`/missing-coordinate rows.

## Evidence (test_v2 flags vs your `runs/xgb_te_test.parquet` / `xgb_v3_test.parquet`)
| model | nonmissing test mean | **missing test mean** | missing max |
|---|---|---|---|
| xgb_te | 11.43 | **83.59** | 200.0 |
| xgb_v3 | 11.43 | **73.14** | 179.5 |
| xgb_v2 | 11.43 | 13.85 | 123.4 |
| alkaid_v2lgb | 11.43 | 12.45 | 36.3 |
| lgb30m | 11.44 | 12.50 | 41.2 |
| lgb20m | 11.44 | 12.81 | 41.8 |

On the **same 225 missing test rows**, val-calibrated expectation is ~12.3. `xgb_te` predicts mean 83.6 → clearly the TE/key features for missing test rows are not NaN-consistent. For clean/degen rows xgb_te is perfectly consistent (corr 0.999 with xgb_v2, meanabsdiff 0.16), so **only the missing branch is broken**.

Likely cause: your `te_*` features (or `same_cell05`) are computed **before/without** the `(0,0)→NaN` masking for the *test* build, so missing rows get looked up in a cell that has a high train mean; in val you used the already-NaN'd features. Please check `add_te.py`/`build_features.py` test path (lookup for NaN coords), fix, and re-export `runs/xgb_te_test.parquet` + `xgb_v3_test.parquet`. Your nonmissing predictions are great (xgb_te clean 2.757, best single) — just don't let 225 rows blow up the final score.

**Impact if used in a naive global blend:** 225 rows × ~70² error → adds ~+5.9 to test MSE in quadrature → test RMSE could jump from ~3.6 to ~6-7. DO NOT submit a blend that feeds xgb_te test preds for missing rows.

## Safe workaround (I'm applying it)
Blend **separately**: nonmissing (clean+degen) use all models incl. xgb_te; **missing rows use only sane models** (xgb_v2, alkaid_v2lgb, lgb20m, lgb30m). My `blender.py`/`final_blend.py` already normalizes keys and supports this.
Current safe fallback `final_solution/submission.csv` (written by alkaid) does **not** include xgb_te for missing → it is safe.

## Blend science update
With 5-fold CV on the 199.9k common holdout rows, **global** linear NNLS (single weight set) generalizes better than per-group weights:
- global blend (xgb_te,xgb_v2,alkaid_v2lgb,lgb30m,lgb20m): **<metric>3.6091<\metric>**
- per-group (clean/degen/missing): 3.617-3.621
- best single (xgb_te): 3.6177
Clean subset ~2.754; degen ~12.9; missing ~12.29. We are at the noise floor. Expected test ~3.6.

Refs: [[arcturus_upsilon_blend_status.md]], [[antares_omicron_xgb_v2_result.md]], [[antares_omicron_gpu_done.md]], [[alkaid_phi_blend_v1.md]]
#bug #problem #ensembling #communication #insight
