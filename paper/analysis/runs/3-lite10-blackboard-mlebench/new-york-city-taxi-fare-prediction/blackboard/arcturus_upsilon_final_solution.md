# arcturus_upsilon: FINAL solution written to final_solution (val 3.6073)

`final_solution/submission.csv` updated (safe, exact sample keys, 9,914 rows, no NaN, range [3.21, 133.7]).

## What is in it
Blend of **5 diverse swarm GBDTs** (global non-negative linear weights on the canonical holdout, fitted **separately** for non-missing vs missing-coord rows; clip [2.5,200]):
- non-missing: `xgb_te .492, xgb_lg .157, cat .000, alkaid_v2lgb .203, lgb30m .148`
- missing: `xgb_te .399, cat .017, alkaid_v2lgb .326, lgb30m .210, b .682`

| subset | n | RMSE |
|---|---|---|
| non-missing | 195,888 | 3.192 |
| missing coords | 4,012 | 12.285 |
| **all (holdout, in-sample weights)** | 199,900 | **<metric>3.6073<\metric>** |
| 5-fold CV estimate | | ~3.605 |

Models used: `xgb_te` (best single, 3.667 all / 3.257 nonmiss), `xgb_lg` (lossguide), `cat` (CatBoost full 55M), `alkaid_v2lgb` (54.2M LightGBM), `lgb30m` (my TE LightGBM). `cat`/`xgb_lg` add algorithm diversity; NNLS drops redundant weight. Also produced but not needed in final: `xgb_v2`, `xgb_v3`, `lgb20m`.

## Files
- `final_solution/submission.csv`
- `final_solution/README.md` — full method, model table, validation, reproduction, **no external sources**.
- `final_solution/code/` — my pipeline (`arcturus_cache_build_v2.py`, `arcturus_train_lgb_te.py`, `arcturus_train_cat.py`, `arcturus_final_blend.py`, `arcturus_blender.py`) alongside alkaid's code.

## Key wins that got us here
1. **`(0,0)`/missing-coordinate rows kept with NaN + flags** ([[arcturus_upsilon_critical_missing_coords.md]]) — 2.2% of test, would otherwise blow up.
2. **Key-format discipline** for the final file ([[antares_omicron_key_format_critical.md]]).
3. **TE test-pipeline bug** caught & fixed ([[arcturus_upsilon_test_te_bug.md]], [[antares_omicron_te_bug_fixed.md]]).
4. **Missing vs non-missing separate blend weights**; global (not per-segment) weights generalize best in CV.

## Status / open
We are at the noise floor (clean non-degenerate ~2.75; ~40% of MSE is corrupted `(0,0)`/degenerate GPS). Any further gains would be <0.01 RMSE. If @antares_omicron or @alkaid_phi produce a materially better member (>0.02 on non-missing), post it and I'll re-blend + update final. I will keep monitoring the board until near hard_end and re-run the blend if new val+test preds appear.

Refs: [[arcturus_upsilon_final_proposal.md]], [[arcturus_upsilon_ownership.md]], [[antares_omicron_te_bug_fixed.md]], [[alkaid_phi_blend_v1.md]], [[arcturus_upsilon_critical_missing_coords.md]]
#submission #ensembling #status #insight
