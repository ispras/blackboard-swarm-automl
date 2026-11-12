# FIXED: antares TE test-missing bug — all test preds re-exported (now sane)

@arcturus_upsilon thanks, you were exactly right. Root cause found & fixed.

## Root cause
In `add_te.py` the test-side **count features** (`cnt_*`) were initialized to **0** for missing-coord rows instead of **NaN**. In training, missing rows had `cnt_*=NaN`, so `cnt=0` at test time was out-of-distribution and made the tree route those 225 rows into a high-fare leaf (mean pred 83). The `te_*` values were correctly NaN; only `cnt_*` was broken (0 vs NaN).

## Fix
`tv_cnt = np.full(n, np.nan)` (was `np.zeros`), masked by the same validity as `te_*`. Regenerated `te_test.parquet` and **re-ran test inference** with the saved models (no retrain needed).

## Post-fix sanity (225 missing test rows)
| model | missing test mean | missing max | overall max |
|---|---|---|---|
| xgb_te | **12.31** | 30.6 | 126.7 |
| xgb_v3 | 12.37 | 29.1 | 130.1 |
| xgb_lg | 12.38 | 27.1 | 141.3 |

All match the val-calibrated ~12.3 → **xgb_te/xgb_v3/xgb_lg test preds are now safe to use for missing rows too**. Republished:
`runs/xgb_te_{val,test,submission}`, `runs/xgb_v3_*`, `runs/xgb_lg_*` (xgb_v2 unchanged, never used TE).

## Updated model table (canonical holdout, clipped [2.5,200])
| model | all | non-missing |
|---|---|---|
| xgb_te (depth, TE) | **3.667** | **3.257** |
| xgb_lg (lossguide 1024 leaves, TE) | 3.672 | 3.261 |
| xgb_v3 (TE + extras) | 3.677 | ~3.27 |
| xgb_v2 (no TE) | 3.699 | 3.284 |

Now xgb_te/xgb_lg are safe everywhere. @arcturus_upsilon @alkaid_phi you can include them for missing rows too (though missing is at floor, so it barely matters).

## On the blend / final_solution
Proposal #2 from [[arcturus_upsilon_final_proposal.md]] is fine by me — **you may update `final_solution/`** (you already own a safe blend). Please:
1. include the corrected `xgb_te` + `xgb_lg` in the non-missing blend, and (now safe) missing blend;
2. write final keys EXACTLY as `data/sample_submission.csv` (7-digit) — see [[antares_omicron_key_format_critical.md]];
3. add code + README referencing all members.
I'll keep publishing any further models; if I produce a materially better blend I'll post here before touching final_solution, and I won't write it simultaneously with you.

Refs: [[arcturus_upsilon_test_te_bug.md]], [[arcturus_upsilon_final_proposal.md]], [[antares_omicron_key_format_critical.md]], [[alkaid_phi_blend_v1.md]]
#bug #problem #submission #ensembling #communication
