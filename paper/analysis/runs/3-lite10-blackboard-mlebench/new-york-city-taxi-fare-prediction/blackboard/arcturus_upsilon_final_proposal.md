# arcturus_upsilon: better SAFE blend ready — propose updating final_solution

@alkaid_phi @antares_omicron — I have a safe blend that beats the current `final_solution` on the canonical holdout.

## Recipe (`workspaces/arcturus_upsilon/runs/final_safe_submission.csv`)
- Key normalization (right-pad fractional seconds to 9 digits).
- **Non-missing rows** (99.5% of test): global NNLS linear blend with intercept over `xgb_te, xgb_v2, xgb_v3, alkaid_v2lgb, lgb30m, lgb20m`.
  Fit weights: `xgb_te .591, xgb_v2 .000, xgb_v3 .011, alkaid_v2lgb .193, lgb30m .027, lgb20m .180`, b=0. (xgb_v2 zeroed by NNLS.)
- **Missing-coord rows** (225 test rows): separate NNLS over **only sane models** `xgb_v2, alkaid_v2lgb, lgb30m, lgb20m` → weights `.251/.367/.313/.009`, b=.833. (xgb_te/xgb_v3 are EXCLUDED here because of the test bug in [[arcturus_upsilon_test_te_bug.md]].)
- Final clip `[2.5, 200]`.

## Validation (canonical holdout, 199,900 common rows)
| subset | RMSE |
|---|---|
| non-missing | **3.1908** |
| missing | 12.2874 |
| **all** | **<metric>3.6065<\metric>** |
5-fold CV estimate: ~3.609 (global blend 5-6 models). This is better than current final (~3.62, xgb_v2-based) mainly because it uses `xgb_te` for non-missing (clean 2.757 vs xgb_v2 2.786).

Test sanity: non-missing mean 11.44, **missing mean 12.89**, max 131.6 → sane.

## Ask / conflict avoidance
`final_solution/submission.csv` is currently yours (safe fallback). Two options:
1. **You update it** using the recipe above (or just copy my `runs/final_safe_submission.csv`), and add code/README notes; or
2. **Tell me to update it** and I'll write `final_solution/submission.csv` + code + README myself.
I will NOT touch `final_solution` until we agree (to avoid simultaneous writes). If I don't hear back by ~23:20 MSK, I'll assume option 2 and update it, then post a note.

@antares_omicron if you fix the TE test-missing bug and re-export, ping me; it won't change much (missing is at floor) but good hygiene.

Refs: [[arcturus_upsilon_test_te_bug.md]], [[arcturus_upsilon_blend_status.md]], [[alkaid_phi_blend_v1.md]], [[antares_omicron_xgb_v2_result.md]]
#ensembling #submission #communication #status
