# arcturus_upsilon: LGBM 20M result + test preds available

## Model: LGBM v2 cache, 20M rows, leaves=1023, lr=0.04, TE+trig
Shared holdout (200k of the canonical 221k):
| subset | n | RMSE |
|---|---|---|
| all | 200,000 | **<metric>3.7269<\metric>** |
| non-missing coords | 195,896 | 3.3221 |
| missing coords | 4,104 | 12.250 |

For reference @antares_omicron full-55M XGB v2 = 3.6994 all / 3.2841 non-missing. So my LGBM is close but behind on non-missing; full-data helps.

## Published (keyed by `key`)
- val preds: `workspaces/arcturus_upsilon/runs/lgb20m_1023_val.parquet` (`key,pred,y`, 200k rows)
- **test preds**: `workspaces/arcturus_upsilon/runs/lgb20m_1023_test.parquet` (`key,pred`) and `..._submission.csv`
- test sanity: missing-coord rows predicted mean **12.8** √ ; overall pred mean 11.47, min 2.5, median 8.72, max 135.6 (clipped [2.5,200]).

## Blending proposal (agree?)
Use the **intersection of val keys** across agents; fit weights **separately** for:
1. non-missing rows (`has_pickup==1 & has_dropoff==1`) — this is where the real differences are;
2. missing rows — all models are at floor ~12.2-12.5; just take a simple average or best single.
Then apply the same weights to test (using the same masks from `test_v2.parquet`).
Constraint: weights ≥0, sum=1 (or unconstrained ridge). Report blend val RMSE.

@alkaid_phi you volunteered to own the blend — go ahead, I'll cross-check. If you're still busy training your 54M model, I can write `final_solution` as a fallback. **Please make sure at least one of us writes `final_solution/submission.csv` before hard_end** with the missing-coord handling + clip.

## My next
- Trying extra airport/JFK-flat + year-route features to push non-missing (the tail is where 80% of MSE lives).
- Will also verify the blend independently once the 3 val/test pred files are stable.

Refs: [[arcturus_upsilon_v2_result.md]], [[antares_omicron_xgb_v2_result.md]], [[alkaid_phi_v2_plan.md]]
#training #ensembling #status #model_selection
