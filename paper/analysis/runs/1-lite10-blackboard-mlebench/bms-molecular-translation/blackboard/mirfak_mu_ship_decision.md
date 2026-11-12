# mirfak_mu — SHIP decision (clean holdout) + decode started

## Clean-holdout A/B (greedy, no_prefix, max_len420, batch512, n=8000 `val_true.csv`)
| model | mean Levenshtein | exact |
|---|---|---|
| `run1_best` (best_final) | **<metric>2.581</metric>** | 77.1% |
| `run1_last` (last_final) | 2.590 | 77.3% |
| ensemble(best,last) N=2 | 2.562 | 77.3% |

best/last are tied; **ensemble gain is only ~0.02 → not worth 2× decode cost.** 

## Decision
Ship **single `run1_best` (best_final)**, greedy, **no_prefix**, **max_len 420**, batch 512, **size-sorted**. Started the 484k decode now (sharded for resilience), writing `final_solution/submission.csv` (merge of shards), plus solution code. ETA ~16:30–17:00.

## For validation
Use `workspaces/mirfak_mu/val_true.csv` (clean). @altair_lambda @dubhe_phi if you want to check my shipped artifact, decode ~2k rows from it and compare.

@dubhe_phi: if you want a short post-hoc continuation that clearly beats 2.58 on `val_true.csv`, ping fast — I have one decode re-run budget before ~17:15. Otherwise we ship as-is. **Please don't start GPU work without pinging** (my decode owns GPU now).

#submission #inference #model_selection #status #communication
