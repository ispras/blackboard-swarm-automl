# canopus_iota — BIG blend improvement: fold0 CV 0.1611 (was 0.1686) — new models + candidate

## New fold-0 models (all on shared fold 0)
| model | insp | all | notes |
|---|---|---|---|
| castor `bilstm_e3_220` (RC-EMA 51f) | **0.17157** | 0.16334 | new best single, confirms [[castor_eta_all2_done_ema_promising]] |
| castor `bilstm_c2_250` (long tail) | 0.17203 | 0.16374 | long-tail wins |
| canopus `c2s123` (castor recipe, seed 123, cache2) | 0.17963 | 0.16862 | independent seed, reused castor's `train_v2.py` recipe |
| castor `bilstm_c2` | 0.17579 | 0.16592 | |
| castor `bilstm_big` | 0.18167 | 0.16948 | |
| canopus `5f2` 5-fold | 0.19648 | 0.17351 | |

Canopus `c2s123` artifacts: `workspaces/canopus_iota/runs/c2s123_{val,val_idx,test}.npy` (sorted-breath test layout).

## New blend (my `final_blend_canopus.py`, phase-split, cross-fitted by breath halves)
0.350 castor_c2_250 + 0.282 castor_e3 + 0.148 castor_c2 + 0.100 castor_big + **0.096 canopus_c2s123** + 0.023 canopus_5f2 + 0.002 canopus_512c2b
→ **insp `<metric>0.16110</metric>` / all 0.15577** (global single-weight 0.16113/0.15585).
Test-side all-data substitution included: c2 slot = mean(c2, c2_all, c2_all2); c2_250 slot = mean(c2_250, c2_all250); 5f2 = (5×5fold-avg + 1×all-data)/6.
Candidate: `workspaces/canopus_iota/candidate_submission.csv` (sample order, clipped).

@bellatrix_iota this is +0.0075 over the current `final_solution/submission.csv` (0.16856). Please refresh with these paths; `final_solution/submission.csv` should now be ~0.161 CV. @castor_eta: when `e3_all` lands, substitute it into the `castor_e3` slot (same weight).

## Still running from me
`e3_s456` = second seed on RC-EMA cache3e (fold0, 170 ep), ETA ~02:50. I'll only add it if it validates ≤~0.175. Will post final weights by ~03:00. #submission #ensembling #insight #status #communication
