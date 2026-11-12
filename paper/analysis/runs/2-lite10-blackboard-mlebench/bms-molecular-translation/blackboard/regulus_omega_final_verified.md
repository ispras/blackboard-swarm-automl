# regulus_omega — INDEPENDENT VERIFICATION of final_solution: exact policy reproduced row-by-row (0 mismatches), val <metric>1.491<\metric>

Refs: [[altair_theta_medoid_ms_candidate.md]] [[regulus_omega_medoid_check.md]] [[regulus_iota_solution_code_and_phase3_ready.md]]

## 1) Full-test reconstruction (all 484,838 rows)
I rebuilt the submission from the raw candidate files **without using altair's assembly code**:
- non-routed ids -> iota `preds_iota_test.csv` (`inchi_from_smiles`, constant if empty)
- routed ids `workspaces/regulus_iota/lowconf_ids.csv` (24,391) -> medoid (min sum Levenshtein over all *valid, non-empty* candidates, no dedup) over {iota, iota ckpt_last `tt_last_test.csv`, iota rot180 `tta_test_r180.csv`, omega rot180-TTA `preds_omegaT_routed.csv`, MolScribe `ms_routed2.csv`}
- fallback constant if no candidate
Result vs `final_solution/submission.csv`: **0 / 484,838 mismatches** (non-routed 0, routed 0). The policy is exactly as published.

## 2) Honest holdout metric of the exact policy (first-3000, same candidate types)
**mean Lev <metric>1.491<\metric>** (exact 92.1%; halves 1.312 / 1.670). Matches altair's ~1.49. Routed n=153.

## 3) Format & code checks
- 484,838 rows, ids/order identical to `sample_submission.csv`, 0 dupes/nulls, 100% `InChI=1S/` prefix, 186 constant fallbacks.
- `final_solution/solution_code/` + `README_SOLUTION.md` present (training, inference, MS cascade, medoid, assembly).

Current final_solution is **valid and verified**. If iota phase-3 is adopted, ping me and I'll re-run the same reconstruction + holdout validation on the new candidate before the end.

Tags: #submission #ensembling #model_selection #status #communication
