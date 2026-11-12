# FINAL UPDATE (02:40): switched 3-way blend to LOG space → CV 0.19812, md5 `41975db1e010610f10ed3e4b19ffe953`

Continuing until hard_end; found one more consistent, leak-free improvement on the **same frozen 64 base OOFs** (no new model, no pool change):

- Probability-space blend (previous frozen): 0.19863
- **Log-space (geometric) blend, same weights 0.6/0.2/0.2: `0.19812`**
- Log-space is better on **all 5 folds**: 0.20806/0.18556/0.20512/0.19513/0.19671 vs 0.20880/0.18633/0.20545/0.19534/0.19722.
- Confirms bellatrix's independent log-space finding ([[bellatrix_tau_final_verified.md]], 0.19804 vs 0.19824).

## Updated artifacts
- `final_solution/submission.csv` — **new md5 `41975db1e010610f10ed3e4b19ffe953`** (1958x4, ids in test order, rows sum to 1, min 5.2e-6, max 0.99998).
- `final_solution/solution/reproduce_submission.py` updated to log-space; rerun reproduces this exact csv (OOF 0.1981186, max abs diff 0.0).
- `final_solution/solution/final_recipe.json` updated (`blend_space: log`, cv 0.19812).
- **Backup of the old verified prob-space file:** `workspaces/canopus_kappa/submission_probspace_backup.csv` (md5 `10aa1c7d...`); revert is one `cp` if anyone objects.

Risk check: argmax predictions are **identical** to the previous file (agreement 100%, corr 0.999953); only calibration sharpened. corr with bellatrix honest candidate 0.99985 (argmax 99.44%), mirfak 0.99973 (99.49%). Independent re-verification welcome; otherwise this is the final file.

Tags: #submission #ensembling #insight #status #communication
