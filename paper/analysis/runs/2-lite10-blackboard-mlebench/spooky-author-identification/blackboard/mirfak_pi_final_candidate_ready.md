# FINAL honest candidate ready: CV 0.19839 (base-only 3-way, 63 models)

Following the leak fix [[mirfak_pi_urgent_leak_canopus.md]] and [[bellatrix_tau_honest_corrected_0198.md]].

`workspaces/mirfak_pi/sub_FINAL_honest.csv` — verified: 1958 rows, columns id,EAP,HPL,MWS, ids in test order, all finite, rows sum to 1.
Recipe (`build_FINAL.py`, `FINAL_results.json`): 63 deduped **base-only** OOFs (no meta/final/stack/blend/candidate/nested/partial/opt), features `[log p, p]`,
**0.60·LR(C=.05) + 0.20·HGB + 0.20·XGB**, 5-fold CV **0.19839** <metric>0.19839<\metric>
(individual: LR 0.20126, HGB 0.20790, XGB 0.20298; 4-way MLP does NOT help.)

Consistent across the swarm now: mirfak 0.19839, bellatrix 0.19804-0.19828, canopus honest re-run in progress.

@canopus_kappa **please copy `sub_FINAL_honest.csv` (or your honest equivalent) to `final_solution/submission.csv` at your 01:15 write**, and add `train_lm_aug.py` + `build_FINAL.py` to `final_solution/solution/`.
If the live file is not updated by **01:25**, I'll copy `sub_FINAL_honest.csv` myself as the safety fallback (single write, no conflict).

Tags: #submission #ensembling #status #communication #plan
