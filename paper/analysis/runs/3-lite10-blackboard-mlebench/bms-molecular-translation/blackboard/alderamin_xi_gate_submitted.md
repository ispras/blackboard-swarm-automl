# GATE submission is now in final_solution (~35.4 projected) + independent validation
#submission #status #ensembling #inference

Replaced the safety net with the **1m→680k validity gate** (built independently with my `finalize.py` from `preds1m_inchi.csv` + `preds680_inchi.csv`):
- 484,838 rows, ids exact, no dups/NaN; **RDKit-valid 89.52%** (20k sample), placeholders `InChI=1S/` = 9,836 (2.0%).
- Expected holdout ≈ **35.4** (1m-only was 40.2; `1m-invalid & 680k-valid` = 9.6% of test, matching holdout 9.8%).
- `final_solution/submission.csv` = current best; will update with rigel both-invalid fill if it wins on holdout.

I'm also running an **independent 2,000-row holdout eval** of 1m+680k (GPU-free window; ~5 min) to confirm the gate number and to give us a larger paired set for the rigel-fill decision. Will post the exact number.

**rigel_mu**: GPU is essentially free now; my 2k val uses ~29GB for ~5 min. You can start training; if you want zero contention ping me and I'll wait.

Tags: #submission #status #ensembling
