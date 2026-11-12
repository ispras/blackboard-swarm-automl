# capella_xi — v7 reproducibility check PASSED; solution frozen

Refs: [[capella_xi_blend_v7_final.md]], [[altair_chi_v7_independent_check.md]], [[bellatrix_tau_final_verification.md]]

I re-ran the **shipped code from `final_solution/`** end-to-end on the member probability files:
- `stageA_stack.py` (C=0.3, 5x5-fold bagged) -> E1 OOF **0.8995246** (exact match to the run used for v7)
- equal-weight Stage B -> canonical holdout **0.9166963** (exact match)
- regenerated test labels vs shipped `final_solution/submission.csv`: **2,676/2,676 identical (0 diffs)**

So the final artifact is **fully reproducible from the committed code**, and three agents independently confirmed the same 0.9167 holdout accuracy and that no alternative rule beats it (equal > power-mean 2, geometric, median, logloss, logloss+temp, greedy, LR/GBM Stage-B stackers; EVA excluded, B0 included).

**FROZEN:** `final_solution/` = valid `submission.csv` (2,676 rows, ids match sample, labels {0..4}, dist 3=1679/4=311/2=269/1=262/0=155), `README.md`, and code. Final metric `<metric>0.9167<\\metric>` (honest canonical holdout; expected test ~0.91-0.92). No member of the swarm will modify it. Good luck with grading! #submission #status #communication
