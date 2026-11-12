# rigel_mu — FINAL Rule C VERIFIED independently (my own from-scratch build = byte-identical; holdout <metric>6.7705<\metric>)

Refs: [[alderamin_xi_FINAL_RULEC.md]] [[altair_epsilon_final_verified.md]] [[rigel_mu_final_verified.md]]

## Independent verification (PASS)
- I rebuilt Rule C **from raw artifacts with my own script** (`workspaces/rigel_mu/rule_c.py`, no reuse of their assembly code):
  `raw-valid -> canon(rigel); elif r180-valid -> canon(r180); elif 1m-valid & score>=0.78 -> 1m; else canon(rigel)`.
- **`final_solution/submission.csv` == my build == altair's build**: md5 `53c7485e83889231a853c03d6a091a21` (0 / 484,838 diffs on all three).
- **My independent full-20k holdout re-derivation**: Rule C T=0.78 = **<metric>6.7705</metric>** (T=0.75 6.8110, T=0.80 6.7985; matches 6.7704/6.7711 reported by altair/alderamin). Root cause confirmed: on rigel-valid rows (n=11,595) canon(rigel)=1.08 vs 1m=34.10, so the old rule was discarding near-perfect predictions.
- Format (my `validate_submission.py`): 484,838 rows, ids exact, **0 empty**, 100% `InChI=1S/`, len mean 125.8, **RDKit-valid 76.6%** (5k sample). RESULT OK.

## Final
`final_solution/submission.csv` = **Rule C**, expected official mean Levenshtein ≈ **6.77**.
Trajectory: MolScribe 40.3 → validity gate 35.2 → rigel-alone 8.58 → canon+r180+1m 7.47 → **Rule C 6.77**.
I am not changing anything further; monitoring the frozen artifact until the deadline.

Tags: #submission #status #ensembling #postprocessing #insight
