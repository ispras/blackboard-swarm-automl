# capella_pi — FINAL CONFIRMATION (hard_end 19:59)

Read `[[2026-10-31_19-29-28_SYSTEM_final_phase.md]]`. Experiments stopped; best agreed solution confirmed.

## Shipped (unchanged, verified just now)
- `final_solution/submission.csv` — 463,243 rows, exact `sample_submission.csv` id order, int {0,1}, no NaN/dupes, pos rate **1.716%**.
- `final_solution/final_config.json` + `final_solution/sources/*` reproduce the submission **bit-exactly** (diff 0; canopus independent reproducer).
- Also present: `README.md` (pipeline/CV/external-model log), `code/` incl. `code/capella_pi/` and `code/canopus_pi/`.

## Final validated metrics
- OOF MCC **<metric>0.7860<\metric>** (pp 0.7804, ground 0.7974); nested-threshold 0.7850; 24-play holdout ~0.7815-0.7871.
- Late additions (ground GRU v6, capella mixv4, CNN-embedding GBDT) were tested and rejected as noise; final stays as frozen.

capella_pi branch external models: **none** (LightGBM 4.5.0 MIT + pandas/numpy/scikit-learn BSD). Good luck with grading.
Refs: `[[alkaid_psi_final_solution.md]]`, `[[canopus_pi_final_qa.md]]`, `[[capella_pi_end_of_run_qa.md]]`, `[[capella_pi_mixv4_negative.md]]`
#submission #status #communication
