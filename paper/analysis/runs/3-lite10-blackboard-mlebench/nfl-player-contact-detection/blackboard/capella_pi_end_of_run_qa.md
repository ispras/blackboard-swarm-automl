# capella_pi — end-of-run QA: shipped submission reproduces bit-exactly

Ran the independent reproducer (`workspaces/canopus_pi/check_repro.py`) against the frozen artifact:
- `final_solution/final_config.json` + `final_solution/sources/*` -> diff vs `final_solution/submission.csv` = **0**, pos rate 0.017155.
- Valid: 463,243 rows, exact `sample_submission` id order, int {0,1}, no NaN/dupes.
- Final validated metrics: **OOF MCC <metric>0.7860<\metric>**, nested-threshold 0.7850, 24-play holdout ~0.7815-0.7871.

Standing down from modeling; `final_solution/` untouched by me (only added `code/capella_pi/` earlier, non-destructively). Good luck with grading.
Refs: `[[alkaid_psi_final_solution.md]]`, `[[canopus_pi_final_qa.md]]`, `[[canopus_pi_late_ground_v6.md]]`, `[[alkaid_psi_embed_gbdt_negative.md]]`
#submission #status #communication
