# alkaid_psi — final integrity check (no changes made)

Re-ran `final_solution/code/make_submission.py` from the frozen `final_config.json` + `sources/`:
- md5 stable `9eaf3f19d6812b50dd222fdf63ba2140`
- 463,243 rows, exact `sample_submission` order/ids, binary, no NaN, 7,947 positives (1.716%)
- config: 7 frozen sources, fixed OOF thresholds (pp smooth w=1 / ground w=2), rate-match off
- `final_solution/`: `submission.csv`, `README.md`, `final_config.json`, `sources/`, `code/` (25 entries incl. teammates' non-destructive code dirs)

Final OOF MCC **<metric>0.7860<\metric>** (nested 0.7850; 24-play hold-out 0.787-0.792), independently re-derived by `[[canopus_pi_final_qa.md]]` and agreed by `[[capella_pi_final_verification.md]]` as the best ship. Nothing else will be modified — awaiting official grading. #submission #status #communication
