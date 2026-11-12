# Independent verification: final_solution/submission.csv is IDENTICAL to my independent build (0 diffs, 484,838)

Refs: [[alderamin_xi_FINAL.md]] [[rigel_mu_alltest_done.md]] [[rigel_mu_verify_final_rule.md]] [[altair_epsilon_canon_rule.md]]

I independently rebuilt the final rule from scratch (`workspaces/altair_epsilon/final_rule.py`):
`rigel_test_all.csv` (484,838) -> `canon()` via RDKit round-trip (changed **6,581** rows) -> override with `preds1m_inchi.csv` where `1m valid & score>=0.85` (**use_1m=109,625**, rigel=375,213, fallback=0) -> align to `sample_submission.csv`.

Result: **`workspaces/altair_epsilon/submission_rigel_rule.csv` is row-for-row identical to `final_solution/submission.csv` (0 differences, same order)**. Format checks: 484,838 rows, unique ids, 0 empty, 100% start `InChI=1S/`.

=> The submitted artifact is exactly the verified rule; expected holdout **7.76** (`[[rigel_mu_verify_final_rule.md]]`).

**One doc nit for alderamin**: `final_solution/README.md` step 1 says "confidence ≥ 0.5" — the shipped rule uses **≥ 0.85** (with the 1m-validity requirement). Please fix that number so the README matches the code/submission.

I still have buffer; I'm testing whether **Indigo's InChI parser** (bundled in MolScribe) can recover the ~42% rigel strings that RDKit rejects (currently kept raw at mean Lev ~18). If it can round-trip them canonically, that's a potential further gain. Will only propose if it validates on holdout.

Tags: #submission #postprocessing #status #insight #ensembling
