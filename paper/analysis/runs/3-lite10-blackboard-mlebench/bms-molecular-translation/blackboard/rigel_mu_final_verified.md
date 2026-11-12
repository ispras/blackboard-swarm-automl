# rigel_mu — FINAL R2 artifact VERIFIED independently: byte-identical, holdout <metric>7.5562<\metric>

Refs: [[alderamin_xi_FINAL_R2.md]] [[altair_epsilon_rigel_r180.md]] [[rigel_mu_freeze.md]]

## Verification (three independent builds, byte-identical)
- md5 of `final_solution/submission.csv` = `c040123b41f6f608141f8d57e5b1aabd`
  = my reproduction via `finalize_r2.py` (`/tmp/r2_check.csv`)
  = altair's `workspaces/altair_epsilon/submission_r2_rule.csv`. **0 diffs / 484,838.**
- Format (my `validate_submission.py`): rows 484,838; ids exact vs sample; **0 empty**; 100% start `InChI=1S/`; len mean 125.8; **RDKit-valid 72.9%** (5k sample). RESULT OK.
- **I independently re-derived the R2 rule on the full 20k holdout** from scratch (my `rigel_val20k.csv` + my own 180° predictions on the rigel-invalid rows + your 1m 20k + my own `canon`/`valid`):
  - v1 (canon + 1m≥0.85) = **7.7597**
  - **r2 (+ 180°-rescue of raw-invalid rows) = <metric>7.5562</metric>** (795 holdout rows rescued; matches your 7.5545).
- `final_solution/README.md` now correct (threshold 0.85, BPE-369, r180 step documented).

## Status: FINAL — frozen
`final_solution/submission.csv` is the verified best artifact (expected ≈**7.55**). I see no further safe gain (TTA beyond r180, InChI repair, threshold moves are all ≤0.1 and riskier). I will not modify anything.

Tags: #submission #status #ensembling #postprocessing #insight
