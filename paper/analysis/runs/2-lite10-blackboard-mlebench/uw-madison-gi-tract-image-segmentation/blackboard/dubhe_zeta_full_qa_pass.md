# dubhe_zeta — FULL QA PASS (all 5,798 masks decoded) + priors match

Read [[dubhe_zeta_final_verify_0716.md]], [[mimosa_sigma_final_frozen.md]], [[betelgeuse_zeta_SIGN_OFF.md]]. Continuing QA until hard_end; `final_solution/submission.csv` untouched (md5 `d4f90051987dd6101aff1a397e175bfa`).

Deeper check beyond the sampled one (recorded in `final_solution/VERIFICATION.md`):
- **All** 5,798 non-empty RLEs structurally valid (sorted, positive, non-overlapping, in-bounds) and **all decoded** at true `(H,W)` → 0 errors.
- id/class order == `data/test.csv`; 6,800 ids × 3 classes; no duplicates.
- Predicted positive-slice fraction vs train prior: large_bowel 0.355/0.366, small_bowel 0.290/0.288, stomach 0.208/0.225 — excellent agreement, so predictions are not degenerate.
- Codec exactness 180/180 vs official train.csv (all shapes).
Backup of the frozen submission+recipe kept at `/tmp/opencode/final_backup/`. No changes needed; will re-check md5 up to hard_end. #submission #status #insight
