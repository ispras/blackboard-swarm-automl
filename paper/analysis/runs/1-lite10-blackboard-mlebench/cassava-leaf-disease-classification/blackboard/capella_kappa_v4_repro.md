# capella_kappa — v4 reproducible (make_final.py NaN-safe)

Refs: [[capella_kappa_nan_fixed.md]], [[capella_kappa_nan_scope.md]]

Made the shipped combiner NaN-robust and verified end-to-end:
- `final_solution/code/make_final.py` now sanitizes NaN (per image/class -> mean of finite members, then 1/5 fallback) before averaging.
- Re-ran it in `--mode arith` with the 11 member paths -> output **md5 matches the frozen v4 exactly**: `e32f8872453c714315ec0d346409a01d`.
- fold-0 holdout (val probs) = **0.9012** unchanged; label dist 0:136 1:276 2:266 3:1673 4:325.

**Final state:** `final_solution/` = `submission.csv` (v4, 1 NaN-induced label fixed) + `submission_meta.json` (with `nan_fix`) + `README.md` (erratum) + `code/` (NaN-safe combiner). Reproducible, validated, honest.

#bug #submission #status #communication
