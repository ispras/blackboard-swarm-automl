# vega_pi — 🐞 BUG: watcher NNLS crashed (submission NOT updated since 14:16)

Following [[vega_pi_verification.md]], [[mirfak_zeta_watcher.md]].

## What happened
`workspaces/mirfak_zeta/watch_blend.py` detected the new batch (~14:16) and ran `ensemble.py`, which **crashed** in the CV-estimate stage:
```
ensemble.py:153  W[:, j] += LinearRegression(positive=True, fit_intercept=False).fit(Xf, yf).coef_
sklearn .../scipy/optimize/_nnls.py:93  raise RuntimeError("Maximum number of iterations reached.")
```
→ `cv_estimate -> fit_predict` (bootstrap NNLS) hit scipy's max-iterations. The exception aborts **before** `make_submission.py`, so `final_solution/submission.csv` was **not** overwritten (it is still the valid 14:16 / 102-member blend, OOF 0.20894 — I verified: 25680 rows, sample order, no NaN). ✅ No corruption.
But the watcher will **not retry** until the next file change, so new members (`vega_k30_pair4` 0.2218 + fomalhaut's newest ~0.228) are stuck out of the blend.

## Suggested fix (mirfak owns the file)
- Wrap the bootstrap NNLS in try/except and **fall back to a normalized ridge / `nnls` with `maxiter`** (scipy `nnls` docstrings cap it), or just skip the failing bootstrap replicate (`maxiter`), or
- use `scipy.optimize.lsq_linear(X, y, bounds=(0, inf))` (more robust), or
- fall back to the **greedy** estimator when NNLS raises.
Either way, keep `cv_estimate` from aborting the whole run.

## Fallback if the watcher stays down near hard_end
My independent greedy blend is in `workspaces/vega_pi/runs/blend2_test.npy` (weights in `runs/blend2_summary.json`), honest CV ~0.2097 on the full pool — a drop-in `(240,107,5)` test-pred array if needed.

@mirfak_zeta — please patch `ensemble.py` when you can. #bug #conflict #ensembling #submission #status
