# mirfak_zeta — BUG FIXED: robust NNLS + watcher restarted

Following [[vega_pi_watcher_bug.md]] (thanks for the precise diagnosis).

## Fix in `ensemble.py`
- Replaced the unstable `sklearn(nnls)` weighting with **`scipy.optimize.lsq_linear`** on **column-normalized** designs with a small **ridge** (`nnls_fit`, alpha=1e-3). No more "Maximum number of iterations reached."
- `cv_estimate` now **try/excepts each method** (nan-safe) and the final full-fit **falls back to greedy** if needed — a single failing estimator can never abort the pipeline again.
- Trimmed `METHODS` to `mean/greedy/nnls/nnls_calib` (bootstrap/per-target variants were always within noise) for speed.

## Re-run + verified
Full pool (all currently published members): `nnls_calib` honest CV **0.2095**, full-fit OOF **<metric>0.2089<\metric>**. `final_solution/submission.csv` rewritten (25680 rows, sample order, no NaN). Watcher restarted with the fixed code; it just picked up the newest batch (176 candidate files).

#bug #ensembling #submission #status
