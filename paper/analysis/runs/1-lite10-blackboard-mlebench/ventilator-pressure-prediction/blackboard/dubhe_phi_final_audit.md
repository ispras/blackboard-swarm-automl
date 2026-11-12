# dubhe_phi — post-freeze distributional audit of shipped submission (PASS)

Deeper sanity audit of the frozen `final_solution/submission.csv` (md5 `2cc6156b…`), beyond format checks:

- **Inspiratory rows** 229,046 / 603,600 (37.9%, matches train) — all finite.
- **Per-(R,C) inspiratory mean/std vs train** (prediction fidelity; all 9 combos):
  R5C10 19.39/10.08 vs 19.27/10.04 · R5C20 16.10/7.50 vs 16.29/7.66 · R5C50 14.91/7.04 vs 14.59/6.76
  R20C10 19.58/9.51 vs 19.99/9.75 · R20C20 17.79/8.62 vs 17.91/8.57 · R20C50 16.92/8.31 vs 16.48/8.10
  R50C10 17.44/9.60 vs 17.36/9.55 · R50C20 19.49/11.22 vs 19.38/11.12 · R50C50 17.87/9.28 vs 17.81/9.31
  → no group is systematically biased/compressed; model transfers the (R,C) conditioning well.
- **Smoothness**: mean |Δp| between consecutive steps = 0.86 (p99 = 11.5; large jumps only at the inspiration→expiration valve transition, which is unscored). Predictions are not noisy.
- Combined with earlier checks (ids == sample order, 603,600 unique, 0 NaN/inf, range [-1.615,64.914], exact recompute from `manifest_final6.json`+report at ~1e-14, NNLS span residual ~1e-10).

**Final audit PASS.** No changes made. Frozen final remains safe to grade (holdout insp MAE 0.16037, honest CV ≈0.16084). #submission #status #postprocessing #communication
