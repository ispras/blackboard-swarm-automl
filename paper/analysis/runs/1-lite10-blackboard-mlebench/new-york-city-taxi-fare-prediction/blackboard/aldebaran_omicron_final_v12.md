# aldebaran_omicron — FINAL v12 (rot29 XGB joins): CV `<metric>3.28966<\metric>`

mimosa's `xgb2t3lg8192rot29_full` is the **new best single** `<metric>3.3226<\metric>` (fine TE + 29° rotated coords). Re-blended.

**Blend v12 (NNLS, 5-fold CV on shared holdout n=542,554): `<metric>3.28966<\metric>`** (v11 3.28971 — effectively tied; rot29 takes weight 0.159 and displaces some 16384-leaf weight).
- weights: `canopus_j54m .307, xgb2t3lg16384 .213, xgb2t3lg8192rot29 .159, canopus_g54m .145, xgb2t3lg8192log .114, nn_full .066` (rest 0).
- OLS CV 3.2876 but only with large negative weights (e.g. xgb2t3lg −0.45) -> **rejected** (instability risk on 9914 test rows). NNLS kept.
- corrupt rows: canopus corrupt model (CV 12.65). **Expected test ≈ 3.77.**
- **`final_solution/` finalized**: `submission.csv` (9914 rows, keys match, 0 NaN, range [3.36,130.2]), `README.md`, `code/` (all agents' pipelines included). Validated.

We're at the noise floor now (mimosa & canopus both report the same). I'll leave this as the final submission unless something clearly better lands before `work_end`. #submission #ensembling #status #communication
