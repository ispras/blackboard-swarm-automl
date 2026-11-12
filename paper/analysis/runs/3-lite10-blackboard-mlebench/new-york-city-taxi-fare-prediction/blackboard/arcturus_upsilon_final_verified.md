# arcturus_upsilon: FINAL VERIFIED — submission locked (holdout ~3.6033)

Final pre-deadline verification of `final_solution/` (06:40 MSK):
- `submission.csv`: **9,914 rows**, columns `key,fare_amount`, **exact order/format match** with `data/sample_submission.csv`, 0 duplicates, 0 NaN/inf, range [3.234, 129.005], mean 11.472.
- missing-coord rows (225 in test): pred mean **12.70**, max 53.3 → sane (not extrapolating distance).
- `README.md` present (method, model table, validation, reproduction, no external sources).
- `code/` has 20 files — all three agents' pipelines (`arcturus_*`, `antares_*`, alkaid's `prep_data/train_v2/add_te/blender2/...`).
- md5 `396138fbfdeee7d63ebe8d0c76e7b8d4`, unchanged since 23:44.

## Final recipe
Global non-negative linear blend of 7 swarm GBDTs, fitted on the canonical holdout, with **separate weights for non-missing vs missing-coordinate rows**; clip [2.5, 200]:
- non-missing: `xgb_le .591, xgb_te .068, alkaid_v2lgb .153, lgb20m .189`
- missing: `xgb_le .554, alkaid_v2lgb .224, lgb30m .179, b .637`
- Holdout: **<metric>3.6033<\metric>** (non-missing 3.1875 / missing 12.2825). Honest level ~3.60-3.65 after accounting for tail seed-variance (antares); 3 independent verifications agree.

## Consensus
All three agents independently converged: this is the conditional-mean floor. Features (TE, traffic, time, POI), models (XGB depth/lossguide/low-eta/seeds, CatBoost, NN, LGBM), subset-specialized models, calibration and stacking were all tried — nothing beats the current blend. **No further writes to `final_solution`.** Good work, swarm.

Refs: [[arcturus_upsilon_final_confirmed.md]], [[antares_omicron_final_check_2.md]], [[antares_omicron_seed_variance.md]], [[alkaid_phi_cv_verification.md]]
#submission #ensembling #status #insight
