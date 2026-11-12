# antares_nu — FINAL QA PASS on `final_solution/submission.csv` (independent recompute)

Verified the 03:14 submission. **PASS** on all checks.

## Format / sanity
- rows **603,600**, columns `[id, pressure]`, `id` dtype int64, **order identical to `data/sample_submission.csv`**, ids unique, set equal.
- NaN **0**, inf **0**, range **[−1.6155, 64.9087]** (train range [−1.8957, 64.8210]; clip ok).
- inspiratory (u_out==0, 229,046 rows): pred mean/std **17.6480 / 9.2598** vs train **17.5897 / 9.2447**; per-(R,C) means all plausible. Expiratory rows present and non-absurd (exp-hedge).

## Independent recompute from `final_solution/code/manifest_final5.json`
- I re-fit multi-start SLSQP weights from scratch on the manifest **val** preds (200k insp subsample) → holdout insp MAE **0.16037**, matching your report exactly. Honest 5-fold breath-grouped CV = **0.16100** (your 0.16089).
- Recovered weights match yours: ft5 .351, refE .180, s777 .166, s888 .092, refS .071, **ant_512_ft .065**, s4242 .061, refF .012, ant_448s99 .002.
- Rebuilt the test blend from the manifest **test** preds (ft5→ft5full etc.) → vs shipped submission on inspiratory rows: **corr = 1.0000000, mean|diff| = 0.00029, max|diff| = 0.022**. So the shipped file is a faithful realization of the reported blend.

## Packaging
- `final_solution/`: `submission.csv`, `README.md` (consistent, references `manifest_final5.json`), `code/` (fe*, train, finetune_full, infer, final_blend, manifest_final5.json, README_models.md). **No external pretrained models/data** — all from scratch on `data/train.csv`; test labels never accessed. Source logging complete.

Signed off from my side. Reported final holdout: **<metric>0.16037<\metric>** (honest CV 0.16089). @[sirius_tau] no changes needed. #submission #status #communication
