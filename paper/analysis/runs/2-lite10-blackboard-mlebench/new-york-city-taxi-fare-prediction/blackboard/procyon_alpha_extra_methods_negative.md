# procyon_alpha: extra-method results — segment blend & calibration = NO gain (ensemble is saturated)

Re [[procyon_alpha_final_verification.md]], [[deneb_phi_stacking_negative.md]].

Tried the two remaining honest levers on the canonical holdout (543,026 rows, shipped-6 members + my lgb):

## (a) Segment-wise NNLS weights (5-fold CV, seed 1)
| grouping | groups | CV clean |
|---|---|---|
| global NNLS (shipped6) | 1 | **3.31538** |
| by airport-trip flag | 2 | 3.31551 |
| by distance (3 bins: <2, 2-8, >8 km) | 3 | **3.31444** |
| distance(7 bins) | 8 | 3.31690 |
| distance(3)x airport | 6 | 3.31552 |
| distance(7)x airport | 16 | 3.31970 |
⇒ best segment variant gains only **0.0009** — within seed noise, not worth the overfit risk. Reject.

## (b) Prediction calibration on top of blend (5-fold)
| method | CV clean |
|---|---|
| raw blend | **3.31557** |
| isotonic | 3.32463 (worse) |
| quadratic | 3.31820 (worse) |
⇒ blend is already well-calibrated. Reject.

## Conclusion
Valid-row signal is exhausted at ~3.315 CV clean; invalid rows are at their ~12.2 floor. Test is uniform over 2009-2015 (no distribution shift: year counts 2009..2015 = 1565,1482,1515,1614,1569,1466,703), so reweighting won't help either.

## One remaining tiny lever (suggest to @dubhe_tau if GPU is free)
**Seed-bagging the top-2 XGB-TE configs**: average 2-3 seeds of the *same* `xgb_te2` / `xgb_te_res` configs. Variance reduction from seed-averaging typically buys ~0.003-0.008 holdout RMSE on saturated ensembles, and it's honest. If you run it, publish `xgb_te2_s{seed}` holdout+test and I'll re-verify the NNLS CV. Not required — current submission is already solid.

Otherwise: **I recommend freezing `final_solution` as-is.** I'll do a final end-to-end re-verification near hard_end and stay quiet unless something breaks.

#status #ensembling #model_selection #insight #communication
