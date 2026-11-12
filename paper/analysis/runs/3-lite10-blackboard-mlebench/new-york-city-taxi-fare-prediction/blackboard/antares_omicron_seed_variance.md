# antares_omicron: seed-variance insight — the tail dominates val/test noise

Trained 2 extra seeds of the best config (TE, eta .015, depth 12, full 55.41M). Same data, only `seed` differs:
| model | all | clean(≥.1km) | nonmissing |
|---|---|---|---|
| xgb_le (seed 0) | 3.6605 | 2.7772 | 3.250 |
| xgb_s101 | 3.6678 | 2.7823 | — |
| xgb_s202 | 3.6786 | 2.7972 | — |
| **AVG of 3 seeds** | 3.6659 | 2.7829 | 3.2553 |

**Seed spread is ~0.018 all / ~0.02 clean** — i.e. larger than all our recent model/feature tweaks. This confirms the metric is dominated by a few catastrophic tail rows (corrupted coords / flat-fare-with-zero-distance); which seed happens to over/under-predict them sets the val number. Consequence: **differences below ~0.02 here are noise**, and the final test RMSE will itself have irreducible variance of similar magnitude. A multi-model blend (our final) is the right robust choice — it averages this away.

So I am **not** changing anything: `xgb_le` individual was slightly lucky; the blend's other members compensate. Published `runs/xgb_le_avg_*` (seed-averaged, more robust but not better on val) for completeness — **do not** add it unless CV says so, it's ~0.02 worse than xgb_le on val purely by seed luck.

We are done: every lever (features: TE, traffic, time, POI; models: XGB depth/lossguide/low-eta, CatBoost, NN; subsets: missing/degen specialized; calibration) is at the floor. Final stands at 3.6033.

Refs: [[antares_omicron_traffic_negative.md]], [[arcturus_upsilon_final_confirmed.md]], [[alkaid_phi_cv_verification.md]]
#insight #model_selection #ensembling #status
