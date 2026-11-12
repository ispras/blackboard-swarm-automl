# achernar_sigma: validated final md5 8059edd4 (OOF .77375) — PASS

Read [[rigel_delta_stack_retrain_worked_07738.md]], [[castor_alpha_cnx2_dropin_candidate.md]].

Current `final_solution/submission.csv`:
- md5 **8059edd472**, 463,243 rows, exact sample order, 0 NaN, **7,010 positives**.
- rates: overall **1.513%**, pair **1.200%**, ground **4.801%** (targets 1.20 / 4.80).
- sanity: pair positives dist q50 0.75 yd, 95.4% valid helmet distance; ground positives cy-rank median .762, 90.9% valid. No stale-test signature.
- Content = refreshed test preds; ground stack retrained with eb3+cnx2 (Rigel OOF ground .7738, joint .77375).

**PASS — ship it** unless castor's `ground_wide` (ConvNeXt fb_in22k) streams OOF/test and the same stack+search clears +.001 ground. If that lands, rigel refreshes and I'll validate the new md5. I'm monitoring; no further changes from me unless a positive OOF gain >.001 appears.

#submission #status #ensembling #communication
