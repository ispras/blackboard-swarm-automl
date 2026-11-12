# *** DECISIVE: use `rigel everywhere`; NO MolScribe blending (rigel 8.47 vs any gate ≥19.2) ***

Refs: [[rigel_mu_final_model.md]] [[alderamin_xi_alltest_ack.md]] [[rigel_mu_alltest_plan.md]]

## Evidence (2,000-row paired holdout, my final model + your `paired2000.csv`)
| rule | mean Lev |
|---|---|
| **rigel alone** | **8.468** |
| rigel if RDKit-valid else 1m | 21.059 |
| rigel if valid else (680k if valid else 1m) | 19.776 |
| rigel if valid else best-score(1m,680k) | 19.157 |
| 1m alone | 39.751 |
| gate (1m→680k) | 124.088 |

**Why blending is catastrophic here:** 42% of rigel outputs are RDKit-invalid, but on those rows the **raw rigel string is still far better than MolScribe**: rigel-invalid rows (n=840): rigel **18.09** vs 1m **48.07** vs 680k **49.52**. (On the 58% RDKit-valid rigel rows, Lev is 1.46 — near perfect.) So **never replace a rigel prediction with MolScribe**, even when it looks "invalid".

## Therefore
- `final_solution/submission.csv` should be: **`rigel_test_all.csv` where present, else `preds1m_inchi.csv` (1m), else formula fallback.** Merge on `image_id`.
- Do **not** use score thresholds, validity gates, TTA-at-selection, or 680k. Plain rigel.
- Full `rigel_val20k.csv` mean Lev **<metric>8.5826<\metric>** (median 0, exact 50.4%) — this is the expected test score.

## All-test progress
`rigel_test_all.csv`: 15.7k/484,838 at **116 img/s**, ~**17:20** ETA; 283k-critical block done ~**16:50**. Incremental flushes are valid; merge by id.

Tags: #model_selection #ensembling #insight #submission #inference
