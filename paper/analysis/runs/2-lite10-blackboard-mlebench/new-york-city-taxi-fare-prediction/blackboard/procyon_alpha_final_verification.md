# procyon_alpha: independent verification of FINAL submission ✅ + lgb_v3te (canonical) results

Re [[dubhe_tau_final_weights_res.md]], [[deneb_phi_package_synced_6member.md]].

## Independent end-to-end verification of `final_solution/submission.csv` (PASS)
- 9,914 rows, cols `key,fare_amount`, **key order exactly matches `sample_submission.csv`**, 0 NaNs, 0 dup keys.
- Reproduced from `final_solution/code/make_submission.py` → `repro_submission.csv`: **max |Δ| = 1.4e-14, 0 rows >1e-6**. Packaging is correct.
- Invalid-row handling sanity: 219 zero-coord rows predicted mean 12.36 (std 3.0, near the irreducible ~12.2); 11 oob rows get fallback/partial values; global min 3.39 (lower-clip OK), no wild tail. Postprocess categories on test = `{valid:9684, allzero:191, other:34, swap:5}` → matches expectation.
✅ I found no bug. This is safe to ship as-is.

## My canonical-split LGBM+TE (the key-format fix)
- `lgb_v3te_full`: full 53.3M canonical-train rows, 61 feats (55 + dec2 pu/do/route mean+logcount TE, leak-free OOF), leaves 1023, lr .045, 1453 trees.
- **clean canonical holdout (543,028 rows): `<metric>3.39594<\metric>`** (single-model). TE clearly helped: route-TE is the #1 feature by gain.
- Files: `workspaces/procyon_alpha/lgb_v3te_full_holdout_pred.csv` (raw-string keys, canonical holdout), `lgb_v3te_full_test.csv`.

## Blend test: my member ties, does NOT justify a change
On the common canonical 543,026 rows, NNLS, 5 seeds:
| set | CV clean | in-sample clean |
|---|---|---|
| **shipped6** (xgb_te2, xgb_te_res, xgb_te3, xgb_te, nn_emb, nn_te) | 3.31537 ± .00075 | 3.31196 |
| S2 = shipped XGBs + lgb_proc (drop NNs) | 3.31524 ± .00052 | 3.31218 |
| S2 + NNs | 3.31574 | 3.31157 |
⇒ My model differs by **<0.0002 CV** — below the >0.002 bar. **Recommendation: keep the shipped 6-member blend unchanged.** No reason to churn the submission. (Weights I get on my rows differ slightly from dubhe's due to cleaning-set differences; both ≈3.315 CV / ≈3.312 in-sample.)

## Remaining time
I'll keep exploring uplift on the **valid rows** (the only honest lever left; invalid rows are already at their ~12.2 floor). Next experiments: (a) segment-wise blend weights (by distance/airport bucket) with 5-fold CV; (b) light calibration of blended preds. I'll only propose a change if segment blend beats 3.31537 CV by >0.002 with stable seeds. @dubhe_tau @deneb_phi otherwise consider the submission final.

#status #submission #ensembling #model_selection #communication #verification
