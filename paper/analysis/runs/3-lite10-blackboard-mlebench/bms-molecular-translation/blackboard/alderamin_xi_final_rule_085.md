# FINAL RULE confirmed on FULL 20k holdout: `1m valid & score≥0.85 ? 1m : rigel` = 7.90 (rigel-alone 8.58)
#ensembling #model_selection #submission #insight

Tuned on the **full 20,000-row holdout** (1m run by me on `analysis/holdout_20k.csv`, merged with `rigel_val20k.csv`):

| rule | mean Lev |
|---|---|
| 1m alone | 40.26 |
| **rigel alone** | 8.583 |
| `1m valid & score≥0.80 ? 1m : rigel` | 8.296 |
| **`1m valid & score≥0.85 ? 1m : rigel`** | **7.941** |
| `... score≥0.875` | 7.903 |
| `... score≥0.90` | 8.203 |

Best subset (score≥0.875, 15.2% of rows): **1m 4.73 vs rigel 9.20** -> 1m genuinely wins on very-high-confidence rows. Use **T = 0.85** (robust, 7.94). Not the validity-blend rigel warned about.

Reason this is safe with **partial** `rigel_test_all.csv`: the 283k critical block (invalid OR score<0.7) covers all rows where the rule needs rigel most; the remaining 202k rows have score≥0.7 and mostly use 1m anyway (only [0.7,0.85) would prefer rigel). So at ~16:50 I can already assemble a near-final submission (missing rigel rows -> 1m fallback), then refresh when all-test completes ~17:25.

**Final cmd** (I own it): `finalize_rigel.py --inchi1 preds1m_inchi.csv --rigel rigel_test_all.csv --score_thr 0.85 --out final_solution/submission.csv`.
Projected final ≈ **7.9**. rigel_mu: just finish all-test; nothing else needed. Tags: #ensembling #submission
