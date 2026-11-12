# regulus_iota — cascade analysis: low-confidence 20% carries ~72% of total Levenshtein; routing it to MolScribe could ~halve the metric

Refs: [[regulus_iota_6k_lev12_confirm_endgame.md]] [[altair_theta_tta_and_endgame.md]] [[regulus_iota_critical_predict_bug_fix.md]]

## Measurements on shared val (first 1000 ids), my step-6000 ckpt (`ckpt_main/ckpt_last.pt`)
- overall mean Lev **<metric>11.19<\metric>**, exact **71.2%**, invalid SMILES rate **5.6%**.
- valid-only mean Lev **6.46**; invalid-only mean Lev **90.8** (fallback is bad for these long/hard molecules).
- Errors concentrate at long InChIs: lev 13.9 for len 140-170 (8% invalid), **45.4 for len 170-250 (19% invalid)**.

### Sorting val by my model confidence (mean token logprob, ascending=worst)
| worst-conf fraction | mean Lev of that slice | mean Lev of the rest |
|---|---|---|
| 5% | 72.96 | 7.94 |
| 10% | 57.02 | 6.09 |
| **20%** | **40.30** | **3.91** |
| 30% | 31.45 | 2.50 |
| 50% | 21.44 | 0.93 |

So ~**72% of the total loss sits in the worst-confidence 20%**. If a second model achieved Lev 10-20 on that slice, overall would become **~5-7 instead of 11.2**. This is the single biggest remaining gain, much bigger than squeezing the base model.

## Recommendation for the endgame (concrete)
1. My full-test pass (altair runner) emits `conf` + `valid`. **Route the worst ~20-30% by confidence** (tune the exact fraction/threshold on val; include all invalid) to a **cascade**:
   - MolScribe corrected-transform (altair/omega) on those ids, and/or omega's fine-tune, and rot180-TTA of my model.
   - Per-image selection by calibrated confidence; tune source offsets/thresholds on val.
2. Even with a mediocre second model (Lev ~20 on that slice), expected overall ~7, i.e. a ~4-point gain over 11.2.
3. Keep my predictions for the confident 70-80% (top-50% slice Lev 0.93 — don't touch it).

## Ask
- @altair_theta: before/while the test pass, please produce **MolScribe (corrected) predictions + confidence for a val subset (>=1000, ideally the same first 1000 ids and/or a random 2000)** -> `workspaces/altair_theta/preds_zeroshot_val.csv`. That lets us tune the cascade *before* committing GPU to MolScribe on test. `val_20k.csv` first 1000 are the ids in `workspaces/regulus_iota/val_preds_6k.csv`.
- @regulus_omega: same for your fine-tune val preds (`image_id,inchi,conf,valid`). Your model may be complementary on the hard/long slice where mine fails.
- Cost estimate for routing: 20% of test = 97k images; MolScribe at ~0.07 s/img exclusive = **~1.9 h**. 30% = 2.9 h. Fits 13:00-15:45 if my pass ends ~12:45.

#insight #ensembling #inference #model_selection #plan #communication
