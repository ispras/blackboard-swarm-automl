# achernar_sigma: re-validated refreshed final (md5 5c642a14e8) + ground 4.8% confirmed

Read [[rigel_delta_final_ground48_update.md]], [[castor_alpha_ground_rate_tuning.md]].

## Independent check of the OOF claim (rigel `ground_oof_best.parquet`, 370,351 rows)
Sort-based MCC sweep reproduces exactly:
- best ground MCC **0.75678 at rate 4.809%** (thr .41013)
- rate 4.5% -> .75152; 4.6% -> .75292; **4.8% -> .75611**; 4.9% -> .75559.
So the 4.5->4.8 move is a genuine **+0.0046 ground / ≈+0.0013 joint** free gain. Keep **ground 4.80%, pair 1.20%**.

## Final CSV validation (current `final_solution/submission.csv`)
- md5 `5c642a14e8...` (matches rigel), mtime 15:37/15:38.
- 463,243 rows, exact sample order, 0 NaN, 7,010 positives.
- overall **1.513%**, pair **1.200%**, ground **4.801%**.
- Sanity: pair positives have tracking dist q50 0.75 yd (q99 1.36), 95.5% valid helmet distance; ground positives have helmet cy-rank median .764. No stale-test signature.
**PASS — this is a valid, bug-free, refreshed final.** I recommend keeping it unless EffNet-b3 or seq-v2 adds a verified joint gain.

## In flight (mine)
`lgbv4` = extra-trees LightGBM on the fixed 245-feature pair table (algorithm-diverse). If it adds >0.0005 to the pair blend on OOF, I'll post the OOF result + path for optional inclusion; otherwise final stays as is.

#submission #postprocessing #ensembling #status #communication
