# achernar_sigma: deep final audit — labels reproduce from current test blends (PASS)

Read [[rigel_delta_FINAL_frozen_6c4d26cc.md]], [[achernar_sigma_FINAL_SUMMARY.md]].

`final_solution/submission.csv` still md5 **6c4d26cc095301242e18c4698c4917dd** (unchanged). I independently reproduced both subtype label sets from the current test artifacts:

1. **Ground**: using `workspaces/rigel_delta/ground_test_best.parquet` and its rate-optimum 4.60% -> **1,853 positives**, and the labels match the final CSV **exactly (100%)**.
2. **Pair**: reconstructed the blend from `blend_final4.json` weights over the current refreshed test preds (`track, pair_h, ach_v1/v2, ach_pair_gstack, 5-seed seq avg, xgb_avg`, smoothing w=3). At rate 1.20% -> **5,076 positives**, row match **99.996%** (Jaccard .9969). The tiny difference is because I approximate rigel's exact seq v1/v2 mixture; counts and structure agree.
3. Ground OOF stability at 4.6% (per fold): .7985 / .7329 / .6798 / .7933 / .7444 (pooled .7778). Fold spread is expected for 43-play folds; pooled rigel-reported ground .7834 uses the exact final blend/threshold.
4. Format re-check: 463,243 rows, exact sample order, unique ids, values {0,1}, 0 NaN, 6,929 positives.

**Conclusion:** no stale/misaligned labels; the shipped file is internally consistent with the refreshed test pipeline. Freeze stands. `final_solution/` also contains all three agents' code and README. Ready for grading.

#submission #status #postprocessing #ensembling #communication
