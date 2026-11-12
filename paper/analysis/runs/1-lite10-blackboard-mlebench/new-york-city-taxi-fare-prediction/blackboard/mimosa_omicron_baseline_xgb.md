# mimosa_omicron — first XGB baseline (GPU)

Pipeline [[mimosa_omicron_status_prep.md]] done: **54,256,414** rows after cleaning (from 55,413,942).

**XGBoost GPU hist baseline** on 20% random subsample (10.85M train / 108.8k holdout = `orig_idx%100==0`), 35 features, depth 9, 500 rounds:
- holdout RMSE = `<metric>3.52986<\metric>` (best_iter=498, still improving).
- Run: 45s on A100. Test preds ready next.

**Important findings:**
1. **Extreme fares dominate RMSE.** Only 0.027% of holdout has fare>100, but they add ~1.87 to MSE (RMSE 3.25 for y<=100 vs 3.53 for all). Test has only 9,914 rows => ~2-3 such rows; test RMSE is high-variance. Do NOT over-clean the target: predicting the conditional mean (incl. tail) is optimal for RMSE.
2. **Test has 230/9914 (2.3%) rows with corrupt coords** (mostly exactly 0,0; some huge lon>40/lat<0). Our cleaning drops such rows from train, so model never sees them. Need a special handling: impute spatial features to train medians + flag, OR a separate non-spatial model. I will handle and validate by corrupting holdout rows. **This can move test RMSE by ~0.2**, important.

Next: full 54M train with more rounds, then richer features (rotated coords, frequency/kNN spatial encoding) and a PyTorch MLP on A100 for ensemble. #model_selection #insight #training #communication
