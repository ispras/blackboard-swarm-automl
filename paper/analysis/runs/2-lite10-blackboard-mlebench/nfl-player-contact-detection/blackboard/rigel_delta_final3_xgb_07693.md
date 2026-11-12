# rigel_delta: final refresh #3 — add XGBoost pair model -> OOF joint 0.76933

Read [[rigel_delta_final_ground48_update.md]].

New independent pair model: **XGBoost** on the same tracking+helmet feature space (5-fold, hist, depth 8, eta .05, early stop; fold AUCs .9969/.9978/.9984/.9983/.9970, OOF AUC **.99757**). It is genuinely different from the LightGBM family and adds diversity.

Re-ran the 7-model pair weight search on unchanged OOF:
- weights: track .004, pair_h .167, ach_v1 .020, ach_v2 .015, ach_pair_gstack .385, rigel seq .251, **rigel XGB .160**
- Pair MCC **<metric>0.7730<\metric>** (was .7725); ground MCC .7568 unchanged; **JOINT OOF <metric>0.76933<\metric>** (was .76908), at same pair/ground rate targets 1.20% / 4.80%.

`final_solution/submission.csv` updated: md5 `efcfbd00…`, mtime 15:5x; 463,243 rows, exact order, 0 NaN, 7,010 positives (overall 1.513%, pair 1.200%, ground 4.801%). README/code updated.

@achernar_sigma please re-validate md5 `efcfbd00…`. Next and last: fold in castor EffNet-b3 ground (training) and achernar lgbv4 pair if their OOF+test arrive and clear +0.001 joint. Otherwise this is the final.

#submission #ensembling #model_selection #status
