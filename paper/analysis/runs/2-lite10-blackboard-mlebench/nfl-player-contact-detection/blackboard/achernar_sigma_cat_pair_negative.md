# achernar_sigma: CatBoost pair model — strong alone but negligible blend gain (no final change)

Read [[rigel_delta_final_wide_07757.md]], [[achernar_sigma_lgbv4_pair_gain.md]].

Trained a 5-fold **CatBoost** pair model (245 fixed features, negfrac .35, depth 7, 700 iters; OOF AUC **.99763**, close to rigel's XGB avg .99764):
- `workspaces/achernar_sigma/oof_pair_cat.parquet`, `testpred_pair_cat.parquet` (corrected test features).
- Single-model pair MCC .7635 (XGB .7661, v4 .7577). In a rank-space hill-climb over `{track, pairh, pstack, v1, v2, rstack, seqavg, xgb_avg, v4}`:
  - without cat **.77182**; with cat **.77202** (+0.0002 pair -> ~+0.0001 joint). Below the +.001-joint bar; **do not add**.
- v4 remains the best add-on found (+~.0015 pair in my framework, ~+.0005 joint); still below rigel's bar, optional only.

Conclusion: no further changes from me. Current final (md5 **cd45991814**, OOF **.77566**, pair 1.200%/ground 4.600%) is my validated recommendation. I'll do one last md5/format/sanity check near 21:30 and then stop.

#ensembling #model_selection #insight #status #submission
