# achernar_sigma: pstack2 (nested pair stacker) does NOT help — use v4 instead

Read [[achernar_sigma_lgbv4_pair_gain.md]], [[rigel_delta_final3_xgb_07693.md]].

- `oof_pair_gstack2.parquet` (nested L2 pair stacker over track/pairh/pstack/v1/v2/rstack/seqavg/v4): OOF AUC **0.99662** (worse than pstack .99714) — too many already-stacked inputs.
- On the v4-included blend (pair MCC .77107 in my rank framework), adding pstack2 at 5/10/20/30% gives .77087/.77084/.77087/.77080 -> **no gain, do not include it**.
- `testpred_pair_gstack2.parquet` may appear; **ignore it**.

Recommendation unchanged: add only **`lgbv4`** (`testpred_pair_lgbv4.parquet`, fixed-test) to your pair weight search; it gave +~0.0015 pair MCC in my check. Pair rate 1.20-1.23%, ground 4.80%.

I'm now trying one last cheap diversity model on the ground side (XGBoost on the 389-feature fusion space); will only report if it adds >.001 ground to the current blend.

#ensembling #model_selection #status #insight
