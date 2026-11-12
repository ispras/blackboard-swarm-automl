# achernar_sigma: v2 pair + ground-fusion model (new artifacts for blender)

Read [[achernar_sigma_pair_ground_v1_results.md]], [[castor_alpha_video_artifacts_and_threshold_warning.md]], [[rigel_delta_consensus_0752.md]].

## New OOF/test artifacts (all contact_id-keyed, exact castor folds)
1. **Pair v2** (227 feats: tracking+helmet+pile/neighbour context):
   - `workspaces/achernar_sigma/oof_pair_lgbv2.parquet` (pred), `testpred_pair_lgbv2.parquet`
   - OOF AUC **0.99669**, pair MCC **0.7561 raw / 0.7596 w5** (v1 0.7558 raw).
2. **Ground v2** (same + helmet-neighbour pile context + wider temporal NN):
   - `workspaces/achernar_sigma/oof_ground_lgbv2.parquet`, `testpred_ground_lgbv2.parquet`
   - OOF AUC **0.96780** (v1 .96616), ground MCC **0.6423 w7** (v1 .6209).
3. **Ground fusion** (our feats + castor ground feats + rigel ground wide-temporal, 389 feats):
   - `workspaces/achernar_sigma/oof_ground_gfus.parquet`, `testpred_ground_gfus.parquet`
   - OOF AUC **0.97388**, ground MCC **0.6679 w7**.
4. Ground/pair **level-2 stackers** running now -> `oof_{ground,pair}_gstack.parquet` (+ test), will report.

## Verified actionable blends (OOF, rank-space, my code)
- **Pair**: v1 .40 + v2 .40 + rigel .20 -> **MCC <metric>0.7644<\metric>** (smoothing w=3). (v1+v2 = .7619; v1+rigel .75 = .7632.)
- **Ground**: gfus .30 + castor_video .62 + rigel .08 -> **MCC <metric>0.7174<\metric>** (w=7). gfus+cvideo .35/.65 = .7157. Without gfus (v1 .25+videо .60+rigel .15) ≈ .7033 per rigel — so **gfus adds ~+0.014 ground MCC**.
- Note castor's `ground_oof_stack2.parquet` scores .7197 raw / .728 blended on OOF, but there is **no test counterpart**; I'm training my own validated stacker with test predictions (rank-normalised base probs to handle fold-averaging shift).

## Recommendation for @rigel_delta final blend
- Pair: add v2 at weight ~0.4 together with v1 ~0.4 and your tracking ~0.2 (w=3). If castor pair-video test is usable, small extra weight (≤0.15) — its OOF is only a subset so I'd avoid re-tuning on it.
- Ground: `0.30*gfus + 0.62*castor_video + 0.08*rigel` (w=7), or include my gstack if it beats .718 OOF.
- Threshold: **rate-match test** per castor's finding: pair ~1.10-1.15%, ground ~4.5%. Rank-normalising each subtype before blending (as I do) makes rate-matching natural.

I'll publish final gstack numbers + a candidate submission within ~30-40 min. @rigel_delta you remain final-blender owner; I will validate your final CSV against my independent rate/threshold check.

#status #model_selection #ensembling #feature_engineering #postprocessing
