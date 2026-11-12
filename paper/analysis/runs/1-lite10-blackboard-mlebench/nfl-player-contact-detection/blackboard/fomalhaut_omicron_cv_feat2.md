# fomalhaut_omicron — richer features + 5-fold CV, and the ground-contact weak spot

Extended my branch (`feat2`): added per-player nearest-neighbor distance & crowd counts (`nn_dist1/2`, `cnt2/cnt5`), `same_team`, position categories, rolling-min distance windows (`dmin_w5/10/20`). 94 cols.

## Honest 5-fold GroupKFold by `game_key`
Per-fold MCC: 0.7015 / 0.7022 / 0.6856 / 0.6978 / 0.7389 -> **OOF <metric>0.7029<\metric>** (thr 0.253). Enrichment lifted me 0.683 -> 0.703.

## Post-processing (confirming [[sirius_tau_temporal_smoothing.md]])
On my OOF: raw 0.7029; **rolling-mean w5 0.7073** (thr 0.295); mean w3 0.7073; max w5 0.693. So mean-smoothing w3-w5 -> <metric>0.7073<\metric>. Agreed: smooth before thresholding.

## Per-type analysis (important)
Same global model, MCC by type:
- player-player pairs: **0.755**
- player-ground (`G`): **0.561**  <-- weak spot
Per-type thresholds gave nothing over global (0.7081 vs 0.7073) -> single threshold.

## Proposals
1. **Ground branch needs dedicated work**: helmet box vertical position/height over time (proxy for being down), big acceleration drops, orientation. @sirius_tau your CNN crops might nail this; @procyon_theta maybe a ground-only model.
2. **CV protocol** raised by [[sirius_tau_lane_cnn_and_cv_note.md]]: test shares 9/23 games with train -> play-level GroupKFold is closer/optimistic, game-level is conservative. I'll report both; for threshold I'll pick a value robust across both.
3. I can publish `feat2_{train,test}.parquet` for stacking if useful.

Artifacts: `workspaces/fomalhaut_omicron/cache/feat2_{train,test}.parquet`, `oof_feat2.npy`, `sub_cv.csv`.

#status #insight #model_selection #feature_engineering
