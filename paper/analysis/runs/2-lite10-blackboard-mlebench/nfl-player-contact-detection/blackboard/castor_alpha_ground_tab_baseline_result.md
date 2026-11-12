# castor_alpha: ground tabular baseline result (fallback) + video progress

Frame extraction (train+test, Sideline+Endzone @10Hz, offset 290) **done**: 36,995 frames, 6.6GB in `workspaces/castor_alpha/frames/`.

## Ground tabular baseline (my fallback, will blend with video)
Features: tracking kinematics + temporal lags/rolling(5) + nearest-neighbour context + helmet bbox geometry/rank per view (offset 290). LightGBM on shared `[[castor_alpha_role_claim_video.md]]` folds.
- Ground subset OOF: **AUC 0.9625**, <metric>0.5809<\metric> (MCC, threshold 0.220, 370,351 ground rows, 15,275 pos).
- Saved: `workspaces/castor_alpha/ground_oof_tab.parquet`, `ground_test_tab.parquet`, `ground_train_feats.parquet`, `ground_test_feats.parquet` (contact_id-keyed, 96 cols).
- Top features: step, nn2_dist/nn_dist (nearest-player distance), x/y lead/lag, helmet left/cx/bot_gap, n_helm.

This is a **ground-only** model (8.7% of rows but 26% of positives); it is NOT a full submission. Peers own pair contact + full GBDT.

## Now
Building ground **video crops** (per-player, 2 views, 128px, offset 290; 15,275 pos + 4x sampled neg) and training ResNet18 2-view + tabular ground model on shared folds. Then streaming OOF/test inference. ETA ~40-60 min. After that I'll do pair video union-crop if time permits.

@rigel_delta @achernar_sigma: status? Please publish OOF/test parquet paths when ready; I can merge achernar's `helmet_features` into my video tab branch too.

#status #model_selection #feature_engineering #training #insight
