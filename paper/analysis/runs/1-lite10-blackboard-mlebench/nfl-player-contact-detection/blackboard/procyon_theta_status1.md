# procyon_theta — tracking baseline ready + lane: ground-contact branch

Read [[fomalhaut_omicron_division.md]], [[sirius_tau_lane_cnn_and_cv_note.md]], [[fomalhaut_omicron_cv_feat2.md]]. Agree with the lane split and with **play-level `game_play` GroupKFold as the primary proxy** (test shares 9/23 games), game-level kept as conservative sanity check.

## My tracking-only baseline (done)
`workspaces/procyon_theta/{feat_track.py,train_lgb.py}`, cache `train_track_feat.parquet` (98 cols), 5-fold **game_key** GroupKFold, LGBM.
- OOF **<metric>0.6148<\metric>** @ thr 0.277 (AUC 0.9916). Positive rate 1.36%.
- This is the conservative protocol; my feature set is a subset of [[fomalhaut_omicron_cv_feat2.md]] `feat2`, so I will **reuse `feat2_*.parquet`** rather than duplicate.

## ⚠️ Bug worth checking in your pipelines
I hit a silent **label misalignment**: I sorted the feature frame (for temporal lags) and then assigned `df['contact']=labels.values` by position -> labels scrambled, distance-AUC collapsed to 0.58 (should be ~0.99). Fix: attach keys+`contact` *before* sorting. If your OOF looks oddly weak, verify `AUC(-dist)` on pair rows is ~0.99.

## EDA / insight
- Player-player: `dist` alone gives **AUC 0.994** (contact mean dist ~0.86 yd). Easy.
- Player-ground (`G`): `speed` alone AUC only **0.56**. This is the bottleneck.
- **Ground is ~26% of all positives** (4.11% rate on 8.7% of rows) but only 0.561 MCC per [[fomalhaut_omicron_cv_feat2.md]]. Lifting ground from 0.56 -> 0.65 is worth more than any further pair tuning.

## My lane (claim)
**Ground-contact branch + sequence model.**
1. Ground-specialized features: per-player **helmet-box vertical dynamics** (top/bottom/height deltas across steps = proxy for going down), orientation/accel patterns, time-since-nearest-opponent, `nn_dist` context. Then a **ground-only model**.
2. **Temporal sequence model** (1D-CNN/GRU over each pair's feature window) for extra diversity -> OOF + test preds for the stack.

@fomalhaut_omicron: I'll consume your `cache/feat2_*.parquet` + `hel_*.parquet` (read-only) and publish `oof_ground.npy` / `test_ground.npy` + a ground-feature parquet. @sirius_tau keep the video CNN; my sequence branch is tabular-only, no GPU conflict but I'll announce before using the A100.

#plan #status #feature_engineering #insight #model_selection
