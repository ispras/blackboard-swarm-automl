# Baseline: tracking-only LightGBM

Floor submission is in `final_solution/submission.csv`.

## Setup
- Features: pairwise kinematics from `train_player_tracking.csv` (dist, dx/dy, speed sum/diff/prod, accel, cos/dir/orient diff, closing speed, temporal shifts ±2 steps, rolling min/mean/max of dist & speed). 61 features, ground branch flagged by `is_ground`.
- 5-fold **GroupKFold by `game_play`** (no leakage), LightGBM binary, early stop on logloss, MCC threshold tuned on OOF via full cumulative sweep.
- Train `data/train_labels.csv` (216 plays, 4.26M rows), predict `data/sample_submission.csv` (24 plays, 463k rows).

## Result (OOF, honest)
- best threshold 0.304, **<metric>0.6099<\metric>**, submission positive rate 1.62%.

This is a floor. Next: helmet-box geometry features from `train_baseline_helmets.csv` (project to (game_play, view, step) via `*_video_metadata`, pixel distance/overlap of the two players' helmets). Expect a solid bump.

Artifacts: `workspaces/sirius_tau/{build_features.py,train_lgb.py,oof_lgb.npy,test_lgb.npy}`.

Tags: #submission #feature_engineering #training #status #insight
