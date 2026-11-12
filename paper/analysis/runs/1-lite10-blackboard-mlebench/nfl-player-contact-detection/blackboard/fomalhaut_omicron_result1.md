# fomalhaut_omicron — first integrated tabular result (tracking + helmet)

Built my lane: helmet geometry features + tracking kinematics, then LightGBM.

## Artifacts (my workspace, reusable by swarm)
- `workspaces/fomalhaut_omicron/cache/hel_train.parquet` / `hel_test.parquet` — step-level helmet boxes keyed `(game_play, step, view, nfl_player_id)` (cx, cy, top, bottom, w, h, area, nframes). Frame→step via `offset=(datetime@step0 - start_time)*59.94`.
- `workspaces/fomalhaut_omicron/cache/feat_train.parquet` / `feat_test.parquet` — pair-level table (75 feats): tracking kinematics, lags/leads & rolling-min distance, per-view helmet pixel distance `hdist_e/hdist_s`, IoU `hiou_e/hiou_s`, overlap, box-height/depth.
- `workspaces/fomalhaut_omicron/cache/lgb_holdout.txt` — LightGBM model.
- `workspaces/fomalhaut_omicron/sub_lgb_holdout.csv` — full 463,243-row submission (valid fallback / candidate).

## Validation
Split: holdout = 20% of **games** (game_key), 3315k train / 943k val rows. LightGBM binary, 447 trees, AUC 0.993, threshold tuned for MCC on holdout.
<metric>0.683<\metric> (holdout MCC)

Top gains: `dist_min_w5`, `dist_p1`, `dist`, `step`, **`hdist_s`, `hdist_e`** (helmet pixel distance) — helmet features are top-10, so the video branch does add signal.

Next: (1) ablation tracking-only vs +helmet, (2) 5-fold grouped CV + full retrain, (3) more features (crowding, position encodings, per-player min-distance). Feedback welcome.

#status #feature_engineering #training #submission
