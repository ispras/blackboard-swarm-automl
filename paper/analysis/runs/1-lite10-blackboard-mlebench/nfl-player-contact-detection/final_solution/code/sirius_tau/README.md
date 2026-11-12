# sirius_tau — video (ground) CNN branch + stacked model

Task: 1st & Future NFL Player Contact Detection (binary `contact`, metric MCC).

## Pipeline
1. `build_features.py` / `train_lgb.py` — tracking-only baseline (floor).
2. `build_crops_ground_all.py` — extract temporal crops around each player's helmet (Endzone+Sideline, T=7 frames, 64²) for **all** ground rows. Helmet boxes from `*_baseline_helmets.csv`; frame mapping `frame = round((snap_time-start_time)*59.94 + 5.994*step)`. Test crops: `build_crops_ground_test.py`.
3. `train_cnn_ground_all.py` — 5-fold CNN (play-level GroupKFold) over the crops; saves OOF for **all** train ground rows + fold models; predicts test.
4. `export_cnn_feature.py` — publish `ground_cnn_feat_{train,test}.parquet` (contact_id, logit) for reuse.
5. `train_final.py` — LightGBM on `featg_all` ([[procyon_theta_global_0725]]) + the CNN OOF feature; w7 temporal smoothing; threshold chosen by OOF-optimal positive-rate quantile (calibration-free). Produces `oof_pred.parquet`, `test_pred.parquet`, `sub_final.csv`.

## Result (own, play-level OOF)
- Ground-contact CNN OOF AUC 0.884 (candidate) / ground-only stacking lift fold0 0.957 -> 0.966.
- Stacked global model: smoothed OOF MCC **0.7053**; ground AUC **0.9552** (+0.016 from video).

Full result lives in the swarm ensemble (see `../README.md`).
