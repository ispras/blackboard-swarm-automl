# canopus_pi branch — code, models, licenses

All models in this branch are trained **only** on `data/train_*` (5-fold GroupKFold by
`game_play`, fold map `workspaces/alkaid_psi/cv_folds.csv`). No external pretrained
checkpoints, no competition-specific code/data, no test labels.

## Pipeline
1. `build_features.py` — candidate table from `train_labels.csv` + `sample_submission.csv`;
   per-player tracking features (lags/leads/rollings), pair geometry + temporal distance,
   and helmet-box features from `*_baseline_helmets.csv` mapped by
   `frame ≈ 300 + 5.994*step` (per view/step/player boxes, pair pixel distances/IoU,
   y-position/size "on-ground" cues). Writes `features/all_features.parquet`.
2. `augment_features.py` — per-player nearest-neighbour context (mutual NN, opponent/team
   distances, counts within 2/5 yd, temporal deltas), pair third-player distance,
   time-to-collision, helmet temporal deltas; `ground_extras.py` — low-speed run lengths,
   centered windows, cross-sectional ranks.
3. `train_gbm_v2.py` — LightGBM (per-task: player-player with 12% negative subsampling;
   ground with all rows), early stopping on the held-out fold.
4. `train_xgb.py` — XGBoost `hist` on the same features (family diversity).
5. `train_ground_gru.py` / `train_pp_gru.py` — bidirectional GRU over each candidate's
   sequence within a play (per player-ground and per player-pair), optional 2 layers;
   inputs are tracking + helmet + neighbour features (PP also uses video-CNN probabilities
   produced by alkaid_psi). This is the largest single gain, especially for ground
   (GBDT 0.67 -> GRU 0.73-0.736 MCC).
6. `ground_eval3.py` / `greedy_ens.py` — per-task logit-space weighting of base models +
   temporal smoothing + joint 2-D MCC threshold sweep (hedge ensembles; not the shipped
   stacker). `make_ratematch_sub.py` / `alkaid_ratematch.py` — quantile (positive-rate
   matched) threshold experiments.

## Reported OOF (alkaid folds, full train)
- best ground GRU (v5, 2-layer): MCC 0.7365, AUC 0.98324
- best PP GRU (v1/v4): MCC ~0.760-0.764, AUC ~0.9975
- hand-blend hedge `ens v4/v5`: global OOF ~0.777-0.780
These base predictions feed the swarm-level stacker/blend that produced the shipped
`submission.csv`.
