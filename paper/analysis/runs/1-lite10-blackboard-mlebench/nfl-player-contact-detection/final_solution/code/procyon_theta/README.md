# procyon_theta — solution components

Task: NFL player contact detection (binary, MCC). Play-level 5-fold GroupKFold by `game_play`.

## Pipeline (my contribution)
1. `feat_track.py` — initial tracking-only feature builder (baseline).
2. `feat_ground.py` — augments fomalhaut's `feat2` with temporal deltas/leads/rolling of
   helmet-box geometry and kinematics -> `cache/featg_all_{train,test}.parquet`.
3. `build_relpose.py` — **novel relative helmet-pose features**: per (play,step,view) z-score/rank
   of each player's helmet `cy/top/height/bottom` vs all players in that frame -> `cache/relpose_*`.
4. `add_relpose.py` — joins relpose for p1 & p2 -> `cache/featg_rel_all_*`.
5. `train_global.py` — LightGBM base model (negative downsampling for speed); `gall`, `gallr`.
6. `level2.py` — **level-2 stacker** with cross-pair context (max/mean base prob over all opponent
   pairs involving a player) + relative-pose features + sirius ground-CNN prob -> `oof_pred_l2.parquet`.
7. `seq_cnn.py`, `seq_gru.py` — 1D-CNN / GRU over per-pair temporal feature windows (extra diversity).
8. `train_cb.py` — CatBoost variant (algorithmic diversity).
9. `ens_greedy.py` — greedy rank-average over all `oof_pred*.parquet` (global weights).
10. `ens_pertype.py` — **final method**: separate greedy rank-ensembles for ground vs pair rows,
    combined on the global rank scale with a single threshold -> `sub_pertype.csv`.

## Reproduce the final submission
Requires the published OOF/test artifacts from all agents under `workspaces/*/oof_pred*.parquet`
and `test_pred*.parquet` plus my cached feature parquets, then:
```
pyenv shell sis-black && python ens_pertype.py
```
Final chosen file: `final_solution/submission.csv` (per-type ensemble incl. feat5/feat5_d2, OOF MCC 0.7516 by fomalhaut's integrator).
