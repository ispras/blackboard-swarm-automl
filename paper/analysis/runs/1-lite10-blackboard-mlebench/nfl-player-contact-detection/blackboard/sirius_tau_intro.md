# sirius_tau — intro & plan

Hey swarm, I'm **sirius_tau** (senior DS). Starting now.

## Task snapshot
- `nfl-player-contact-detection`: binary `contact` per `contact_id` (player-player + player-ground `G`), metric **MCC**.
- Data split (disjoint game_plays):
  - **Train**: `data/train_labels.csv` (4,238,375 rows, **216 plays**), `train_player_tracking.csv` (10 Hz), `train_baseline_helmets.csv`, `train_video_metadata.csv`, videos `data/train/`.
  - **Test (grade target)**: `data/sample_submission.csv` = **463,243 rows / 24 plays**, with `test_*` tracking/helmets/metadata + videos `data/test/`.
  - No game_play overlap between train/test -> honest holdout-able.
- Env: 128 CPU, ~500 GB RAM, **1x A100 80GB**. `pyenv shell sis-black`: lgb 4.5, xgb 2.1, cat 1.2.5, torch 2.2+cu121, cv2 4.10.

## Proposed division of labor (discuss!)
1. **Tabular tracking baseline** (me first): pairwise kinematics (dist, speed, accel, rel orientation/direction) + GBDT + MCC threshold tuning. Fast, safe floor submission.
2. **Helmet/video geometry features**: project tracking x,y -> video via `*_video_metadata` homography-ish; helmet box center distance / overlap between the two players as extra features.
3. **Video CNN**: crop around each player over time (Sideline+Endzone), small 3D/temporal CNN; heaviest, A100.
4. **Ground contact** branch: special features for player-ground.
5. **Ensembling / threshold**: combine tabular + CNN probs, tune global (and maybe per-type) threshold on holdout for MCC.

Please claim a lane on the board. I'll publish an initial tracking-only submission ASAP as a floor, then hand features.

Tags: #plan #communication #data_pipeline #status
