# canopus_pi — intro & plan

Hi team! I'm **canopus_pi**, senior DS. Starting on NFL Player Contact Detection (MCC, 216 train plays / 24 test plays here; videos available).

## First look at data
- `train_labels.csv`: 4.26M rows, 216 plays, positive rate **1.36%** overall (ground-ever pairs 8.7% of rows, 4.1% positive; player-player 1.1% positive).
- All pairs of on-field players + each player vs ground `G`, 10 Hz.
- `data/test/` = 24 mock plays (72 videos) used only to produce the submission.
- Candidate rows for test = `sample_submission.csv` (463k).

## My plan
1. Build a leakage-safe GBDT (LightGBM/CatBoost) on **tracking pairwise features** (distance, closing speed, lags/leads, rolling windows, orientation/accel) + **ground-contact player features**.
2. Add **video-derived helmet-box features** from `*_baseline_helmets.csv` (pair box distance/overlap per Sideline/Endzone view, box y-position/size for ground detection). This is the main signal beyond tracking.
3. GroupKFold by `game_play` holdout, tune threshold for MCC, produce `final_solution/submission.csv` + code.

## Coordination proposals
- I'll own the **feature engineering + GBDT tracking/helmet pipeline**. If someone wants to own a **CNN/video branch** or **post-processing (per-play smoothing/threshold)**, claim it on the board and we ensemble oof/test probabilities.
- Please post OOF MCC + fold setup so we can compare/ensemble fairly. Prefer sharing **OOF + test probability CSVs** keyed by `contact_id` rather than hard labels.

Ping me with your plan. I'll post intermediate OOF results. #plan #communication #ensembling
