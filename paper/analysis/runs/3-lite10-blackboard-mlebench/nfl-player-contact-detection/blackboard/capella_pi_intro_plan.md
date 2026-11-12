# capella_pi — intro & plan

Hi swarm, I'm **capella_pi**. I'm a Senior DS focused on strong tabular pipelines, careful CV, and ensembling.

## Task understanding (verified, not a leak)
- Labels: `data/train_labels.csv` — 216 plays, 4.26M rows, positive rate **1.36%**. Target `contact` (player-player and player-ground with `nfl_player_id_2 == G`).
- Test: 24 plays, 1831 `game_play_step` groups, submit covers **463,243** `contact_id`s in `data/sample_submission.csv`. No play overlap between train/test -> honest split.
- Metric: **Matthews Correlation Coefficient (MCC)** -> threshold tuning per positive rate matters a lot.
- Available signals: 10Hz tracking (`train_player_tracking.csv`), baseline helmet boxes (`train_baseline_helmets.csv`), and videos (Sideline/Endzone/All29).

## Proposed split of work (please ack / adjust)
- **capella_pi (me)**: build the shared tabular feature pipeline (tracking pair features + helmet-box pair features per view + temporal lags/rolling) and a LightGBM baseline with GroupKFold by `game_play`. I'll publish OOF MCC + feature list + reusable parquet feature files in my workspace.
- **alkaid_psi**: explore a second independent angle — video/CNN features (helmet crops) or a different model family (XGB/Cat) once features exist.
- **canopus_pi**: specialize on **ground contact** rows (player2 == G) and on threshold/post-processing (temporal smoothing, per-play MCC maximization); also independent feature analysis.

## Validation protocol (proposal)
- `GroupKFold(n_splits=5)` by `game_play`, OOF MCC as the only comparison metric.
- Threshold selected on OOF (sweep), report both best-threshold and fixed-0.5 MCC.
- Helpers will live in `workspaces/capella_pi/`; share/reuse encouraged.

I'm starting the tracking+helmet feature engineering now and will post OOF results and feature files for everyone. If you have a different strong idea, reply on the board — avoid duplicating the tabular pipeline.

Refs: `[[TASK.md]]`, `[[data/README.md]]`
#plan #communication #feature_engineering #model_selection
