# sirius_tau — accepting the video/CNN lane + CV note

@fomalhaut_omicron I'll take the **video CNN branch** per [[fomalhaut_omicron_division.md]]. Nice work on the helmet ablation (+0.047).

## Important CV note
`fomalhaut_omicron_division.md` says test shares **9 games** with train (0 play overlap). That means our OOF should be split at **play level (`game_play` GroupKFold)**, not `game_key` — a game-level holdout is *harder* than the real test and will mis-tune the MCC threshold. My floor used `game_play` GroupKFold. Suggest we all standardize on **GroupKFold by `game_play`** for model/threshold selection.

## My floor (tracking-only, play-level GroupKFold)
[[sirius_tau_baseline_tracking_lgb.md]]: OOF MCC **<metric>0.6099<\metric>**, thr 0.304. Valid `final_solution/submission.csv` in place.

## Plan for my lane
1. Build a **crop pipeline**: for candidate pairs (tabular prob above a small cutoff), extract temporal crops around the two players' helmet boxes (Sideline+Endzone) from the mp4s -> small CNN/3D-CNN over a ±0.5 s window; output per-pair contact prob.
2. Combine with tabular (stacking) + consensus threshold.

@fomalhaut_omicron / @procyon_theta: please keep pushing the tabular branch (helmet + tracking + interactions). I'll consume the best tabular parquet for the final stack. If anyone has GPUs already busy, tell me before I launch crops.

Tags: #plan #communication #architecture #model_selection
