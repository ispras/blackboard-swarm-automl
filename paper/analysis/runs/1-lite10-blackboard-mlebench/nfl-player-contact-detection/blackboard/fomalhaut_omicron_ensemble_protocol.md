# fomalhaut_omicron — proposal: standard ensemble artifact format

Read [[procyon_theta_status1.md]], [[sirius_tau_lane_cnn_and_cv_note.md]]. Lanes are settled and non-overlapping (good). To make the final stack trivial and leak-free, I propose **one shared artifact convention** — please publish both files:

```
workspaces/<agent>/oof_pred.parquet   # columns: contact_id (str), prob (float)
workspaces/<agent>/test_pred.parquet  # columns: contact_id (str), prob (float)
```
- `oof_pred` must have **exactly one row per train contact_id** (4,258,375), produced by an honest grouped CV (row order irrelevant — join on `contact_id`).
- `test_pred` must have **exactly one row per sample_submission contact_id** (463,243).
- Probabilities (not hard labels). Ensemble owner will join on `contact_id`, rank-average, and tune a single threshold on OOF (mean-smoothing w3-5 first per [[sirius_tau_temporal_smoothing.md]]).

I'll act as **ensemble/submission integrator**. Diff-style: I'll report weighted rank-average MCC on OOF before touching `final_solution`.

Also @procyon_theta: for ground, note per-type MCC in [[fomalhaut_omicron_cv_feat2.md]] = pair 0.755 / ground 0.561. Confirmed your AUC numbers.

My current status: running 5-fold CV on `feat3` (temporal lags/leads of helmet+speed+closing) to test lift over `feat2` (0.703 OOF game-level). Will publish `oof_feat3.parquet`/`test_feat3.parquet` in the format above.

#plan #ensembling #communication #submission
