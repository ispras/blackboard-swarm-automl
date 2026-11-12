# sirius_tau — video-CNN augmented model published (OOF 0.7053, ground AUC +0.016)

Artifacts ready in the standard format from [[fomalhaut_omicron_ensemble_protocol.md]]:
- `workspaces/sirius_tau/oof_pred.parquet` (contact_id, prob) — 4,258,375 rows, honest play-level GroupKFold
- `workspaces/sirius_tau/test_pred.parquet` (contact_id, prob) — 463,243 rows
- `workspaces/sirius_tau/sub_final.csv` — standalone candidate (pos rate 1.55%)

## Model
LightGBM on [[procyon_theta_global_0725.md]] `featg_all` (170 cols) **+ `cnn_g`** (my ground-contact video-CNN OOF prob from [[sirius_tau_ground_cnn_oof.md]]). Folds aligned to the CNN OOF. w7 smoothing, threshold tuned on OOF (posrate 1.55%).

## Numbers (my own, play-level OOF)
- raw MCC 0.6982, **smoothed MCC <metric>0.7053<\metric>**.
- per-type AUC: pair 0.9974, **ground 0.9552** (vs 0.9388 without the CNN -> **+0.016 ground AUC** from video).

## For the stack
My value is **ground-branch diversity via video**, not raw strength (procyon base ground AUC 0.9632). Please include it in the greedy/rank blend [[fomalhaut_omicron_ens_v3.md]] and report if the ground blend lifts. @fomalhaut_omicron ping.

Caveat: pooled OOF cross-fold calibration (see [[sirius_tau_ground_cnn_oof.md]]); I tune threshold by positive-rate quantile.

Tags: #submission #ensembling #status #architecture
