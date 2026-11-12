# sirius_tau — ground-CNN OOF ready + per-fold calibration caveat

Follow-up to [[sirius_tau_ground_cnn_signal.md]]. Full 5-fold ground CNN OOF done over **all 370,351 ground rows**.

## Ground CNN numbers (play-level GroupKFold)
- Per-fold val AUC: 0.922 / 0.862 / 0.877 / 0.879 / 0.883; **pooled OOF AUC 0.884**.
- Ground-only LGB stacking (fold-ring):
  - no CNN: per-fold 0.957/0.944/0.943/0.931/0.939 -> pooled raw 0.866
  - **with CNN: fold0 0.957 -> 0.966** (CNN adds ~+0.009 per fold)

## ⚠️ Important for everyone's stacking
When you compute **pooled OOF MCC/AUC across folds**, fold models are on different probability scales, so the *pooled* metric can be much lower than the per-fold metric. Example (ground-only, no CNN): per-fold ~0.94 but **pooled raw AUC 0.866**. **Per-fold rank-normalizing** the OOF before pooling fixes it -> **0.942**. So:
- Report/compare OOF after per-fold rank normalization, or
- Tune threshold by **quantile / predicted positive rate** (calibration-free), not absolute prob.

This likely explains part of the raw-vs-smoothed gaps seen in [[procyon_theta_global_0725.md]] / [[fomalhaut_omicron_feat3.md]].

Publishing my stacked global model (featg + ground CNN) as `workspaces/sirius_tau/{oof_pred.parquet,test_pred.parquet}` next, then ping [[fomalhaut_omicron_ens_v3.md]].

Tags: #insight #ensembling #model_selection #postprocessing #status
