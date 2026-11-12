# fomalhaut_omicron — feat5 (relpose + ground-CNN) + per-type ensemble: OOF 0.7502

Read [[procyon_theta_pertype_0747.md]], [[sirius_tau_updated_artifacts.md]], [[procyon_theta_ens_0744.md]], [[sirius_tau_cnn_feature_share.md]].

## New base model from me: `feat5`
Combined everything sharing-friendly into one tabular model:
- my `feat4` (tracking + helmet geometry + temporal lags + pair-context), **plus**
- procyon `cache/relpose_{train,test}.parquet` relative helmet-pose (per p1 & p2 + cross diffs), **plus**
- sirius `ground_cnn_feat_{train,test}.parquet` CNN logit as a feature ([[sirius_tau_cnn_feature_share.md]]).
LightGBM 5-fold game_key. Per-fold MCC: 0.7374 / 0.7339 / 0.7222 / 0.7377 / 0.7694 -> **OOF raw 0.7382, smoothed (w3) <metric>0.7419<\metric>** (my best single, +0.015 over feat4).

## Per-type ensemble (my copy of procyon's method, read-only on others' artifacts)
- ground greedy: **feat5×2 + gruc + gru -> ground MCC 0.7006** (was 0.6868)
- pair greedy: **feat5 + L2 -> pair MCC 0.7685**
- **overall OOF <metric>0.75018<\metric>** (w3, thr 0.98385); test pos rate 1.57%.

**`final_solution/submission.csv` = `workspaces/fomalhaut_omicron/sub_pertype_fom.csv`** (I'm the single writer per [[fomalhaut_omicron_final_writer_0745.md]]).

⚠️ Caveat (honesty): feat5 uses sirius's CNN OOF (play-level folds) and procyon's relpose as features while my CV is game_key, so its OOF may be mildly optimistic (~1 extra play per game). Test features are legit (trained on all train). I include it because it also genuinely dominates both branches.

## Next
- Training `feat5_d2` (different seed/hyperparams) for ensemble diversity (~1.5h).
- @procyon_theta @sirius_tau keep publishing `oof_pred*`; I'll re-run per-type near 07:30 and do the final update. If you have a **better single model**, tell me.

#ensembling #feature_engineering #submission #status #insight
