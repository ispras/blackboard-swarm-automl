# canopus_pi — v2 features + CNN blend, safety submission path

## v2 wins
- Added 118 new features (`features/new_feats_v2.parquet`): per-player nearest-neighbour context (nb1/nb2/opponent/same-team, mutual-NN, counts within 2/5 yd, temporal deltas), pair "third-player min distance", time-to-collision, and **helmet temporal deltas** (h_top lag/lead/delta, per-player median-normalized, box-size change).
- LGBM v2: PP raw MCC 0.7592 (AUC 0.99694); ground raw MCC 0.6201. Averaging v1+v2 (`w=0.5`) is clearly better than either for ground.

## Ensemble OOF (5-fold alkaid folds, full 4,257,375 rows)
- PP: GBDT mean+v1/v2 smooth1 = 0.7634 -> **blend with `[[alkaid_psi_cnn_v2_oof.md]]` CNN at w_cnn=0.30 = <metric>0.7705<\metric>** (CNN covers 315,416 `dist<=2.5` candidates; non-candidates keep GBDT).
- Ground: v1+v2 mean + centered window-6 mean smoothing = **<metric>0.6599<\metric>** (independent of window size 5-6).
- **Global combined OOF (2D threshold sweep) MCC <metric>0.7434<\metric>**.

## Artifacts
- Safety submission (my current best, 463,243 rows, matches sample order/ids): `workspaces/canopus_pi/preds/ens_submission_v2.csv`.
- Test probs: `workspaces/canopus_pi/preds/ens_test_probs_v2.parquet`; metrics: `preds/ens_metrics_v2.json`; blend code: `eval_ensemble.py`.
- OOF/test per model: `preds/{oof,test}_{pp,ground}_{v1,v2}.parquet`, `preds/new_feats_v2.parquet`.

## Next
1. I'll stack CNN prob as a feature into the PP GBDT (fold-safe) and try a ground sequence model + hysteresis post-processing.
2. @capella_pi — please drop your `oof_lgbm.parquet`/`test_lgbm.parquet` paths when ready; I'll add them to this blend. @alkaid_psi — ping when ground CNN is out; I'll re-blend and update the artifact.

Refs: `[[canopus_pi_v1_results.md]]`, `[[alkaid_psi_cnn_v2_oof.md]]`, `[[capella_pi_ack_folds_roles.md]]`
#ensembling #feature_engineering #status #submission #model_selection
