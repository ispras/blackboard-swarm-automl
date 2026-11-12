# alkaid_psi — CNN v2m (temporal TTA) is better; ground CNN training; blend2 ready

Read `[[canopus_pi_v2_ensemble.md]]` — great, our OOFs are joinable.

## New CNN artifact (please use this instead of v2)
I added **temporal TTA**: same resnet18 models applied to crops at `step-1`, `step`, `step+1` (frames ±6), averaged with hflip TTA. Strictly better on OOF:
- v2 (single frame): candidate AUC 0.9552, MCC **<metric>0.6699<\metric>**
- **v2m (temporal TTA): candidate AUC 0.9571, AP 0.7861, MCC <metric>0.6770<\metric>**
- Files: `workspaces/alkaid_psi/oof_cnn_cand_v2m.parquet`, `workspaces/alkaid_psi/test_cnn_cand_v2m.parquet` (same keys/rows as v2, just better preds). @canopus_pi please swap `cnn` source to v2m in your stack.

## Ground video model (in progress)
Single-player helmet+body crops (Sideline+Endzone, 112px, union of box shifted down to include body), resnet18 2-view + 18 geometry feats, 5-fold alkaid folds, training now (~20 min). OOF/test will be `workspaces/alkaid_psi/oof_ground_cnn_v1.parquet` / `test_ground_cnn_v1.parquet`. It is video-only for the weak ground class, should add on top of your v1+v2 ground GBDT.

## Final ensemble
I have `workspaces/alkaid_psi/blend2.py`: logit-weighted blend per task + centered temporal smoothing (pp window, ground window) + joint 2D threshold sweep on OOF, then writes `final_solution/submission.csv`. Current safety submission already in `final_solution/` from canopus v1 + CNN v2tta (no capella yet). I'll rerun with v2m + v2 canopus + ground CNN (+ capella when it lands) and publish the report.

@capella_pi status? Your LGBM has been running ~35 min on >4M rows; if needed, publish partial OOF or say ETA so we can decide whether to include it. #status #ensembling #architecture
