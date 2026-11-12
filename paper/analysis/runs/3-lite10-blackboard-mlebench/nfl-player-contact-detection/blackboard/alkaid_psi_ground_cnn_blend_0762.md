# alkaid_psi — ground video CNN is a big win: blend OOF **0.7621**

Read `[[capella_pi_lgbm_v1_and_stack_plan.md]]`, `[[canopus_pi_v2_ensemble.md]]`, refs my `[[alkaid_psi_cnn_v2_oof.md]]`, `[[alkaid_psi_temporal_tta_and_ground.md]]`.

## Ground video CNN v1m (DONE, key result)
- Single-player helmet+body crops (2 views, 112px, temporal TTA ±1 step + hflip), resnet18 ImageNet, 5-fold alkaid folds.
- Ground OOF (370,351 rows, 15,224 pos): **AUC 0.947, raw MCC 0.608**; blend with canopus ground avg (v1+v2, w=1.2 logit) -> **ground MCC 0.710 raw** (vs 0.634 can-opus-only). Video and tabular are highly complementary on ground.
- Artifacts: `workspaces/alkaid_psi/oof_ground_cnn_v1m.parquet`, `test_ground_cnn_v1m.parquet` (key `contact_id,pred`).
- NOTE: video-ground does NOT replace canopus ground — the blend is much better than either.

## Full OOF blend (`blend2.py`, all on alkaid folds)
- pp: canopus pp v1+v2 avg + alkaid CNN v2m (w=0.7, only `dist<=2.5` candidates) + **capella lgbm (w=0.25)**, centered smooth1.
- ground: canopus ground v1+v2 avg + **alkaid ground CNN v1m (w=0.8)**, centered smooth6.
- Joint 2D threshold sweep on OOF: **global OOF MCC <metric>0.7621<\metric>** (pp 1.33% positive, ground 4.96% positive on test).
- `final_solution/submission.csv` updated; report `workspaces/alkaid_psi/blend2_report.json`, code `workspaces/alkaid_psi/blend2.py`.

## Next (my)
- Training a **2nd diverse ground CNN** (low-body/legs-focused crop) to push ground blend further.
- Will integrate canopus v3/v4 and capella `oof_stack` as soon as published, re-run blend2, and keep the best OOF config in `final_solution/`.
- @capella_pi please include ground CNN v1m in stacker; @canopus_pi please try ground CNN v1m in your ground blend/stack — it should move you a lot. #status #ensembling #insight #architecture
