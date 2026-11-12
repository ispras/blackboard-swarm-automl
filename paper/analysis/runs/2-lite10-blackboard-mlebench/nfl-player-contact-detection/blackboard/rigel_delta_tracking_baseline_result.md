# rigel_delta: tracking-temporal GBDT baseline — OOF results + artifacts

Read [[rigel_delta_folds_decision.md]], [[castor_alpha_video_scope_and_folds_final.md]]. Trained on **castor's game_key folds** (castor_alpha/folds.csv).

## Results (OOF, separate pair / ground LightGBM, no helmet features)
- Pair contact: AUC **0.99679**, MCC **<metric>0.7339<\metric>** (thr 0.268)
- Ground contact: AUC **0.94336**, MCC **<metric>0.4589<\metric>** (thr 0.090)
- **Joint MCC (optimized pair+ground thresholds): <metric>0.6599<\metric>** (thr_pair 0.215, thr_ground 0.220)

This is tracking-only (91 pair cols / 56 ground cols: relative kinematics, closing speed, ttc, rich lags/leads + rolling min/mean, temporal neighbourhood). Intentionally **no helmet/video** = diversity for the blend.

## Artifacts (contact_id-keyed, on castor folds)
- `workspaces/rigel_delta/oof_pair.parquet` — contact_id, contact, fold, prob (3,888,024)
- `workspaces/rigel_delta/oof_ground.parquet` — same (370,351)
- `workspaces/rigel_delta/testpred_pair.parquet`, `testpred_ground.parquet` — 24 test plays
- fallback full submission: `workspaces/rigel_delta/submission_lgb_v1.csv`
- feature tables: `train/test_{pair,ground}.parquet`

## Next
1. Merge achernar's `train/test_helmet_features.parquet` (per [[achernar_sigma_helmet_features_and_model_start.md]]) into my pipeline -> tracking+helmet pair model (CatBoost or different LGBM seed for diversity).
2. Blend ground: my tracking (MCC .459) + castor tab ground (MCC .581, per [[castor_alpha_ground_tab_baseline_result.md]]) + achernar + castor ground-video, then joint threshold + temporal smoothing.
3. I can own the **consensus blender / threshold / post-processing**; please publish OOF+test preds (contact_id, prob) on castor folds as they finish, and I'll produce the blend with documented weights. @castor_alpha @achernar_sigma — OK with me owning the blender?

#status #model_selection #ensembling #postprocessing #submission
