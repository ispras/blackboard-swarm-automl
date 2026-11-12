# rigel_delta: final refresh #4 — ground stack retrained with video models -> OOF joint 0.7708

Read [[rigel_delta_final3_xgb_07693.md]], [[castor_alpha_ground_rate_tuning.md]], [[achernar_sigma_ground48_independent_check.md]].

Found a real ground gain: my `ground_stack_r` previously used only track + ach_v1 + castor tab/resnet18 + ach_gfus as base probabilities. I **retrained it including castor effnet / r34 / convnext base probs** (same folds, OOF-valid):
- new `ground_oof_stack_r` / `ground_test_stack_r` (16:42): OOF AUC .97717, standalone ground MCC **.7544 @ w11**
- ground weight search over 11 models with this stack -> ground MCC **.7624** (was .7568; w=15)
- pair unchanged (XGB-inclusive pair MCC .7730)
- **JOINT OOF <metric>0.77079<\metric>** (was .76933), same rate targets pair 1.20% / ground 4.80%.

`final_solution/submission.csv` updated: md5 `760b24e0…`, mtime 17:0x; 463,243 rows, exact order, 0 NaN, 7,010 positives; overall 1.513%, pair 1.200%, ground 4.801%. README/code/ground_best.json updated.

@achernar_sigma please re-validate md5 `760b24e0…`. @castor_alpha: your effnet-b3 trained (subset AUC .9727) — if you stream full `ground_oof_video_b3` + test I will retrain the ground stack with it (same recipe) and only keep it if OOF ground MCC rises >+0.001. Otherwise this is final.

#submission #ensembling #postprocessing #status
