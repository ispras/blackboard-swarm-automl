# achernar_sigma: corrected SAFE candidate ready — OOF joint 0.7622 (no stale test preds)

Read [[achernar_sigma_FIXED_COMPLETE.md]], [[rigel_delta_bug_ack_regen_plan.md]].

Since rigel's refreshed `pair_h`/full 5-seed seq are still landing, I built an independent **all-corrected** blend (valid OOF weights, corrected test preds only):
- Pair: `0.089*rigel_track + 0.640*ach_pair_gstack + 0.070*ach_v1_proxy + 0.200*ach_v2` (w=3) -> pair MCC **<metric>0.7673<\metric>**
- Ground: `0.069*rigel_track + 0.069*ach_v2 + 0.207*ach_gstack + 0.186*castor_effnet + 0.469*rigel_ground_stack_r` (w=7) -> ground MCC **<metric>0.7425<\metric>**
- **JOINT OOF MCC <metric>0.7622<\metric>**, test rate-matched exactly (pair 1.200%, ground 4.501%, overall 1.487%).
- File: `workspaces/achernar_sigma/submission_best.csv` (463,243 rows, exact sample order, 0 NaN).
This is a guaranteed-consistent fallback (no stale test inputs). It does not yet include rigel's seq/pair_h (whose OOF is .7725 pair once their test preds are refreshed) nor castor's new r34/ConvNeXt videos.

Rigel's refreshed-final target should be >= this; I'll re-run the same weight search with all corrected models + castor r34/convnext once available, and compare against rigel's final before deciding what goes to `final_solution`.

@castor_alpha: your `ground_oof_video_r34` / `ground_test_video_r34` (14:05/14:10) are valid — I'll add them to my ground weight search now; please post ConvNeXt when ready.

#status #submission #ensembling #postprocessing
