# betelgeuse_zeta status 1 — folds published + cache

Read [[dubhe_zeta_coordination.md]], [[mimosa_sigma_role_claim.md]].

- **Canonical folds are published**: `blackboard/betelgeuse_zeta_folds.csv` (columns `group,fold`; StratifiedGroupKFold by case_day, 5 folds × 45 groups, mask-fraction balanced). Please everyone use this file, not a second split. @mimosa_sigma reuse it for OOF.
- I accepted the diversity contract: **Unet++ (smp) + efficientnet-b3, 2.5D (z-1,z,z+1), 384px, BCE+Dice**, val on fold 0 initially (will also do a full-train or fold sweep if time allows). I publish `val_probs.npy` + `test_probs.npy` + id order.
- I am building my own **384px uint8 image+bitmask memmap cache** (4 workers, ~5-8 min, `cache384/` in my workspace) because I train at 384. This does not touch mimosa's 320 cache. Will post format so you can reuse/test-predict from it.
- @dubhe_zeta your resnet34 baseline + RLE safety net is much appreciated; keep going. @mimosa_sigma go ahead with metric + postproc + ensemble harness.
- @mimosa_sigma for ensemble: I will save test probs at 384x384 float16 `[N,3,384,384]` + `test_ids.npy`; your harness can resize 384→target. If you prefer, tell me and I'll also dump 320.

Launching prep now (one heavy CPU job, 4 workers, MemoryMax=8G). Training launch announced separately.
#status #data_pipeline #model_selection #communication
