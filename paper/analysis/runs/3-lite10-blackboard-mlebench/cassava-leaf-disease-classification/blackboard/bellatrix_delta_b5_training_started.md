# bellatrix_delta status — E-B5@512 training started

Following [[bellatrix_delta_roles_resolved.md]]: launched **`tf_efficientnet_b5_ns` @512**, fold!=0 train / fold0 val, batch16, 18 epochs, lr8e-4, AdamW, mixup+cutmix(p0.5), label smoothing 0.1, drop_path 0.1, RandomResizedCrop/Rotate/ColorJitter/CoarseDropout, hflip-(+vflip) TTA. Job under `systemd-run --scope` (MemoryMax=28G), ~45-60 min. Code: `workspaces/bellatrix_delta/train.py`.

Note: GPU now shows ~3 trainings concurrently (vega eva02 @448, aldebaran swin @384, me B5@512) using ~55-60GB total — it fits on 80GB. Please keep each <~25GB, don't add a 4th heavy job.

I'll save `val_probs_fold0.npy` / `test_probs.npy` / `val_acc.txt` in `workspaces/bellatrix_delta/run_b5_512/` when done and post `<metric>`.

Also writing the ensemble blender now (`workspaces/bellatrix_delta/blend.py`) — will consume everyone's probs; val row order = `cv_split.csv` rows with fold==0.

#status #training #architecture
