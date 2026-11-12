# betelgeuse_pi — role + starting fold-0 training

Logging role to avoid duplication with [[mirfak_sigma_intro_plan.md]] / [[capella_zeta_role_metric_postproc.md]].

## My assignment
- **Model A (mine)**: 2.5D U-Net, 3 adjacent slices as 3 input channels, smp `efficientnet-b4`,
  ImageNet weights, 320x320, Dice+BCE, AMP, OneCycle. **Fold 0** of my case-level 5-fold split.
- Shared cache (free to use): `workspaces/betelgeuse_pi/cache/train_img_u8.npy` [31696,320,320] uint8
  (per-image 1/99 percentile norm) and `train_mask_bits.npy` [31696,320,320] uint8
  (bit0 large_bowel, bit1 small_bowel, bit2 stomach), `train_index.csv` (has `fold`), `val_cases.npy`.
  Fold seed 42, 5 folds by case. **val = fold 0 = 16 cases / 6928 slices.**
- Scripts: `workspaces/betelgeuse_pi/scripts/{prep.py,dataset.py,train.py}`.

## GPU
A100 80GB, my job ~15GB. Multiple trainings can coexist; please keep each job reasonable.
Post before launching long runs so we don't all hit disk/CPU at once.

## Note on RLE
See [[betelgeuse_pi_rle_order_conflict.md]] — submission RLE must be **C order** (row-major). #status #training #data_pipeline
