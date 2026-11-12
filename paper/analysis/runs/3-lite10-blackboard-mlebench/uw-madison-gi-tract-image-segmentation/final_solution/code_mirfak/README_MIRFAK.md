# mirfak_sigma — model code & provenance (contributed members of the ensemble)

These scripts are the exact code behind **mirfak_sigma's four contributed models** in
`final_solution/submission.csv` (see the main `../README.md`). They use SMP/timm **ImageNet** encoders only
(no competition-specific weights, no write-ups).

## Contributed models (all class order `[large_bowel, small_bowel, stomach]`, 3-channel 2.5D input z-1,z,z+1)
| role | arch / encoder | epochs | seed | notes |
|---|---|---|---|---|
| fold0 honest member | `UnetPlusPlus` + `efficientnet-b3` | 24 | 42 | `--arch UnetPlusPlus --encoder efficientnet-b3 --fold 0 --epochs 24 --bs 32 --lr 1e-3` |
| fold0 honest member | `DeepLabV3Plus` + `efficientnet-b3` (ASPP decoder, adds diversity) | 20 | 123 | `--arch DeepLabV3Plus --encoder efficientnet-b3 --fold 0 --epochs 20 --bs 32 --lr 1e-3` |
| all-data view | fine-tune of the above `UnetPlusPlus/b3` | 8 | 7 | `--fold -1 --epochs 8 --lr 5e-5 --init_ckpt <fold0 best>` |
| all-data view | fine-tune of the above `DeepLabV3Plus/b3` | 8 | 11 | `--fold -1 --epochs 8 --lr 5e-5 --init_ckpt <fold0 best>` |

## Files
- `common.py` — RLE (row-major / `order='C'` on true `(H,W)`, 1-indexed) + cache builder (16-bit PNG → P1/P99.5 clipped uint8 → 320×320; masks INTER_NEAREST).
- `dataset.py` — case-level 5-fold split (`seed42 i%5`), 2.5D slice sampling, augmentation (flips/rot90/affine/brightness/gamma).
- `train.py` — training (0.5·BCE + 0.5·soft-Dice, AdamW + cosine, AMP, grad clip); validation = per-slice Dice + per-scan 3D HD; `--fold -1` = all-data; `--init_ckpt` = fine-tune.
- `predict.py` — 4-way flip TTA inference → fp16 sigmoid `[N,3,320,320]` + `_meta.npz` (`ids, orig_hw, scan_id, k`) + `_ids.csv`.
- `metric.py` — Dice / normalized 3D Hausdorff (EDT-based) used for validation/tuning.
- `tune_classes.py` — per-class 2D threshold + 3D-CC `min_vol` + min-area tuner on OOF (per-slice Dice + per-scan 3D HD, empty-vs-empty = perfect).
- `make_submission.py`, `ens_submit.py` — id-aligned prob ensembling → threshold + per-slice fill/min-area + 3D-CC → native-resolution C-order RLE writer.

## Reproduce
```bash
python common.py both                 # build train/test cache (needs data/ at ../../data)
python train.py --arch UnetPlusPlus --encoder efficientnet-b3 --fold 0 --epochs 24 --bs 32 --lr 1e-3 --seed 42 --name unetpp_b3_f0
python predict.py --ckpt runs/unetpp_b3_f0/best.pt --split test --out oof/unetpp_b3_f0_test --tta 1
```
The shipped `submission.csv` is the swarm blend produced by capella's `run_finalB.sh`; my four prob dumps
(`unetpp_b3_f0`, `deeplab_b3_f0`, `unetpp_b3_all`, `deeplab_b3_all`, 4-way TTA) are inputs listed there.
Honest fold0 OOF of my individual models (per-slice Dice + per-scan 3D HD, empty=perfect):
`unetpp_b3_f0` ≈ 0.936, `deeplab_b3_f0` ≈ 0.925. Adding DeepLab to the 3-model blend raised OOF 0.9383→0.9395;
the 5-fold0-member blend reached ≈0.9399–0.9400.

External models: `efficientnet-b3` (ImageNet) via `timm`/`segmentation_models_pytorch` 0.3.4, license Apache-2.0.
