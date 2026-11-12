# Ensemble recipe & reproduction notes (mimosa_sigma)

The final `submission.csv` is the **3-model equal-weight (1:1:1) 2.5D ensemble**.
Machine-readable recipe: `recipe_ens_eq.json`. Scripts: `code_ensemble/`.

## Members
| agent | architecture | fold-0 OOF probs (script input) | all-data test probs (script input) |
|---|---|---|---|
| mimosa_sigma | smp `Unet` + `efficientnet-b4`, 320px, in_chans=5 (z-2..z+2) | `workspaces/mimosa_sigma/models/b4v2_f0_valprobs.npy` | `workspaces/mimosa_sigma/models/b4v2_all_f0_testprobs.npy` |
| dubhe_zeta | smp `Unet` + `efficientnet-b3`, 384px, in_chans=3 (z-1,z,z+1) | `workspaces/dubhe_zeta/handoff/val0_probs_final.npy` | `workspaces/dubhe_zeta/handoff/test_probs_all.npy` |
| betelgeuse_zeta | smp `UnetPlusPlus` + `efficientnet-b3`, 384px, in_chans=3 | `workspaces/betelgeuse_zeta/betel_f0_valprobs.npy` | `workspaces/betelgeuse_zeta/betel_all_testprobs.npy` |

All probs are `[N,3,320,320]` float16, class order `large_bowel, small_bowel, stomach`;
OOF rows are fold-0 order (`np.where(train_fold==0)[0]`), test rows are `cache/test_ids.json` order.
Members trained on folds 1-4 (group-KFold by `case_day`), validated on fold 0, then fine-tuned on all
225 case-days for test predictions (weights/thresholds transferred from clean fold-0 OOF).

## Postprocessing (per class, from `recipe_ens_eq.json`)
| class | thr | cc2d min area | cc3d min voxels | keep_top |
|---|---|---|---|---|
| large_bowel | 0.75 | 0 | 800 | 0 |
| small_bowel | 0.50 | 0 | 1500 | 0 |
| stomach | 0.50 | 100 | 4000 | 1 |

Steps: blend probs -> per-class threshold -> 2D connected-component min-area -> 3D 26-connectivity
component min-voxels (or keep-top-1 for stomach) -> upsample to original `(H,W)` -> RLE.
Metric used for selection: `0.4*Dice + 0.6*(1 - symmetric 3D Hausdorff / sqrt(H^2+W^2))`, Dice
both-empty=1, tuned on 30 fold-0 case-days, reported on 15 held-out case-days.

## Honest result
3-model equal blend, held-out fold-0 case-days: **combo 0.9529** (large .9418 / small .9420 / stomach .9749).

## Rejected variants (honest holdout, same protocol)
- +4th member FPN/b4: 0.9506-0.9531 (weights grid) -> not better, rejected.
- +4th member betel b4-Unet (small weight): holdout consistently -0.0004..-0.004 -> rejected.
- Per-class weight vector (dropping weak members): 0.9498 -> rejected.
- Hole filling: slightly worse -> rejected.
- 3D-CC / keep-top / threshold local grid confirmed the recipe above.

## Reproduction commands
```bash
# 1) caches (C-order RLE!)
python code_ensemble/build_cache.py          # images 320 uint8; use rebuild_masks.py for C-order masks
python code_ensemble/rebuild_masks.py
# 2) train each member (example mimosa)
python code_ensemble/train.py --arch Unet --encoder timm-efficientnet-b4 --fold 0 \
  --epochs 20 --bs 16 --nc 5 --lr 2e-4 --tag b4v2
# 3) all-data fine-tune
python code_ensemble/train.py --arch Unet --encoder timm-efficientnet-b4 --fold 0 \
  --epochs 5 --bs 16 --nc 5 --lr 1e-4 --tag b4v2_all --all-data 1 --init-ckpt <fold0 best.pt>
# 4) blend OOF/test probs and tune thresholds
python code_ensemble/make_blend.py --probs <oof1> <oof2> <oof3> --weights 1,1,1 --out oof_eq.npy
python code_ensemble/tune_postproc.py --probs oof_eq.npy --tag ens_eq --out recipe_ens_eq.json
# 5) build submission
python code_ensemble/make_submission.py --probs test_eq.npy --recipe recipe_ens_eq.json \
  --out final_solution/submission.csv
```

## Critical format note
`train.csv` RLE is **row-major (C-order)**: `flat.reshape(H,W)` with actual image `(H,W)` from the PNG
(filename numbers are `W,H`). Encoding is `mask.reshape(-1)`. Verified by exact string round-trip on
266x266 and 310x360 scans, and by image-mask intensity alignment. Using `order='F'` transposes masks
and destroys non-square scans.

## External models
See `README.md` (timm / segmentation_models_pytorch, ImageNet weights; Apache-2.0). No competition
data/solutions/APIs used.
