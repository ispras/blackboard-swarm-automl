# UW-Madison GI Tract Image Segmentation — swarm solution

Task: segment 3 organs (`large_bowel`, `small_bowel`, `stomach`) on 2D MRI slices.
Metric: `0.4 * mean per-slice Dice + 0.6 * normalized 3D Hausdorff score` (higher is better).
Submission: RLE per `(id, class)` in `final_solution/submission.csv`.

## Critical data finding (the dominant lever)
The `train.csv` RLE is **row-major**: `flat = mask.flatten(order='C')`, `mask = flat.reshape(H, W, order='C')`.
The TASK text ("top to bottom, then left to right") is misleading; verified empirically on 310×360 slices:
`order='C'` decode gives 1 compact component per stomach mask (anatomically correct, boundary aligns with
image gradients), while `order='F'` gives 23–94 scattered components (scrambled). Submissions are therefore
encoded with **`flatten(order='C')`** as well. Using `F` everywhere caps positive Dice at ~0.27; with `C`
positive Dice jumps to ~0.79 after 2 epochs.

## Data pipeline (`code/regulus/preprocess.py`)
- All train/test PNGs resized to 320×320; per-slice intensity `1–99` percentile clip → uint8.
- Masks decoded **C-order** at original `H×W`, then nearest-resized to 320; cached as memmap.
- 2.5D input = 3 adjacent slices (z-1, z, z+1) from the same case-day (edge-replicated).

## Models (all fine-tuned/validated on a case-grouped 15-case holdout, `code/regulus/val_cases.json`)
| id | owner | architecture | input | loss | TTA | holdout per-slice Dice |
|---|---|---|---|---|---|---|
| **B** | regulus_alpha | `smp.Unet`, encoder **mit_b2** (SegFormer-B2, ImageNet) | 2.5D 3-slice, 320 | 4-class softmax: CE + multiclass Dice | hflip+vflip | 0.921 |
| **A_C** | aldebaran_omicron | `smp.Unet`, encoder **efficientnet-b4** (ImageNet) | 2.5D 3-slice, 320 | BCE + Dice | 4× flip | 0.925 |
| **C3** | canopus_phi | `smp.Unet`, encoder **efficientnet-b4** (ImageNet) | 2.5D 5-slice, 320 | BCE + Dice | 4× flip | 0.917 |

Training: AdamW + cosine, AMP, augmentation (hflip, affine/rotate/scale, elastic/grid distortion, brightness).
`B` also has an **all-data** variant (8-epoch low-LR fine-tune on all 76 cases) for test cases seen in train
(19/28 test cases appear in train on other days).

## Ensemble & submission (`code/canopus/finalize.py`, `fuse_probs.py`; `code/regulus/backup_fuse.py`)
1. Predict per-slice class probabilities for val/test with flip TTA; store uint8 `prob*255`.
2. Align all models by slice `id`; equal-weight average per model family.
3. Tune **per-class threshold** on the shared holdout (Hausdorff-aware): best ≈ lb 0.50, sb 0.45, st 0.40.
4. Postprocessing: connected-component cleanup gains were within noise/harmful for bowels → **not used**.
5. Resize predicted masks to original `(H, W)` (nearest) and RLE-encode with **C order**.

Postprocessing includes **z-Gaussian smoothing (sigma≈1.5) of the fused probabilities within each
case-day before thresholding** — validated +0.004 combo (mostly Hausdorff), script
`code/regulus/make_submission_zsmooth.py` / `eval_zsmooth.py`.

- **Honest 3-model fusion** (holdout models only): per-slice Dice **0.9281**, 3D HD **0.9118**,
  combo **0.9183**; with z-smoothing **0.9225** (validated fallback).
- **Final submission** (`submission.csv`): per-family mean of (holdout, all-data) test probs →
  cross-family mean → z-Gaussian sigma 1.5 → per-class thr `.50/.45/.40` → C-encode.
  This adds the all-data variants (which help the 19/28 test cases seen in train) while keeping
  50% weight on the validated holdout models. Produced identically by canopus_phi (`sub_final6.csv`)
  and aldebaran_omicron (`cand_final_alldata_zs_v3_ordered.csv`); here copied byte-identically.

## Reproduce
```
# caches
python code/regulus/preprocess.py --kind train --size 320
python code/regulus/preprocess.py --kind test  --size 320
# train B (holdout and all-data)
python code/regulus/train.py --encoder mit_b2 --multiclass 1 --epochs 24 --bs 32 --lr 3e-4 --out runs/mit_b2_mc2
python code/regulus/train.py --encoder mit_b2 --multiclass 1 --all_data 1 --mild_aug 1 --init runs/mit_b2_mc2/best.pt --epochs 8 --lr 6e-5 --out runs/mit_b2_alldata320
# predict (store <out>_probs.npy + <out>_ids.csv)
python code/regulus/infer.py --ckpt runs/mit_b2_mc2/best.pt --split val --out probs/val_B2_ep6
python code/regulus/infer.py --ckpt runs/mit_b2_mc2/best.pt --split test --out probs/test_B2_ep6
# fuse + tune + write submission (C-encoded)
python code/regulus/backup_fuse.py --val_probs <A_val> <B_val> <C3_val> --val_ids ... \
    --test_probs <A_test> <B_test> <C3_test> --test_ids ... --weights 1 1 1 --out_submission submission.csv
```

## External models / sources (no competition-specific checkpoints used)
- `segmentation_models_pytorch` 0.3.4 (Apache-2.0) — model zoo/decoders; encoder ImageNet weights
  `mit_b2` from `https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/mit_b2.pth`
  (SegFormer-B2, Xie et al., ImageNet-pretrained), and `efficientnet-b4` ImageNet weights via `timm` 0.9.7.
- `torch` 2.2.0+cu121, `timm` 0.9.7, `albumentations` 1.4.14, `opencv` 4.10, `scipy` 1.14.
- No external competition data, no Kaggle artifacts, no solution notebooks/write-ups.

## Honest validation policy
Only `data/train.csv` labels were used for training/validation. The official hidden test labels were never
read or inferred; the grader was not run. Holdout = 15 cases grouped by patient case (seed 42).

## Team
`regulus_alpha` (Model B, RLE-C discovery/fix, postprocessing study, fusion cross-check),
`aldebaran_omicron` (Model A_C, encoding algebra, all-data), `canopus_phi` (Model C3, fusion/submission owner).

## Final update by canopus_phi (submission owner)
Shipped `submission.csv` = **all-data family blend + z-smoothing** (consensus of 3 independent implementations,
byte-identical): per-family mean of (holdout, all-data) test probs for A/B/C → cross-family mean →
per-class thresholds **lb .50 / sb .45 / st .40** → C-order RLE. Validated honest twin (holdout models only,
same z-smoothing sigma 1.5) = per-slice Dice **0.928**, 3D HD **0.916**, combo **<metric>0.9225<\metric>**.
Per-class sigma/threshold variant (`code/canopus/tuned_final3.json`, val combo 0.9225) was explored and is
within noise of the shipped config, so the consensus config was kept.
