# aldebaran_omicron — Model A + verification code

Model A in the final ensemble: `smp.Unet` with **efficientnet-b4** ImageNet encoder
(`segmentation_models_pytorch` 0.3.4 / `timm` 0.9.7), **2.5D** input (z-1,z,z+1),
320×320, BCE + soft-Dice, AdamW + OneCycle. Trained decoding masks with
`order='C'` (row-major; see swarm README), 4× flip TTA at inference.

- Holdout (15 case-grouped cases): per-slice Dice **0.925** (4× TTA).
- All-data variant: 6-epoch low-LR fine-tune on all 76 train cases.

## Reproduce Model A
```
pyenv shell sis-black
python prep_cache.py train      # cache 320px images (order = train_index.csv)
python train.py --encoder efficientnet-b4 --arch Unet --stack 3 --size 320 \
  --epochs 18 --batch 24 --lr 3e-4 --workers 6 --tag A
python predict.py --ckpt runs/A_f0/best.pt --split test \
  --out probs/test_A_AC_probs.npy --batch 32 --workers 4 --tta 1
```
`train.py --mask_cache_dir <cache>` uses a precomputed combined mask memmap
(canopus cache `data/cache_train_320`, values {0=none,1=lb,2=sb,3=st}) for speed.

## Key files
- `convention.py` — RLE convention proof/algebra (`C` encode/decode; submission encode must match train-decode convention).
- `zsmooth.py` — Gaussian along z within each case-day on probabilities (sigma≈1.5, +~0.004 combo).
- `postproc.py` — threshold + optional CC cleanup; `make_submission` resizes to original H×W and RLE-encodes `order='C'`.
- `metric.py`, `qa_submission.py` — local Dice/HD proxy + submission QA.
- `build_final_candidate.py`, `fuse_test_check.py` — id-aligned family-mean fusion used to build the final all-data blend.

External artifacts: only ImageNet-pretrained `efficientnet-b4` weights (timm/smp official
hub, Apache-2.0/BSD-style). No competition-specific checkpoints or solutions used.
