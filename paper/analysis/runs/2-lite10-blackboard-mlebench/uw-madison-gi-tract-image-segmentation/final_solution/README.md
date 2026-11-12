# UW-Madison GI Tract Image Segmentation — final solution

Task: 2D MRI segmentation of 3 organs (`large_bowel`, `small_bowel`, `stomach`).
Metric: `0.4 * Dice + 0.6 * (1 - normalized 3D Hausdorff)` (`dice-hausdorff-combo`, higher is better).
Submission: RLE masks in `submission.csv` with columns `id,class,predicted`.

## Final submission (swarm ensemble)
`submission.csv` is the **3-model equal-weight ensemble** (weights 1:1:1) of independently trained
2.5D segmentation networks, one per agent. Blend is done on per-slice class probabilities at 320x320,
then postprocessed with per-class threshold + 3D connected-component filtering:

| class | threshold | cc2d min area | cc3d min voxels | keep_top |
|---|---|---|---|---|
| large_bowel | 0.75 | 0 | 800 | 0 |
| small_bowel | 0.50 | 0 | 1500 | 0 |
| stomach | 0.50 | 100 | 4000 | 1 |

Recipe file: `recipe_ens_eq.json`.

**Honest fold-0 OOF (45 case-days; tuned on 30, reported on 15 held-out case-days): 0.9529 combo**
(large_bowel 0.9418 / small_bowel 0.9420 / stomach 0.9749). Ensemble members (fold-0 OOF):

| agent | architecture | input | test model used |
|---|---|---|---|
| mimosa_sigma | smp `Unet` + `efficientnet-b4`, 320px, 2.5D **5-channel** (z-2..z+2) | all-data fine-tune |
| dubhe_zeta | smp `Unet` + `efficientnet-b3`, 384px, 2.5D 3-channel (z-1,z,z+1) | all-data fine-tune |
| betelgeuse_zeta | smp `UnetPlusPlus` + `efficientnet-b3`, 384px, 2.5D 3-channel | all-data fine-tune |

All members were first trained on folds 1-4 (group-KFold by `case_day`) and validated on fold 0;
they were then fine-tuned on all 225 case-days for the test predictions (thresholds/weights are
selected on the clean fold-0 OOF and transferred, since only the training data changed).

## Reproducible pipeline (see `code/`)
1. `build_manifest.py` — map every `id` to its image path and true `(H,W)`; group by case-day.
2. `common.py` — dataset: 2.5D input, per-image 1-99 percentile normalization, augmentation,
   RLE encode/decode, metrics helpers.
3. `train.py` — smp model, BCEWithLogits + Dice loss, AdamW, warmup + cosine LR, AMP, grad clip.
   Group K-fold by `case_day` (folds from the swarm board `betelgeuse_zeta_folds.csv`).
   `--all_data 1 --init_ckpt <fold0 best>` performs the all-data fine-tune.
4. `infer.py` — 4-way TTA (identity, hflip, vflip, rot180), writes `[N,3,320,320]` float16 probs.
5. `make_submission.py` — per-class threshold, 3x3 close, 2D CC min-area, upsample to original
   `(H,W)`, row-major RLE.
6. `simlib.py` — local metric (Dice both-empty=1, 3D symmetric Hausdorff, combined).
7. `verify_submission.py` — validates row order/ids, RLE structure, bounds, decodability.

## Critical data-format finding (verified empirically, contradicts TASK.md)
The RLE in `train.csv` is **row-major (C-order)**: with image `(H,W)` from the actual PNG
(filename numbers are `W,H`), `mask = flat.reshape(H, W)` and encoding is `mask.reshape(-1)`.
Using `order='F'` (as the task text suggests) transposes square masks and destroys non-square
(310x360) masks; fixing this raised fold-0 positive-slice Dice from ~0.10 to ~0.65 already after
one epoch. All members use the verified C-order convention.

## External models (general ImageNet backbones; no competition-specific data)
| model | source | revision | purpose | license |
|---|---|---|---|---|
| `efficientnet_b3`, `efficientnet_b4` (timm) via `segmentation_models_pytorch` (`Unet`, `UnetPlusPlus`), `encoder_weights='imagenet'` | https://github.com/huggingface/pytorch-image-models , https://github.com/qubvel/segmentation_models_pytorch | timm 0.9.7, smp 0.3.4 | encoder initialization | Apache-2.0 |
| ImageNet pretrained weights | https://download.pytorch.org (torch hub / timm) | — | weights | Apache-2.0 |
| `resnet34` via smp `Unet` (early baseline, not in final ensemble) | torchvision 0.17.0 | — | encoder initialization | BSD-3-Clause |

Only general ImageNet-pretrained weights were used. No test labels, private data, competition
solutions, or external APIs were used. `torch 2.2.0+cu121`, Python 3.11.

## Code layout (swarm)
- `code/` — dubhe_zeta pipeline: `build_manifest.py`, `common.py`, `train.py`, `infer.py`,
  `make_submission.py`, `simlib.py`, `handoff.py`, `verify_submission.py`.
- `code/mimosa_sigma/` — ensemble/postproc/metric implementation used for the final blend:
  `blend_search.py`, `make_blend.py`, `tune_postproc.py`, `eval_model.py`, `simlib.py`,
  `make_submission.py`, plus that member's `train.py`/cache builders.
- `code/betelgeuse_zeta/` — that member's `train.py`, `prep_cache.py`, `infer.py`, evaluators.

## Ensemble selection notes
- 3-model equal weights (1:1:1) was the best honest fold-0 holdout: **0.9529**.
- A 4th diverse member (`FPN + efficientnet-b4`, fold-0 OOF 0.9471) was tested and **rejected**:
  honest 4-model retune holdout 0.9506 < 0.9529. Hole-filling postprocessing also slightly hurt.
- Final postprocessing is 3D connected-component filtering (plus keep-largest for stomach);
  see `recipe_ens_eq.json`.
