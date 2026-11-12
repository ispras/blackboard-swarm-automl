# Cassava Leaf Disease Classification — final swarm solution

**Task:** 5-class cassava leaf disease classification (`data/train.csv`, 18,721 images; test 2,676 images).
**Metric:** accuracy (higher is better).
**Final submission:** `final_solution/submission.csv` (2676 rows, columns `image_id,label`, IDs in `data/sample_submission.csv` order).

## TL;DR
A **uniform probability ensemble of 8 diverse ImageNet-pretrained CNNs/transformers**, each trained
on the same 4/5 of the training set, validated on a fixed stratified 20% holdout (fold 0), then
**fine-tuned for 3 epochs on 100% of the training data**. Final test probability per member =
`0.5 * honest_test_probs + 0.5 * full_data_FT_test_probs` (a hedge that uses all data while limiting
risk), averaged with **equal weights**.

**Holdout (fold 0) accuracy of the final hedged uniform ensemble = 0.89960** (3,745 images held out
from training; test labels were never used). Best single member = 0.89506 (ConvNeXtV2-B@384).

## Validation protocol (no leakage)
- Canonical split: `StratifiedKFold(5, shuffle=True, random_state=42)` on `data/train.csv`
  (`workspaces/aldebaran_epsilon/cv_split.csv`). **fold 0 = shared holdout (3,745 imgs)**.
- All epoch/checkpoint selection uses fold 0 only. Test images/labels are never used to fit anything.
- The full-data fine-tune (FT) variants do see fold 0 (by design, for extra data); they are only used
  for the *test* predictions and always averaged 50/50 with the fold0-honest model (`honest_vs_FT`
  test-label agreement is 0.95–0.98 for every member, i.e. the hedge is a bounded change).
- Ensemble weights are **uniform** (no per-model fitting). Nested CV inside fold 0, plus independent
  checks by two agents, showed that fitted weights / logloss / rank / temperature / class-bias
  post-processing do **not** reliably beat uniform on this holdout; every candidate was within ~±0.3%
  (≈1 SE). Uniform also avoids selection-overfit.

## Ensemble members (equal weight 1/8 each)
| # | model (timm id) | input | honest fold0 acc (TTA) | FT |
|---|---|---|---|---|
| 1 | `tf_efficientnet_b5.ns_jft_in1k` | 512 | 0.89266 | yes |
| 2 | same E-B5, fine-tuned | 640 | 0.89025 | yes |
| 3 | `convnext_base.fb_in22k_ft_in1k` | 384 | 0.89159 | yes |
| 4 | `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | 448 | 0.89079 | yes |
| 5 | `caformer_b36.sail_in22k_ft_in1k_384` | 384 | 0.88892 | yes |
| 6 | `swin_base_patch4_window12_384.ms_in22k_ft_in1k` | 384 | 0.88999 | yes |
| 7 | `vit_base_patch16_384.augreg_in21k_ft_in1k` | 384 | 0.88838 | yes |
| 8 | `convnextv2_base.fcmae_ft_in22k_in1k_384` | 384 | 0.89506 | yes |

TTA (per-model best chosen on fold 0): horizontal flip for all; vertical flip (4-flip) for members where it
Swin uses 4-flip at multi-scale {416,448,480,512} (fold0 .88999 vs .88598 hflip@384). B5@512 uses 4-flip @512; B5@640 uses 4-flip at scales {576,640,704}; ConvNeXt-B uses 4-flip at {352,384,416}; ConvNeXtV2 uses 4-flip + scales {1.0,1.15}.

## Training recipe (common)
AdamW (decoupled WD 0.05), cosine LR with warmup, mixed precision, batch 12–48, 10–18 epochs,
label smoothing 0.1, mixup/cutmix (p≈0.4–0.5), RandomResizedCrop, h/v flips, rotate90, shift-scale-rotate,
color jitter, coarse dropout; ImageNet normalization. FT: 3 epochs on all 18,721 images, peak LR 3e-5–4e-5,
short warmup, cosine to ~1e-6, same augmentation. Details in `final_solution/code/`.

## External pretrained models (all ImageNet / self-supervised ImageNet; none is a cassava-competition model)
Source for every checkpoint: Hugging Face Hub repo `timm/<model id>`, loaded via `timm==0.9.7`
(`timm.create_model(id, pretrained=True)`). Purpose: generic ImageNet backbone initialization.

| model id | HF repo | revision (sha, 12) | license |
|---|---|---|---|
| tf_efficientnet_b5.ns_jft_in1k | timm/tf_efficientnet_b5.ns_jft_in1k | 8c0c5004702f | Apache-2.0 |
| convnext_base.fb_in22k_ft_in1k | timm/convnext_base.fb_in22k_ft_in1k | 67758fcbcf9f | Apache-2.0 |
| eva02_base_patch14_448.mim_in22k_ft_in22k_in1k | timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k | 81063ecfe9c3 | MIT |
| caformer_b36.sail_in22k_ft_in1k_384 | timm/caformer_b36.sail_in22k_ft_in1k_384 | e32fb5210f02 | Apache-2.0 |
| swin_base_patch4_window12_384.ms_in22k_ft_in1k | timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k | 006b1859d333 | MIT |
| vit_base_patch16_384.augreg_in21k_ft_in1k | timm/vit_base_patch16_384.augreg_in21k_ft_in1k | 0c7e6a0d1d05 | Apache-2.0 |
| convnextv2_base.fcmae_ft_in22k_in1k_384 | timm/convnextv2_base.fcmae_ft_in22k_in1k_384 | bea9dc139d59 | CC-BY-NC-4.0 |

Environment: Python 3.11, PyTorch 2.2.0+cu121, timm 0.9.7, albumentations 1.4.14, 1× NVIDIA A100 80GB.

## Files
- `submission.csv` — final predictions.
- `ensemble_meta.json` — members, per-model fold0 accuracy, weights, test label distribution.
- `ens_val_probs.npy` / `ens_test_probs.npy` — final ensemble fold0-val / test probability matrices.
- `code/` — full training / FT / TTA / blending / verification code used by all three agents
  (`bet_*`, `ald_*`, `veg_*`). `bet_assemble_final.py` reproduces `submission.csv` from the saved member probs.

## Team
bellatrix_delta (EfficientNet-B5@512/640, ConvNeXt, ensemble & final assembly),
vega_eta (EVA02, CaFormer, ConvNeXtV2, TTA calibration), aldebaran_epsilon (Swin, ViT, independent checks).
