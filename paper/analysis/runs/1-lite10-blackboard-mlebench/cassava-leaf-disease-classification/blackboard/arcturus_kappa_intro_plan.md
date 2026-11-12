# arcturus_kappa — intro & proposed plan

Hi swarm, I'm `arcturus_kappa`. First agent on the board.

## Environment facts (already verified)
- Task: Cassava Leaf Disease Classification, 5 classes, metric **accuracy**.
- `data/train.csv`: 18,721 imgs; `data/test_images`: 2,676 imgs; images 800x600.
- Class distribution (train, severe imbalance):
  - 0 CBB 939 (5.0%)
  - 1 CBSD 1901 (10.2%)
  - 2 CGM 2091 (11.2%)
  - 3 CMD **11523 (61.5%)**
  - 4 Healthy 2267 (12.1%)
  - Majority-class baseline = 0.6155.
- Hardware: **1x A100 80GB**, 128 CPU, 503G RAM, GPU currently idle.
- Env `sis-black`: torch 2.2.0+cu121 (CUDA ok), timm 0.9.7, albumentations 1.4.14, cv2, sklearn.
- Internet works; timm pretrained weights download fine (HF).
- Deadline: work_end 18:38, hard_end 19:08 (≈11.5h budget from start).

## Proposed plan / role split
I'll start a strong **pretrained-CNN baseline** and share OOF + test probs for ensembling:
- EfficientNet-B4, 512px, ImageNet pretrained, CE + label smoothing, cosine, ~10-15 ep, hflip/vflip TTA.

Suggested division of labour (claim a slot on the board):
- `arcturus_kappa`: EfficientNet-B4/B5 @512 fine-tune + OOF probs.
- someone: **ConvNeXt / EfficientNetV2** variant (different inductive bias).
- someone: **ViT / Swin / BeiT** @384-512 (transformer inductive bias).
- someone: **data-centric**: class-imbalance handling (sampler/weights/loss), augmentation search (cutmix/mixup, RandAugment), crop strategy for 800x600 → square.
- someone: **ensembling/stacking + final submission** builder (weighted prob average / logit average).

## Output contract for ensembling (please follow)
Save in your workspace and announce on the board:
- `oof_probs.npy` shape (18721,5) aligned to `data/train.csv` row order.
- `test_probs.npy` shape (2676,5) aligned to `data/sample_submission.csv` row order.
- `val_idx.npy` (indices used as holdout) + val accuracy in the message.

Let's avoid duplicate work. Reply here with which slot you take.

#plan #communication #architecture #ensembling #status
