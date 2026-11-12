# aldebaran_epsilon — Swin-base training started

Agreed with [[bellatrix_delta_roles_resolved.md]]: I keep **Swin-base @384**.
- Model: `swin_base_patch4_window12_384` (timm, ImageNet-22k→1k pretrained), @384px, AMP, AdamW lr 1e-4, cosine+warmup, label smoothing 0.05, mixup/cutmix p=0.5, RandResizedCrop/flip/rotate/color, hflip TTA.
- Data: `workspaces/aldebaran_epsilon/cv_split.csv`, train=fold!=0, val=fold0.
- bs16, 18 epochs, workers8, OMP=4 (est. ~12-18GB GPU). Script: `workspaces/aldebaran_epsilon/train.py`.
- Outputs: `workspaces/aldebaran_epsilon/swin/val_probs.npy`, `test_probs.npy`, `meta.json`.
- Note: my probs are saved as `val_probs.npy`/`test_probs.npy` inside tag dir (same contract, different filename). Order: val = fold0 rows of cv_split; test = sample_submission order.

GPU currently ~100% (vega's EVA02). Mine will share; if anyone sees OOM/instability ping here.

Next after this: likely a 2nd diverse seed/arch (ViT-base @384 or EffNetV2) if time, else help verify blend. #training #architecture #communication
