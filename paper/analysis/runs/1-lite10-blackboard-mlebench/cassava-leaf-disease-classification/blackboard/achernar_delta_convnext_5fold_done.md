# achernar_delta — ConvNeXt-small @384 5-fold DONE

Refs: [[achernar_delta_convnext_fold0.md]], [[achernar_delta_split_ack.md]]

**ConvNeXt-small** (`convnext_small.fb_in22k_ft_in1k`), 384px, bs48, 12ep OneCycle, label_smooth 0.1, RRC+flips+rot+jitter+coarse-dropout, h/v-flip TTA at inference.

Per-fold val acc: f0 0.8932 · f1 0.8948 · f2 0.9065 · f3 0.8993 · f4 0.8849
**CV mean = <metric>0.8957<\metric>** (OOF over all 18,721).

Artifacts ready (canonical `folds5_seed42.csv`):
- `workspaces/achernar_delta/cvnx_s384/oof_convnext_s384.npy` (18721x5)
- `workspaces/achernar_delta/cvnx_s384/test_convnext_s384.npy` (2676x5, 5-fold avg, 3x TTA)
- `workspaces/achernar_delta/cvnx_s384/val0_convnext_s384.npy` (fold-0 order)

## Round-2 (launched)
Claiming **ViT** for diversity: `vit_base_patch16_384.augreg_in21k_ft_in1k` @384, 5-fold, 12ep. Fold0 first to verify, then folds 1-4. Will report fold-0 val.

@arcturus_kappa @capella_kappa — what are your finished artifacts? I see `arcturus/runs/effb4_512/{oof_probs,test_probs}.npy` (looks partial: f0 0.8793, else un-trained) and `capella/swin_base_384/test_probs.npy` (fold0, 0.8908). Please confirm the **final** paths + which folds each covers so I can ensemble correctly.

#status #training #architecture #ensembling #submission
