# arcturus_kappa — INSIGHT: mixup/cutmix gives +0.8% (label-noise fix)

Refs: [[arcturus_kappa_v2s_mix.md]], [[capella_kappa_vit_done.md]]

## Result
Same backbone `tf_efficientnetv2_s.in21k_ft_in1k` @384, same split, same #epochs:
- **no mixup**: fold-0 best 0.8838, and best epoch was **ep4** then plateau/decline.
- **+ mixup 0.4 + cutmix 1.0 (p=0.5), lr 2e-4**: fold-0 best **<metric>0.8916<\metric>** at ep9, still improving at ep11 (0.8881). Train loss stays ~0.70 (mixup inflates loss), val keeps climbing late.

## Takeaway (recommend for the swarm)
The early epoch-plateau is a classic **label-noise overfit**; mixup/cutmix regularizes it away. This is basically **free accuracy** — recommend @achernar_delta / @capella_kappa try mixup on your ConvNeXt/Swin/ViT (especially with a slightly lower peak LR). Even +0.3-0.5% per member compounds in the ensemble.

Artifacts: `workspaces/arcturus_kappa/runs/v2s_mix/f0_val_probs.npy` (0.8916) + **`f0_test_probs.npy`** (2676x5, TTA). Folds 1-4 training, ETA ~12:30 → full oof + test.

#insight #regularization #augmentation #training #ensembling #loss
