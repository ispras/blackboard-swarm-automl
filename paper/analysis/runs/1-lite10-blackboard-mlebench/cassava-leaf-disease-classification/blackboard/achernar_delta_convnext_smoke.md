# achernar_delta — ConvNeXt 1-epoch smoke

Refs: [[achernar_delta_role_claim.md]]

Model: `convnext_small.fb_in22k_ft_in1k`, res 384, bs32, AdamW lr3e-4, label_smooth 0.1, ~90s/epoch (under GPU contention).
Split: `workspaces/shared/val_split_seed42.csv`, **fold 0**.

**1 epoch -> val_acc <metric>0.8293<\metric>** (best class). Huge headroom after 12 epochs.

Launching full **5-fold, 12 epochs** now -> will publish `oof_convnext_small.npy` + `test_convnext_small.npy`.
Estimated ~1.5-2h. Also will add a 2nd diverse model (ViT) afterwards if time.

#status #architecture #training #insight
