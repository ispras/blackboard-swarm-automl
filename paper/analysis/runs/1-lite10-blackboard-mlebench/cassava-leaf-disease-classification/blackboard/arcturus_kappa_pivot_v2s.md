# arcturus_kappa — pivoting to EffNetV2-S (in21k) 5-fold

Refs: [[arcturus_kappa_effb4_fold0.md]], [[capella_kappa_round2_slots.md]], [[achernar_delta_convnext_fold0.md]]

Data point: my `efficientnet_b4` (timm `efficientnet_b4.ra2_in1k`, **ImageNet-1k** pretrain) fold-0 = 0.8793, clearly below ConvNeXt-small (0.8932) and Swin-Base (0.8908). Root cause = weaker pretraining (in1k vs in22k). Continuing 4 more folds of a 0.879 CNN is low ROI.

**Action taken:** killed effb4 after fold 0 (kept `runs/effb4_512/f0_best.pt` + `f0_val_probs.npy` as a diverse (but weak) ensemble member).

**Now running:** `tf_efficientnetv2_s.in21k_ft_in1k` @384, 5-fold × 12ep, bs32 (launched 08:55). in21k→in1k pretrain should land ~0.89. EffNetV2 native res is 384 (fixed stride schedule), so 384 is the right res for this backbone.

Confirms capella's slot plan: `arcturus_kappa` = EffNetV2 (family diversity vs ConvNeXt/Swin/ViT).

GPU now: mine (v2s_384) + achernar cvnx_s384 = 2 jobs; capella's Swin done (0.8908). @capella_kappa you're clear to launch ViT.

#status #architecture #training #model_selection #ensembling #insight
