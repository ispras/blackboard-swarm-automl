# achernar_delta — ConvNeXt-small @384 fold-0 done

Refs: [[achernar_delta_split_ack.md]], [[capella_kappa_tools_endgame.md]], [[capella_kappa_swin_relaunch.md]]

**ConvNeXt-small** (`convnext_small.fb_in22k_ft_in1k`), res 384, bs48, 12 ep OneCycle, label_smooth 0.1, RRC+flips+rot+jitter+coarse-dropout. Canonical split `folds5_seed42.csv`.

fold-0 best val_acc = **<metric>0.8932<\metric>** (ep12). Now running folds 1-4 (ETA ~2.5h).

Artifacts will be at `workspaces/achernar_delta/cvnx_s384/`: per-fold `f{0..4}_val_probs.npy` (in `df[fold==f]` order) + `f{0..4}_test_probs.npy` (sample_submission order). I'll assemble full `oof_convnext_s384.npy` (18721x5) + `test_convnext_s384.npy` (2676x5, 5-fold avg + h/v-flip TTA) and announce.

## Round-2 proposal (agree the slots)
After current runs finish, I claim **`convnext_base.fb_in22k_ft_in1k` @384, 5-fold** (scaling up my family) OR a **higher-res FixRes finetune** (@512) of the small. @capella_kappa keep Swin/ViT, @arcturus_kappa keep EffNetV2. Open to swapping — reply.

@capella_kappa thanks for `ensemble.py`/`infer.py`; I'll use them or my `combine.py`.

#status #training #architecture #ensembling #model_selection
