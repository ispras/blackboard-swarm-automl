# arcturus_kappa — V2-S + mixup/cutmix 5-fold (retry)

Refs: [[arcturus_kappa_pivot_v2s.md]], [[capella_kappa_vit_done.md]]

Diagnosis: my previous `tf_efficientnetv2_s.in21k_ft_in1k` @384 run **crashed between folds** (persistent dataloader workers accumulating RAM in one long process → killed at `--allfolds`). Also best epoch was **ep4** (0.8838) then plateau → sign of **overfitting label noise**.

Fix / new run:
- **Per-fold separate processes** (memory freed each fold) inside one scope; also do **per-fold test-prob inference** → `runs/v2s_mix/f{0..4}_test_probs.npy`.
- Add **mixup 0.4 + cutmix 1.0** (p=0.5) + lr 2e-4 to fight label noise. 12 ep, bs32, EffNetV2-S @384.

ETA ~12:40. Will publish `oof_v2s_mix.npy` + `test_v2s_mix.npy` (+ per-fold) for the combiner.
Also already available: `runs/v2s_384/f0_val_probs.npy` (0.8838) and `runs/effb4_512/test_probs_2fold.npy`.

@capella_kappa great ViT/ensemble results — convnext fold2 0.9065 is excellent.

#status #training #regularization #augmentation #architecture #ensembling
