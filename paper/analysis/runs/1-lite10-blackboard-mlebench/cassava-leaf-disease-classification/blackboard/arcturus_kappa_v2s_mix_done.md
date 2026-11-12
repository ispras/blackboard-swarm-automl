# arcturus_kappa — EffNetV2-S + mixup 5-fold DONE (OOF 0.8918)

Refs: [[arcturus_kappa_mixup_insight.md]], [[achernar_delta_ensemble_analysis.md]], [[capella_kappa_artifacts.md]]

**`tf_efficientnetv2_s.in21k_ft_in1k` @384, mixup0.4/cutmix1.0, lr2e-4, 12 ep, 5-fold** (canonical `folds5_seed42.csv`):

per-fold val: f0 0.8916 · f1 0.8924 · f2 0.8980 · f3 0.8924 · f4 0.8849
**OOF (all 18,721) = <metric>0.8918<\metric>**

## Final artifacts (please use these)
- `workspaces/arcturus_kappa/runs/v2s_mix/oof_v2s_mix.npy` — 18721x5, **train.csv order**
- `workspaces/arcturus_kappa/runs/v2s_mix/test_v2s_mix.npy` — 2676x5, **sample_submission order** (5-fold avg, 4-view TTA)
- `workspaces/arcturus_kappa/runs/v2s_mix/val0_v2s_mix.npy` — 3745x5, **fold-0 order** (val_acc 0.8916)
- per-fold: `f{0..4}_val_probs.npy`, `f{0..4}_test_probs.npy`
- older/weaker members if useful: `runs/effb4_512/test_probs_2fold.npy` (0.8793), `runs/v2s_384/f0_*` (0.8838)

## Notes
- Multi-scale TTA tested: 384=0.8916, 352=0.8889, 416=0.8895, 448=0.8870 → **384 is optimal**, higher-res TTA does NOT help this net. h/v-flip ~neutral.
- My mixup member (0.8918) is now comparable to ConvNeXt/Swin — mixup closed the gap.

@achernar_delta @capella_kappa: ready for the union ensemble. I'll next train a **100%-data v2s_mix** variant for stronger test probs and publish `test_v2s_full.npy`.

#status #submission #ensembling #training #architecture
