# arcturus_kappa — effb4_512 fold-0 done

Refs: [[arcturus_kappa_smoke_effb4.md]], [[capella_kappa_tools_endgame.md]]

EfficientNet-B4 @512, 12 ep, fold 0 (shared `folds5_seed42.csv`):

**fold-0 val accuracy `<metric>0.8793<\metric>`** (macro-F1 0.776). Train loss 1.01→0.47, val plateaued ~ep6-12 (val ~0.877-0.879) → 12 ep is about right; aug is quite strong (RRC 0.6-1.0).

Folds 1-4 still training (ETA ~13:20 under 3-job GPU contention). Will then:
1. Publish `runs/effb4_512/f0_val_probs.npy` (3745x5, fold-0 order) + `test_probs.npy` (2676x5, sample_submission order) + full `oof_probs.npy` (18721x5, train.csv order).
2. **Plan**: higher-res fine-tune @640 from these ckpts (2 ep, low LR) → likely +0.3-0.5% and adds scale diversity to the ensemble.

Note: strong aug (RRC 0.6-1.0) may be slightly underfitting; HR fine-tune should recover leaf detail.

#status #training #architecture #ensembling
