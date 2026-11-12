# arcturus_kappa — effb4 fold-0/1 test probs now available

Refs: [[capella_kappa_swin_done_ensemble.md]], [[arcturus_kappa_pivot_v2s.md]]

@capella_kappa: noted — test probs were missing for effb4. Now saved (I had finished folds 0 AND 1 before killing):
- fold-0 val probs: `workspaces/arcturus_kappa/runs/effb4_512/f0_val_probs.npy` (3745x5, `df[fold==0]` order) — val_acc 0.8793
- test probs (h/v-flip TTA, avg of fold0+fold1 ckpts): `workspaces/arcturus_kappa/runs/effb4_512/test_probs_2fold.npy` (2676x5, sample_submission order)
- full `oof_probs.npy` (folds 0-1 coverage) at `runs/effb4_512/oof_effb4_2fold.npy`

EffNetV2-S @384 fold-0 in progress: ep4 val 0.8838, ep6 0.8801 (still improving). ETA all folds ~11:30.

#status #ensembling #inference #submission
