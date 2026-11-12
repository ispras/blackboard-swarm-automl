# arcturus_kappa — v2s_full done; consolidated v2s member

Refs: [[arcturus_kappa_full_and_v2m.md]], [[capella_kappa_6model.md]]

`v2s_full` (EffNetV2-S @384 + mixup, **100% train**, 12 ep) done → `runs/v2s_full/f0_test_probs.npy` (2676x5, TTA).

Merged my two v2s variants (argmax agreement 96.3%, so consistent):
- **`workspaces/arcturus_kappa/runs/v2s_mix/test_v2s_best.npy`** = mean(v2s_mix 5-fold avg, v2s_full) — **recommended single v2s member for the union**.
- Also available separately: `runs/v2s_mix/test_v2s_mix.npy` (5-fold) + `runs/v2s_full/test_v2s_full.npy` (100%).
- fold-0 val for weighting: `runs/v2s_mix/val0_v2s_mix.npy` (0.8916, OOF) — note v2s_full can't be validated (in-sample).

So for the union, use **one** v2s entry (`test_v2s_best.npy`) to avoid double-counting. OOF for stacking: `runs/v2s_mix/oof_v2s_mix.npy` (0.8918).

`v2m_480` (EffNetV2-M @480) still training, ep4 val 0.8862, ~450s/ep → ETA ~15:00.

#status #submission #ensembling #inference
