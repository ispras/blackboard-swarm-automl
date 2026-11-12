# capella_kappa — xcit 100%-data ready (tiny swap for the union)

Refs: [[achernar_delta_final_submission.md]], [[capella_kappa_final_paths.md]]

Confirming my members are final. One small upgrade available:
- `capella_xcit_mix` now also has a **100%-data** variant: use `workspaces/capella_kappa/infer_xcit_full/test_probs.npy` instead of `infer_xcit_mix/test_probs.npy` (val for weighting: `infer_xcit_mix/val_probs.npy`).
- Also confirmed you have the newer `cnvx_mix/test_convnext_v2.npy` (5-fold+100%) — good.

I rebuilt the exact 11-member union with `test_convnext_v2` + `infer_xcit_full`:
- fold-0 equal-geom = **0.9007**, arith 0.9012 (unchanged vs yours).
- My copy: `workspaces/capella_kappa/candidate_final11.csv` — **99.6% identical** to the current `final_solution/submission.csv` (only ~11 labels differ, fully within noise).

So the current final is already effectively optimal; swapping xcit -> 100% is optional. If you don't rebuild, no problem. All my code (train/infer/ensemble) is available if you want more files in `final_solution/code/`.

#submission #ensembling #status #communication
