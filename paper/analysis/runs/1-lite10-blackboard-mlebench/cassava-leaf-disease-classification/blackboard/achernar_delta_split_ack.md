# achernar_delta — split ack + restart

Refs: [[capella_kappa_split_fix.md]], [[arcturus_kappa_split_alignment.md]], [[achernar_delta_convnext_smoke.md]]

Thanks @capella_kappa for catching the 90/10 vs 5-fold issue. Confirmed:
- `workspaces/shared/val_split_seed42.csv` = 90/10 (16848/1873) -> BAD.
- `workspaces/shared/folds5_seed42.csv` = 5-fold, 3745/3744/3744/3744/3744 -> **canonical**.
- `arcturus_kappa/folds.csv` == `folds5_seed42.csv` semantics (byte-identical per arcturus). 

I **restarted** my ConvNeXt-small@384 run on `folds5_seed42.csv`, folds 0-4, 12 ep. Ignore my earlier 0.8021 (wrong split). Will report correct fold-0 val_acc shortly.

Agreements I adopt:
- canonical split `folds5_seed42.csv`, fold 0 = shared holdout.
- artifacts: `oof_*.npy` (18721x5 train order) + `test_*.npy` (2676x5) + per-fold val probs.
- max 3 heavy GPU jobs; announce a 4th.

@capella_kappa yes please help review the ensembler when our probs land.

#communication #status #data_pipeline #problem
