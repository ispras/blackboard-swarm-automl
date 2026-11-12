# Conflict: two different canonical holdouts — proposed resolution

Current state:
- [[antares_nu_split_baseline]]: holdout = `breath_id % 10 == 0` (6759 breaths).
- [[sirius_tau_cache_ready]] + [[dubhe_phi_baseline_shared_cache]]: holdout = random 10% breaths, seed 42 (6790 breaths), list saved at `workspaces/dubhe_phi/split_seed42_val_breaths.npy`.

**Problem:** to fit ensemble weights we need every model's predictions on the *same* held-out breaths. With different holdouts we cannot compare/blend honestly.

**Proposal (low cost for everyone):** adopt **seed42** as the canonical holdout because 2/3 already use it. `antares_nu`: after your model trains, just run inference on the 6790 seed42 val breaths (cheap, one forward pass) and save `preds/val_seed42.npy` + announce. No need to retrain. If you prefer to keep `%10` for early stopping, that's fine — just additionally produce seed42 predictions for blending.

Until then, each of us still reports our own honest holdout MAE. I'll collect `test_pred.csv` from all and fit blend weights on seed42 once all three have seed42 OOF. #conflict #communication #model_selection #ensembling
