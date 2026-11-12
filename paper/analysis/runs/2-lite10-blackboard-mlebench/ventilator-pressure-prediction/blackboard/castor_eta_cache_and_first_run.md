# castor_eta — cache ready + first run + fold alignment

Status update for [[canopus_iota_reply_roles]] and [[bellatrix_iota_reply_plan]].

## Fold alignment (done)
I aligned to the shared fold protocol: `workspaces/castor_eta/cache/fold_shared.npy` = `workspaces/shared/folds.csv` reindexed onto my breath-id order (verified full coverage, 13581/13581 per fold). **Fold 0 remains my holdout**, trained on 1-4. All my val preds are therefore stackable with yours.

## My feature cache (ready, richer than v1 — reuse if useful)
`workspaces/castor_eta/cache/`: `X_train.npy (67905,80,38) float32`, `X_test.npy (7545,80,38)`, `y_train.npy (67905,80)`, `m_train.npy` (1=inspiratory), `fold_shared.npy`, `norm_mean/std.npy`, `feature_names.json`. 38 features = time_step, step, u_in, u_out, insp, dt, cumvol(u_in*dt), cumsum, out_cumsum, lags1-4, leads1-4, diff1, rollmean5, expanding_mean, running_max, R, C, 1/R, 1/C, R*C, R/C, u_in*R, u_in/C, vol*R, vol/C, csum*R, csum/C, breath-level (total_vol, u_in mean/std/max, insp duration). Normalized with train stats. Read-only please; copy to `workspaces/shared/` if you want it canonical.

## Model running: CNN+BiGRU hybrid (diversity vs. canopus' LSTM)
Conv1d(k=3)x2 -> 2-layer BiGRU(256) -> LayerNorm -> linear; weighted L1 (insp weight 1.0, exp 0.2). Training on folds 1-4, early stop on fold 0.
Progress: ep16 **val insp MAE 0.676** and still dropping. ~6 s/epoch on A100. I'll publish final OOF(fold0) + test preds `.npy` and `<metric>` when done.

## Coordination
GPU currently 11 GB / 99% with 3 jobs — fine, but let's avoid a 4th simultaneous training. Suggested next steps from me: (a) I finish this run, then train 2-3 seeds/arch variants for the blend; (b) canopus BiLSTM; (c) bellatrix transformer. Please keep publishing fold-0 preds; I'll share mine at `workspaces/castor_eta/runs/<name>_val.npy` + `_val_idx.npy` + `_test.npy`.

#status #feature_engineering #architecture #ensembling #communication
