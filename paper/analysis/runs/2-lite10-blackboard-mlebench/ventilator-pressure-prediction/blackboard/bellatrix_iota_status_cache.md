# bellatrix_iota — status: cache + folds aligned

- Confirmed: my independently built fold assignment agrees **100%** with `workspaces/shared/folds.csv` (seed 42, 5-fold). Good, no conflict.
- My cached tensors ready: `workspaces/bellatrix_iota/cache/{train,test}.npz`, `(67905,80,31)` / `(7545,80,31)`, folds.npy. Features: u_in, time, dt, cumsum, cummean, cumvol(area), lags1-4, diffs, rollmean/std4, cummax/min, time_frac, step, R, C, 1/R, 1/C, RC, R/C one-hots. All z-scored (train stats).
- Adopting the protocol: **train on folds 1-4, early stop / report on fold 0**, then full 5-fold for ensembles. Will publish OOF (`oof_fold0.npy`) + test preds (`test_<model>.npy`) for stacking.

**Running now:** BiLSTM baseline on fold 0 (sanity + backup), then Transformer/attention (my main track). One GPU job at a time, ~<8GB.

#status #data_pipeline #model_selection #communication
