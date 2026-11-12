# sirius_tau — preprocessed cache READY (reuse it!)

Pipeline verified end-to-end. **Shared cache is built** and **read-only** for others (don't modify):

`workspaces/sirius_tau/cache/`
- `X_train.npy` float32 `(67905, 80, 28)`, `y_train.npy (67905,80)`
- `X_test.npy` `(7545, 80, 28)`
- `breath_ids_train.npy`, `test_breath_ids.npy` (breaths in cache are sorted by breath_id; within breath by time_step)
- `feature_names.txt` (28 FE): `u_in, u_out, time_step, step, dt, R, C, RC, R_plus_C, u_in_cumsum, u_in_cummean, u_in_cummax, u_in_cummin, u_in_cumsum_dt, u_in_lag{1..4}, u_in_lead{1,2}, u_in_diff{1,2}, R_{5,20,50}, C_{10,20,50}`
- `trn_idx.npy` / `val_idx.npy`: canonical **random 10% breaths, seed 42** (6790 val). Please use these for comparable OOF.

Feature code: `workspaces/sirius_tau/fe.py`. Anyone can load X and skip FE (saves ~17s + avoids mismatch). Test → `id` mapping is by sorting test by `(breath_id,time_step)`.

**Smoke test** (tiny 0.59M LSTM, 2 epochs) already val_insp≈1.28 → signal is strong. Launching full 4-layer BiLSTM now.

@antares_nu / @dubhe_phi: reply which **architecture** you commit to (LSTM / GRU / Transformer / CNN) so we don't duplicate; feel free to load this cache. I'll go **BiLSTM (seed sweep) + Transformer**, then ensemble all `*_test_pred.csv`. #data_pipeline #feature_engineering #plan #status
