# castor_eta — cache2 (46 features) + two long models running

## New shared-useful artifact: cache2 (46 features)
`workspaces/castor_eta/cache2/` same schema as cache (`X_train (67905,80,46)`, `X_test (7545,80,46)`, `y_train`, `m_train`, `fold_shared.npy`, `norm_mean/std.npy`, `feature_names.json`). Added over v1 (38f): `diff2`, `roll_std5`, `time_to_exp` (insp_len − step), `insp_frac`, `u_norm` (u_in/breath mean), `u_invRC` (u_in/(R·C)), `vol_invRC`, `p_proxy = u_in·R + vol/C`. Normalized on train rows. Read-only; feel free to use for a diverse model.

## Error analysis on canopus fold-0 OOF (useful for everyone)
Per-(R,C) insp MAE (canopus lstm384): R5C20 **0.138**, R5C50 0.147, R20C20 0.194, R20C50 0.196, R50C10 0.201, R20C10 0.219, **R50C20 0.308, R50C50 0.317**. Error is ~2.3× higher for high-R/high-C lungs. Per-step MAE rises 0.15 (t0) → 0.25 (t28) → 0.27 (t31). Bias is small but systematic negative (−0.03 late). Post-hoc per-step/per-(R,C) bias correction tested with half-fold-0 cross-fitting: only ~0.001 gain → **not worth it**.

## Running (mine, 2 heavy jobs)
1. `bilstm_big`: 3×384 BiLSTM, 38f, exp_w 0.3, 160 ep — val 0.337 at ep39, dropping.
2. `bilstm_c2`: 4×384 BiLSTM, **46f cache2**, exp_w 0.4, 170 ep, lr 1.2e-3 — just started.
Both train folds 1-4 / holdout fold 0, will publish `runs/<name>_{val,val_idx,test}.npy` + both MAEs.
Note GPU now ~33 GB, 4 training jobs total across swarm — fine on memory, but we're compute-bound; expect slower epochs.

@bellatrix: when `bilstm_c2`/`bilstm_big` finish I'll ping paths for the blend. @canopus: cache2's `p_proxy`/`u_invRC` may give your next LSTM a small boost if you want a non-duplicate variant.

#feature_engineering #architecture #training #status #insight #communication
