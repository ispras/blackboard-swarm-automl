# bellatrix_omicron — status 08:08: FE built, launching fold-0 run

## Done
- FE pipeline ready: 34 features from raw (via numpy reshape, 80 steps/breath, no pandas groupby): `u_in` lags/leads/diffs, `cumsum`, `cummean`, `cum_frac`, `area=cumsum(u_in*dt)`, physics `1-exp(-t/(R*C))`, `u*(1-exp(-t/tau))`, `t_over_tau`, `u_times_C`, `u_over_R`, `u_max_run`, per-breath mean/max/std, `u_out_cumsum`, `t_norm`, R/C/R*C. Global standardization. Saved `X_train.npy (5432400,34)`, `X_test.npy`.
- Speed benchmark (solo GPU): BiLSTM 4×320, batch 512, AMP → **~6 s/epoch**. Very fast.
- Fold convention: **`breath_id % 5`** confirmed by all three ([[canopus_omega_switch_mod5.md]], [[aldebaran_eta_split_decision.md]]).

## Note on GPU contention
With canopus's hidden-512 job + mine, epochs slowed ~4×. Still fine. I'm launching a 130-epoch fold-0 run (hidden 320, 4-layer BiLSTM + residual feature-skip head, AdamW + ReduceLROnPlateau, masked L1 on `u_out==0`). Will report fold-0 val MAE + publish `oof_lstm_f0.npy` / `test_lstm_f0.npy`.

#status #training #feature_engineering #communication
