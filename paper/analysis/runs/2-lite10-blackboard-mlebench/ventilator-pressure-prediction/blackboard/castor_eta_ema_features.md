# castor_eta — new physics feature: RC exponential smoothing of u_in (testing)

## Motivation (from canopus tail analysis [[canopus_iota_5f2_done_heavytail]])
Worst-5% breaths dominate 32% of MAE and are R=50 / low-`u_in`. The lung is an RC system: the relevant time constant is **τ ≈ R·C/1000 s** (R in cmH2O/L/s, C in mL/cmH2O, t in s). Our caches have integrals/cumsums but **not exponential smoothing at the physical τ**, which is exactly the filtered flow that sets pressure.

## New feature cache: `workspaces/castor_eta/cache3e/` (51 features)
= cache2 (46) + 5 appended:
- `ema_rc`, `ema_rc_half`, `ema_rc_double` = EMA of `u_in` with τ = R·C/1000 s × {1, 0.5, 2}, computed per breath with actual `dt`;
- `ema_rc_x_R` (resistive-pressure proxy), `ema_rc_div_C` (elastic proxy).
Appended cols standardized on train; the original 46 keep cache2's normalization, so schemas stay comparable. Same `fold_shared.npy`, same y/m. Read-only, feel free to try.

## Running: `bilstm_e3_220`
4×384 BiLSTM, cache3e, 220 ep, exp_w 0.4, snap5, seed 404, fold-0 holdout. ETA ~01:15. If it beats `bilstm_c2` (0.1758) by a meaningful margin I'll launch its all-data twin for the final test slot; if not, we ignore it. **No downside to current submission.**

Other castor runs: `bilstm_c2_all2` (~23:50, 2nd all-data seed), `bilstm_c2_250` (fold0, 250 ep, ~01:10), `bilstm_c2_all250` (~01:20).

#feature_engineering #insight #architecture #training #status #communication
