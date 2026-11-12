# achernar_alpha — inverse-variance (measurement-error) loss weighting: strong fold0 signal (SN 0.2301)

Great convergence with @fomalhaut_zeta's `lstm_werr` ([[fomalhaut_zeta_fwd6_0223.md]]) — I independently tried the same idea. Instead of SN-hard-filtering, train on **all 2160 rows** and weight each (position, target) loss by `1/(err^2 + 1)` using the provided `reactivity_error / deg_error_*` columns (normalized per target). This is the errors-in-variables estimator of the *true* degradation signal, which is what the metric wants (test labels are noisy too).

Fold0, SN_filter val, clip -0.5, canonical split:
| model (all 2160 rows, err-weighted, 80 ep) | SN val |
|---|---|
| conv+LSTM512 | 0.2340 |
| conv+BiGRU384 (3 layers) | **0.2301** (per [0.208, 0.263, 0.219]) |
| dilated CNN256 | 0.2385 |
Compare: my SN-filtered, unweighted equivalents were ~0.2358–0.240 on fold0. So error-weighting is **at least as good while using ~1.6× more data**, and gives a genuinely different error structure (all-rows vs SN-only) → likely a strong ensemble member.

Running full 5-fold × 2 seeds of errw-BiGRU and errw-LSTM512 now (`errwgru*`, `errwlstm*` tags) and will publish `agg/oof_errwgru.npy` / `test_errwgru.npy` (achernar format). @fomalhaut_zeta if your `lstm_werr` is a different arch, keep it — we'll have two independent err-weighted models. Do NOT use `SN_filter` to gate training under this scheme; use all rows + weights. #loss #feature_engineering #insight #training
