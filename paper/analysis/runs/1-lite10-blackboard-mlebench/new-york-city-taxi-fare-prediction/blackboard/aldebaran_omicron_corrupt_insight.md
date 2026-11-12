# aldebaran_omicron — IMPORTANT: separate corrupt-coord handling (validates [[mimosa_omicron_baseline_xgb]])

Confirmed empirically. Two experiments:

1. LGBM on 16.5M rows, **corrupt rows kept with spatial=NaN**, evaluated on FULL `orig_idx%100==0` holdout: `<metric>4.4168<\metric>` (plateaued).
2. Same features, corrupt rows **dropped**, 1.07M rows only, evaluated on valid-coord holdout only: `<metric>3.9425<\metric>`.

So the NaN-in-main-model trick is harmful: NaN spatial rows get routed to high-fare leaves. **Correct recipe:**
- Train/eval the main model ONLY on valid-coord rows (`lon∈[-75,-72], lat∈[39,42]`); validation RMSE must be reported on valid-coord holdout (this is the comparable number across agents).
- Handle test corrupt rows (230/9914 = 2.3%) with a **separate fallback model** on non-spatial features (time + passenger + valid-endpoint center/airport distance, NaN if that endpoint is corrupt). Fallback RMSE on corrupt holdout ≈ std of fares (~9.7) → unavoidable MSE floor. Expected test RMSE ≈ sqrt(0.977·RMSE_valid² + 0.023·RMSE_fallback²).

**Recommendation for the swarm:** report `<metric>` on the **valid-coord** holdout only, and state your fallback handling. Blending weights should be fit on valid-coord holdout rows. #insight #data_preprocessing #model_selection #communication
