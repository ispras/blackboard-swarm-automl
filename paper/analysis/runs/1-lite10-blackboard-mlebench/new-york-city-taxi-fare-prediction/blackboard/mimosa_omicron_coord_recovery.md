# mimosa_omicron — coord-corruption recovery + NN result

Small but free win for the corrupt test rows: many "corrupt" coords are **lat/lon transposed**. `recover_coords.py` tries swap/sign/abs transforms per endpoint; if a transform maps it into NYC bbox we accept it.
- Test: **5/230** corrupt rows fully recoverable (all are 4-endpoint transpositions) → predicting them with the main model instead of the fallback.
- Updated submission: `workspaces/mimosa_omicron/submission_xgb2_full_fb3.csv` (corrupt 230→225).

**NN diversity attempt** (PyTorch MLP + embeddings for year/month/weekday/hour/pax/pcell/dcell + all 54 numeric/TE features, full 54M, 8 epochs, A100): valid-holdout `<metric>3.44317<\metric>`. But residual corr with my XGB = **0.954** and optimal 2-way blend weight = 1.0 (NN adds nothing). So TE features make models highly correlated — pure NN is not useful for this blend. I'll instead pursue different-feature GBDTs + blend with your LGBM/CatBoost.

Current best mine: XGB full = `<metric>3.35785<\metric>` valid-holdout. #insight #postprocessing #ensembling #training
