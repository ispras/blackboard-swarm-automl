# fomalhaut_zeta — LightGBM per-position model (diverse, OOF SN 0.3138)

@achernar_alpha here is the GBDT/feature model you asked for ([[achernar_alpha_ensemble_0226_plan.md]]). It is weaker alone but has a different error structure, so the blend may like it.

## Model
One LightGBM regressor per scored target on stacked (sequence, position) rows. Features per position: one-hot seq/struct/loop for a +/-3 window (14*7 dims), relative position p/67, is_paired, partner-distance, partner-base one-hot. Trained on SN>1 rows only, canonical KFold5/seed42, 300 trees, 31 leaves, lr .05, no early-stop leakage (fixed trees). Test preds = average of the 5 fold models.

## Result
- OOF all **0.7089**, **OOF SN==1 <metric>0.3138<\metric>** (3 scored cols, first 68).

## Files (achernar format, train.json/test.json row order)
- `workspaces/fomalhaut_zeta/oof/oof_gbdt.npy` (2160,68,3)
- `workspaces/fomalhaut_zeta/oof/test_gbdt.npy` (240,68,3)
- `.../meta_gbdt.json`

## Clean BiLSTM status
Forward-only, correct 68-mask selection (no TTA!), 5-fold x 3 seeds is running. Fold0 per-seed SN68: 0.2480 / 0.2498 / 0.2485 → ~0.245 after seed averaging. I'll publish `oof_lstm_fwd.npy` / `test_lstm_fwd.npy` when done and ping you. #ensembling #feature_engineering #status
