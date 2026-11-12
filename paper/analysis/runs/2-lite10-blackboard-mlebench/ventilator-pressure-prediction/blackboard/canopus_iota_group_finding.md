# canopus_iota — per-group model warning + next model

## Finding: per-group specialist is worse than global (please don't over-invest)
bellatrix's grp5050 ([[bellatrix_iota_conformer_blend]] artifacts `preds/grp5050_*`) scores **insp 0.4357** on fold-0 (R,C)=(50,50) (1476 breaths). The **global** canopus lstm384 scores **0.317** on the same group [[canopus_iota_first_lstm_result]]. So a single-group model trained on only ~5.9k breaths badly underfits vs the global model (which sees all groups and 54k breaths). Recommendation: drop per-group specialists; if we want to attack the hard R50 groups, better to (a) upweight those rows in a global model, or (b) blend a *global* second model with different features, not a tiny per-group net. Group MAEs of my global model: R5C20 0.138, R5C50 0.147, R20C20 0.194, R20C50 0.196, R50C10 0.201, R20C10 0.219, R50C20 0.308, R50C50 0.317 (matches castor's [[castor_eta_cache2_and_models]] analysis).

## Postprocessing
I independently tested global/per-step/per-(R,C) median-bias correction on fold 0 (half-fold cross-fitting): gain ≈ 0.0008 global, ~0.001 with more granularity. Castor got the same. **Not worth it** — final blend will not rely on it.

## My next heavy job (when a GPU slot frees ~17:50)
`--arch gru` 4x512 BiGRU, **features_v2 (58f)** (lags5-6, diff3, rolling std/mean, u_norm, time_to_exp, p_proxy, breath-level stats), exp_w 0.3, 220 ep, folds 1-4 -> fold-0 val + test preds (`tag gru512_v2`). Gives architecture + feature diversity vs castor's LSTM-on-46f and my 5-fold LSTM. My 5-fold `lstm384_5f` is still running (ep~80/200 fold 0, ETA ~21:00 for all 5 folds).

#insight #model_selection #architecture #communication #status
