# canopus_iota — metric semantics + status

## Metric decision (important for everyone)
[[TASK.md]] and `data/README.md` both say score = MAE **during inspiratory phase** (`u_out==0`). Kaggle evidence: public notebooks trained only on inspiratory rows and still got LB ~0.13-0.16, which would be impossible if all 80 steps were scored. Conclusion: **official metric almost certainly masks expiratory rows**. So:
- **Primary selection metric = inspiratory MAE** (report as `<metric>`).
- But keep a small expiratory weight (exp_w≈0.2-0.3) in the loss as a cheap hedge in case the grader averages all rows; it costs little inspiratory accuracy. castor_eta already uses 0.2 — good. I'll use 0.3.

## Status
- BiLSTM (29 shared features, hidden 384, 3 layers, weighted L1) training on folds 1-4 / early stop fold 0. Will publish `workspaces/canopus_iota/oof_lstm384.npy` (full-length with NaN outside fold0) + `test_pred_lstm384.npy` + fold0 insp MAE.
- castor_eta: CNN+BiGRU, fold0 val insp MAE 0.676 at ep16 and dropping [[castor_eta_cache_and_first_run]].
- bellatrix_iota: BiLSTM sanity + Transformer main track [[bellatrix_iota_status_cache]].

## Blending plan (for bellatrix, owner of blend)
Since these are 3 different architectures, a simple weighted average of **inspiratory-phase predictions only** should give a solid gain. I'll provide a short script `workspaces/canopus_iota/blend_eval.py` that loads any set of `val preds + val_y + u_out` arrays and reports fold-0 inspiratory MAE + non-negative weights. Ping me if you'd rather own it.

#model_selection #ensembling #communication #status
