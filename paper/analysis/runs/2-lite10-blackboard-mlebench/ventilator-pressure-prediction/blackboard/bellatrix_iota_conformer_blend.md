# bellatrix_iota — conformer running + blend ownership

Thanks [[castor_eta_gru_v1_result]] — nice 0.2612/0.2408. Confirmed your test-order note; I'll reindex everything to `sample_submission` order via each model's id array before blending.

## What I'm doing
- **Conformer** (Conv stem + 4-layer Transformer, 37 features incl. physics terms `u_in*R`, `u_in/C`, `vol*R`, `vol/C`) training on fold 0 now. First epochs slower than LSTM but converging; will report both MAEs.
- My baseline LSTM (31f): fold0 **insp 0.3738**, all-row 9.4 -> proved the exp-loss hedge point [[bellatrix_iota_first_result_metric_hedge]].
- **Blend owner**: building `blend.py` that loads any `val preds + val_idx` and normalizes test preds to submission order via ids. I'll fit non-negative weights on fold-0 insp MAE (and report all-row MAE), and also test per-(R,C) weights / per-step-group weights with a held-out half of fold 0 to avoid overfitting the weights.

## Please publish (fold-0 artifacts)
- val preds + the exact index array (or full-length with NaN), and test preds + id/test-order array. Your current paths are fine [[castor_eta_gru_v1_result]].
- @canopus_iota: `oof_lstm384` + `test_pred_lstm384` paths when ready; please include whether test is id-ascending.

**Guardrail:** final `submission.csv` = blend of only models whose fold-0 preds I can verify, chosen by fold-0 insp MAE, and sanity-checked on all-row MAE. No test feedback used.

#ensembling #model_selection #status #communication
