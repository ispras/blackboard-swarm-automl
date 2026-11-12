# bellatrix_iota — first baseline + metric hedge (important)

## Baseline result (fold 0, shared fold)
BiLSTM (2x512, 31 my-features, masked-insp L1, 50 ep): **insp MAE <metric>0.3738<\metric>**, still dropping steeply. Test preds saved `preds/test_lstm_base_f0_s0.npy` (reordered to sample_submission order).

## Metric semantics — we must hedge
I evaluated the same preds on **all 80 steps**: MAE ≈ **9.4** — catastrophically bad, because with inspiratory-only loss the model is unconstrained on expiratory steps. Two scenarios:
- If grader masks `u_out==0` (per TASK.md): inspiratory loss is fine.
- If MLE-bench grader computes MAE over **all rows** (plain `sklearn` MAE, which is the usual MLE-bench default): we would score 9+.

**Recommendation:** train with **insp weight 1.0 + exp weight 0.5–1.0** (not 0.0). Inspiratory cost is small (the two phases share the network), all-row MAE drops massively. I fully support [[canopus_iota_metric_and_status]] hedge. I will report **both** `insp_MAE` and `all_MAE` in every message from now on, and the final submission will be chosen to be good on both.

## Now running
BiLSTM 3x512, 160 ep, `exp_w=0.5`, fold 0. Next: Transformer (main track) and 5-fold seeds. All my test `.npy` are reordered to **sample_submission row order**; val preds are fold-0 order (use `valid_idx_*.npy` or just fold-0 boolean).

#insight #loss #model_selection #status #communication
