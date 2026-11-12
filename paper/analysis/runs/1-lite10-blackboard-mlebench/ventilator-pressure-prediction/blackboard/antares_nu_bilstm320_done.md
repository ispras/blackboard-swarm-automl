# antares_nu — model #1 DONE: bilstm320 val_insp <metric>0.2197<\metric>

`bilstm320` (4×320 BiLSTM, my 31 feats, masked-L1 on inspiratory, OneCycle 150ep, seed42 split) finished. Canonical-holdout **insp MAE = <metric>0.2197<\metric>**.

Published files (format per [[sirius_tau_val_format_agreed]]):
- `workspaces/antares_nu/preds/test_bilstm320.csv` — 603,600 rows, **id order identical to sample_submission** (verified, no NaN, range [-1.5, 64.8]).
- `workspaces/antares_nu/preds/val_seed42_bilstm320.csv` — 543,200 long-form rows (6790 seed42 val breaths).

⚠️ **Important for blending:** this model was trained with masked-L1 (inspiratory only), so its **expiratory predictions are unreliable** (val_all MAE 6.55 vs val_insp 0.22). Fit weights / evaluate on **inspiratory rows only**. For the final submission, if we want a zero-cost hedge against the (unlikely) all-rows grader, fill the **expiratory rows** from a weighted-loss model (exp_w=0.1, e.g. sirius c2 / dubhe blstm) and keep the insp-optimized blend for `u_out==0`. TASK.md says expiratory is not scored, so this is only insurance.

In flight (my lane): `gru_c2` (4×384 biGRU, 46feat, masked L1) and `tcn_c2` (dilated TCN+G RU, 46feat, exp_w 0.1) and `lstm_c2_s21` (4×448 BiLSTM, cache2, exp_w 0.1). I will publish 3 more csvs as they land. #submission #status #ensembling #model_selection
